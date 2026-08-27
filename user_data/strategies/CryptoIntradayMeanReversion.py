"""
CryptoIntradayMeanReversion Strategy
Systematic Crypto Intraday Mean-Reversion Strategy based on:
1. Price Z-Score Extreme Reversions (Z_ENTRY_THRESHOLD)
2. Rolling R/S Hurst Exponent Regime Filter (H < HURST_MAX for anti-persistent mean reversion)
3. Trailing Volatility Regime Filter (ATR Percentile Rank between ATR_PCTL_MIN and ATR_PCTL_MAX)
4. Volume Z-Score Confirmation (Volume_Z >= VOLUME_Z_THRESHOLD)
5. Asymmetric Dual-Anchor Take Profit (MIN/MAX of SMA Mean-Reversion and R-Multiple)
6. Dynamic Risk-Based Position Sizing and 9-Candle Time Stops (<0.5R)
"""

from datetime import datetime
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Trade
from freqtrade.strategy import (
    DecimalParameter,
    IStrategy,
    IntParameter,
    stoploss_from_open,
)


def calc_rolling_hurst(series: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling Hurst exponent via Rescaled Range (R/S) analysis on log returns"""
    log_ret = np.log(series / series.shift(1)).fillna(0.0).to_numpy(dtype=np.float64)
    n = len(log_ret)
    hurst = np.full(n, 0.5, dtype=np.float64)
    if window < 10 or n < window:
        return pd.Series(hurst, index=series.index)

    windows = sliding_window_view(log_ret, window_shape=window)
    means = np.mean(windows, axis=1, keepdims=True)
    devs = windows - means
    cum_devs = np.cumsum(devs, axis=1)
    ranges = np.max(cum_devs, axis=1) - np.min(cum_devs, axis=1)
    stds = np.std(windows, axis=1)

    stds_safe = np.where(stds > 1e-12, stds, 1.0)
    rs = np.where((stds > 1e-12) & (ranges > 1e-12), ranges / stds_safe, 1.0)
    rs_safe = np.where(rs > 0, rs, 1.0)

    log_denom = np.log(window / 2.0)
    if log_denom <= 0:
        log_denom = np.log(float(window))

    h = np.log(rs_safe) / log_denom
    h = np.clip(h, 0.0, 1.0)
    hurst[window - 1 :] = h

    return pd.Series(hurst, index=series.index)


def calc_rolling_percentile_rank(series: pd.Series, window: int) -> pd.Series:
    """Vectorized rolling percentile rank of trailing window"""
    values = series.to_numpy(dtype=np.float64)
    n = len(values)
    pctl = np.full(n, 50.0, dtype=np.float64)
    if window < 2 or n < window:
        return pd.Series(pctl, index=series.index)

    windows = sliding_window_view(values, window_shape=window)
    cur_vals = windows[:, -1:]
    ranks = np.sum(windows <= cur_vals, axis=1)
    pctl[window - 1 :] = (ranks / float(window)) * 100.0

    return pd.Series(pctl, index=series.index)


class CryptoIntradayMeanReversion(IStrategy):
    """
    Crypto Intraday Mean-Reversion Strategy
    """

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "5m"

    minimal_roi = {"0": 10.0}
    stoploss = -0.10
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count: int = 150

    # -------------------------------------------------------------
    # Hyperopt Parameter Spaces
    # -------------------------------------------------------------
    # Fixed Indicator Windows (evaluated once in populate_indicators)
    z_window = IntParameter(10, 40, default=20, space="buy", optimize=False)
    hurst_window = IntParameter(50, 150, default=100, space="buy", optimize=False)
    atr_window = IntParameter(10, 24, default=14, space="buy", optimize=False)
    atr_pctl_window = IntParameter(50, 150, default=100, space="buy", optimize=False)
    vol_window = IntParameter(10, 30, default=20, space="buy", optimize=False)

    # Dynamic Entry Spaces (optimized every epoch in populate_entry_trend)
    z_entry_threshold = DecimalParameter(1.2, 2.8, default=1.8, decimals=2, space="buy", optimize=True)
    hurst_max = DecimalParameter(0.45, 0.65, default=0.55, decimals=2, space="buy", optimize=True)
    atr_pctl_min = DecimalParameter(5.0, 40.0, default=20.0, decimals=1, space="buy", optimize=True)
    atr_pctl_max = DecimalParameter(60.0, 95.0, default=85.0, decimals=1, space="buy", optimize=True)
    volume_z_threshold = DecimalParameter(0.0, 2.0, default=0.8, decimals=2, space="buy", optimize=True)

    # Sell / Exit Spaces
    sl_atr_mult = DecimalParameter(1.0, 3.5, default=1.8, decimals=2, space="sell", optimize=True)
    tp_r_mult = DecimalParameter(1.2, 4.0, default=2.2, decimals=2, space="sell", optimize=True)
    time_stop_candles = IntParameter(10, 60, default=24, space="sell", optimize=True)
    risk_pct_per_trade = DecimalParameter(0.002, 0.02, default=0.005, decimals=3, space="sell", optimize=False)

    plot_config = {
        "main_plot": {
            "mean_close": {"color": "#ffd600"},
        },
        "subplots": {
            "Z-Scores": {
                "price_z": {"color": "#00e5ff"},
                "volume_z": {"color": "#ff007f"},
            },
            "Hurst & ATR Pctl": {
                "hurst": {"color": "#00e676"},
                "atr_pctl": {"color": "#ab47bc"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate Price Z-Score, Hurst Exponent, ATR Percentile, and Volume Z-Score"""
        z_win = int(self.z_window.value)
        h_win = int(self.hurst_window.value)
        atr_win = int(self.atr_window.value)
        atr_pctl_win = int(self.atr_pctl_window.value)
        v_win = int(self.vol_window.value)

        # 1. Price Z-Score
        dataframe["mean_close"] = ta.SMA(dataframe["close"], timeperiod=z_win)
        dataframe["std_close"] = dataframe["close"].rolling(z_win).std()
        dataframe["price_z"] = (dataframe["close"] - dataframe["mean_close"]) / dataframe["std_close"]

        # 2. Rolling Hurst Exponent via R/S Analysis
        dataframe["hurst"] = calc_rolling_hurst(dataframe["close"], h_win)

        # 3. ATR and Trailing Percentile Rank
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=atr_win)
        dataframe["atr_pctl"] = calc_rolling_percentile_rank(dataframe["atr"].fillna(0.0), atr_pctl_win)

        # 4. Volume Z-Score
        dataframe["mean_vol"] = ta.SMA(dataframe["volume"], timeperiod=v_win)
        dataframe["std_vol"] = dataframe["volume"].rolling(v_win).std()
        dataframe["volume_z"] = (dataframe["volume"] - dataframe["mean_vol"]) / dataframe["std_vol"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Trigger entries on extreme Z-Score + Mean-Reverting Hurst + Volatility Filter + Volume Spike"""
        z_thresh = self.z_entry_threshold.value
        h_max = self.hurst_max.value
        atr_min = self.atr_pctl_min.value
        atr_max = self.atr_pctl_max.value
        vol_z_thresh = self.volume_z_threshold.value

        # LONG Entry Rules
        long_condition = (
            (dataframe["price_z"] <= -z_thresh)
            & (dataframe["hurst"] < h_max)
            & (dataframe["atr_pctl"] >= atr_min)
            & (dataframe["atr_pctl"] <= atr_max)
            & (dataframe["volume_z"] >= vol_z_thresh)
        )
        dataframe.loc[long_condition & ~long_condition.shift(1).fillna(False), "enter_long"] = 1

        # SHORT Entry Rules
        short_condition = (
            (dataframe["price_z"] >= z_thresh)
            & (dataframe["hurst"] < h_max)
            & (dataframe["atr_pctl"] >= atr_min)
            & (dataframe["atr_pctl"] <= atr_max)
            & (dataframe["volume_z"] >= vol_z_thresh)
        )
        dataframe.loc[short_condition & ~short_condition.shift(1).fillna(False), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        """
        Stop Loss:
        LONG:  SL = Entry - (SL_ATR_MULT * ATR)
        SHORT: SL = Entry + (SL_ATR_MULT * ATR)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        entry_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        entry_atr = entry_candle["atr"].iloc[-1] if not entry_candle.empty else dataframe["atr"].iloc[-1]

        if np.isnan(entry_atr) or entry_atr <= 0:
            return None

        sl_dist = float(self.sl_atr_mult.value) * entry_atr
        sl_ratio = sl_dist / trade.open_rate

        return stoploss_from_open(-sl_ratio, current_profit, is_short=trade.is_short, leverage=trade.leverage)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> str | None:
        """
        Take Profit & Time Stop Exit Rules:
        LONG:  TP = MIN(TP_meanrevert, TP_Rmultiple)
        SHORT: TP = MAX(TP_meanrevert, TP_Rmultiple)
        Time Stop: Exit if in trade >= time_stop_candles and profit < +0.5R
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        entry_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        entry_atr = entry_candle["atr"].iloc[-1] if not entry_candle.empty else dataframe["atr"].iloc[-1]

        if np.isnan(entry_atr) or entry_atr <= 0:
            return None

        sl_dist = float(self.sl_atr_mult.value) * entry_atr
        r_mult = float(self.tp_r_mult.value)
        tp_rmultiple_dist = r_mult * sl_dist

        last_candle = dataframe.iloc[-1]
        tp_meanrevert = last_candle.get("mean_close", trade.open_rate)

        # 1. Take Profit Logic (Mean reversion or R-multiple target, ensuring min positive return)
        if not trade.is_short:
            tp_rmult_price = trade.open_rate + tp_rmultiple_dist
            tp_target = max(trade.open_rate + 0.5 * sl_dist, min(tp_meanrevert, tp_rmult_price))
            if current_rate >= tp_target:
                return "tp_target_long"
        else:
            tp_rmult_price = trade.open_rate - tp_rmultiple_dist
            tp_target = min(trade.open_rate - 0.5 * sl_dist, max(tp_meanrevert, tp_rmult_price))
            if current_rate <= tp_target:
                return "tp_target_short"

        # 2. Time Stop Logic (Give trades enough time, only exit if stagnant/losing)
        tf_mins = timeframe_to_minutes(self.timeframe)
        elapsed_candles = (current_time - trade.open_date_utc).total_seconds() / (tf_mins * 60)

        if elapsed_candles >= float(self.time_stop_candles.value) and current_profit < 0.0:
            return "time_stop"

        return None

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        """
        Position Sizing:
        Risk_amount = Capital * RISK_PCT_PER_TRADE
        Qty = Risk_amount / ABS(Entry - SL)
        Stake = Qty * Entry = (Risk_amount * Entry) / SL_dist
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or current_rate <= 0:
            return proposed_stake

        last_atr = dataframe["atr"].iloc[-1]
        if np.isnan(last_atr) or last_atr <= 0:
            return proposed_stake

        sl_dist = float(self.sl_atr_mult.value) * last_atr
        if sl_dist <= 0:
            return proposed_stake

        # Calculate stake based on 0.5% risk
        wallet_balance = self.wallets.get_total_stake_amount()
        risk_amount = wallet_balance * float(self.risk_pct_per_trade.value)
        calculated_stake = (risk_amount * current_rate) / sl_dist

        # Bound stake between min_stake and max_stake
        if min_stake is not None:
            calculated_stake = max(calculated_stake, min_stake)
        calculated_stake = min(calculated_stake, max_stake)

        return calculated_stake
