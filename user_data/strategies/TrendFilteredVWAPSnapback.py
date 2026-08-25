"""
Trend-Filtered VWAP Snapback Strategy
Based on "Systematic Strategy Research: BTC Perpetual Futures Rulebook" (Strategy 3)

Concept: Trend-Filtered Mean Reversion after an ATR-defined Stretch
Timeframe: 5 minutes
Target Pair: BTC/USD:USD (Linear Perpetual Futures)
"""

from datetime import datetime
import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    BooleanParameter,
    DecimalParameter,
    IStrategy,
    IntParameter,
    stoploss_from_open,
)


class TrendFilteredVWAPSnapback(IStrategy):
    """
    Trend-Filtered VWAP Snapback Strategy (5m)
    - Session VWAP (00:00 UTC Reset) + EMA(200) Trend Filter
    - ATR Stretch Excursion & Reversal Confirmation
    - Fixed Signal VWAP Target (>= 1.25R)
    - Regime-Failure Exit & 12-Bar (60m) Time Exit
    """

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "5m"

    minimal_roi = {"0": 10.0}
    stoploss = -0.10
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count: int = 250

    # -------------------------------------------------------------
    # Hyperoptable Parameters
    # -------------------------------------------------------------
    ema_len = IntParameter(150, 250, default=200, space="buy", optimize=True)
    slope_lookback = IntParameter(15, 25, default=20, space="buy", optimize=False)
    stretch_atr_mult = DecimalParameter(1.2, 1.8, default=1.5, decimals=2, space="buy", optimize=True)
    stretch_close_mult = DecimalParameter(0.8, 1.3, default=1.0, decimals=2, space="buy", optimize=False)
    max_stretch_range_mult = DecimalParameter(2.0, 3.0, default=2.5, decimals=2, space="buy", optimize=False)
    min_target_r = DecimalParameter(1.0, 1.6, default=1.25, decimals=2, space="buy", optimize=True)

    # Exits
    max_holding_bars = IntParameter(8, 16, default=12, space="sell", optimize=False)
    enable_regime_failure_exit = BooleanParameter(default=True, space="sell", optimize=False)

    plot_config = {
        "main_plot": {
            "vwap": {"color": "#ffea00"},
            "ema_trend": {"color": "#00e5ff"},
        },
        "subplots": {
            "ATR": {
                "atr": {"color": "#76ff03"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate Session VWAP (00:00 UTC Reset), EMA(200), ATR(14), and Stretch conditions"""
        # Daily Session VWAP Calculation
        typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3.0
        pv = typical_price * dataframe["volume"]

        # Group by UTC date for session reset at 00:00 UTC
        utc_dates = pd.to_datetime(dataframe["date"]).dt.tz_localize(None) if dataframe["date"].dt.tz is None else pd.to_datetime(dataframe["date"]).dt.tz_convert("UTC").dt.tz_localize(None)
        day_groups = utc_dates.dt.floor("D")

        cum_pv = pv.groupby(day_groups).cumsum()
        cum_vol = dataframe["volume"].groupby(day_groups).cumsum()
        dataframe["vwap"] = np.where(cum_vol > 0, cum_pv / cum_vol, typical_price)

        # EMA Trend & ATR
        dataframe["ema_trend"] = ta.EMA(dataframe["close"], timeperiod=int(self.ema_len.value))
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)

        # EMA Trend slope (compare with 20 bars earlier)
        slope_lb = int(self.slope_lookback.value)
        dataframe["ema_slope_up"] = dataframe["ema_trend"] > dataframe["ema_trend"].shift(slope_lb)
        dataframe["ema_slope_down"] = dataframe["ema_trend"] < dataframe["ema_trend"].shift(slope_lb)

        # Trading Time Window Filter (01:00 through 22:55 UTC)
        hour = utc_dates.dt.hour
        minute = utc_dates.dt.minute
        dataframe["time_ok"] = (hour >= 1) & ((hour < 22) | ((hour == 22) & (minute <= 55)))

        # Stretch bar metrics on t-1
        dataframe["atr_prev"] = dataframe["atr"].shift(1)
        dataframe["vwap_prev"] = dataframe["vwap"].shift(1)
        dataframe["stretch_range"] = dataframe["high"].shift(1) - dataframe["low"].shift(1)
        dataframe["stretch_range_ok"] = dataframe["stretch_range"] <= (self.max_stretch_range_mult.value * dataframe["atr_prev"])

        # Long Stretch conditions on t-1
        stretch_mult = self.stretch_atr_mult.value
        stretch_close_mult = self.stretch_close_mult.value
        dataframe["long_stretch_ok"] = (
            (dataframe["low"].shift(1) <= (dataframe["vwap_prev"] - stretch_mult * dataframe["atr_prev"]))
            & (dataframe["close"].shift(1) <= (dataframe["vwap_prev"] - stretch_close_mult * dataframe["atr_prev"]))
            & (dataframe["stretch_range_ok"])
        )

        # Short Stretch conditions on t-1
        dataframe["short_stretch_ok"] = (
            (dataframe["high"].shift(1) >= (dataframe["vwap_prev"] + stretch_mult * dataframe["atr_prev"]))
            & (dataframe["close"].shift(1) >= (dataframe["vwap_prev"] + stretch_close_mult * dataframe["atr_prev"]))
            & (dataframe["stretch_range_ok"])
        )

        # Signal bar reversal metrics
        stretch_midpoint = (dataframe["high"].shift(1) + dataframe["low"].shift(1)) / 2.0
        dataframe["reversal_long_ok"] = (
            (dataframe["close"] > stretch_midpoint)
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["close"] > dataframe["close"].shift(1))
            & (dataframe["close"] < dataframe["vwap"])
            & (dataframe["high"] < dataframe["vwap"])
        )

        dataframe["reversal_short_ok"] = (
            (dataframe["close"] < stretch_midpoint)
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["close"] < dataframe["close"].shift(1))
            & (dataframe["close"] > dataframe["vwap"])
            & (dataframe["low"] > dataframe["vwap"])
        )

        # Stop, Risk & Target Distance Calculation
        dataframe["stop_long_price"] = dataframe["low"].shift(1) - (0.25 * dataframe["atr"])
        dataframe["stop_short_price"] = dataframe["high"].shift(1) + (0.25 * dataframe["atr"])

        dataframe["risk_long"] = dataframe["close"] - dataframe["stop_long_price"]
        dataframe["risk_short"] = dataframe["stop_short_price"] - dataframe["close"]

        dataframe["target_dist_long"] = dataframe["vwap"] - dataframe["close"]
        dataframe["target_dist_short"] = dataframe["close"] - dataframe["vwap"]

        min_r = 0.50 * dataframe["atr"]
        max_r = 1.50 * dataframe["atr"]
        min_target_r_val = self.min_target_r.value

        dataframe["risk_ok_long"] = (
            (dataframe["risk_long"] >= min_r)
            & (dataframe["risk_long"] <= max_r)
            & (dataframe["target_dist_long"] >= min_target_r_val * dataframe["risk_long"])
        )

        dataframe["risk_ok_short"] = (
            (dataframe["risk_short"] >= min_r)
            & (dataframe["risk_short"] <= max_r)
            & (dataframe["target_dist_short"] >= min_target_r_val * dataframe["risk_short"])
        )

        dataframe["target_vwap_frozen"] = dataframe["vwap"]

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Evaluate VWAP snapback reversal on completed signal candle"""
        # Long Entry
        long_condition = (
            (dataframe["time_ok"])
            & (dataframe["ema_slope_up"])
            & (dataframe["vwap"] > dataframe["ema_trend"])
            & (dataframe["long_stretch_ok"])
            & (dataframe["reversal_long_ok"])
            & (dataframe["risk_ok_long"])
        )

        dataframe.loc[long_condition & ~long_condition.shift(1).fillna(False), "enter_long"] = 1

        # Short Entry
        short_condition = (
            (dataframe["time_ok"])
            & (dataframe["ema_slope_down"])
            & (dataframe["vwap"] < dataframe["ema_trend"])
            & (dataframe["short_stretch_ok"])
            & (dataframe["reversal_short_ok"])
            & (dataframe["risk_ok_short"])
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
        """Fixed initial stop based on Stretch Bar low/high +/- 0.25 * ATR"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        signal_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if signal_candle.empty:
            return None

        last_signal = signal_candle.iloc[-1]

        if not trade.is_short:
            stop_price = last_signal.get("stop_long_price", 0.0)
            if stop_price > 0 and stop_price < trade.open_rate:
                initial_stop_ratio = (stop_price - trade.open_rate) / trade.open_rate
                return stoploss_from_open(initial_stop_ratio, current_profit, is_short=False, leverage=trade.leverage)
        else:
            stop_price = last_signal.get("stop_short_price", 0.0)
            if stop_price > 0 and stop_price > trade.open_rate:
                initial_stop_ratio = (trade.open_rate - stop_price) / trade.open_rate
                return stoploss_from_open(initial_stop_ratio, current_profit, is_short=True, leverage=trade.leverage)

        return None

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
        Custom Exits:
        1. Fixed Target at Signal Bar VWAP
        2. Regime-Failure Exit: Long if VWAP <= EMA(200); Short if VWAP >= EMA(200)
        3. Time Exit: After 12 completed 5m bars (60 minutes)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        signal_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if signal_candle.empty:
            return None

        last_signal = signal_candle.iloc[-1]
        target_vwap = last_signal.get("target_vwap_frozen", 0.0)

        # 1. Fixed Target Exit at frozen signal VWAP
        if target_vwap > 0:
            if not trade.is_short and current_rate >= target_vwap:
                return "target_fixed_vwap_long"
            elif trade.is_short and current_rate <= target_vwap:
                return "target_fixed_vwap_short"

        # 2. Regime-Failure Exit
        if self.enable_regime_failure_exit.value:
            current_bar = dataframe.iloc[-1]
            vwap_val = current_bar.get("vwap", 0.0)
            ema_val = current_bar.get("ema_trend", 0.0)
            if vwap_val > 0 and ema_val > 0:
                if not trade.is_short and vwap_val <= ema_val:
                    return "regime_failure_exit_long"
                elif trade.is_short and vwap_val >= ema_val:
                    return "regime_failure_exit_short"

        # 3. Time Exit: 12 completed 5m bars (12 * 5m = 60 min)
        elapsed_seconds = (current_time - trade.open_date_utc).total_seconds()
        max_duration_seconds = int(self.max_holding_bars.value) * 5 * 60
        if elapsed_seconds >= max_duration_seconds:
            return f"time_exit_{self.max_holding_bars.value}_bars"

        return None
