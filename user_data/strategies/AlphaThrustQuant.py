"""
AlphaThrustQuant Strategy
Quantitative Leading-Indicator & Price-Action Intraday Futures Strategy

Key Components:
1. Higher-Timeframe (1h) KAMA Adaptive Trend Alignment (@informative)
2. Kaufman's Adaptive Moving Average (KAMA) with Dynamic Efficiency Ratio (15m)
3. TTM Squeeze Release & Linear Regression Momentum Wave (Leading Oscillator)
4. Choppiness Index (CHOP) & Chaikin Money Flow (CMF) Volatility Filters
5. Institutional Displacement & Liquidity Sweep Pin-Bar Market Structure
6. Asymmetric Multi-Tier Take-Profit Ratchet & Chandelier ATR Trailing Stop
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
    informative,
    stoploss_from_open,
)


class AlphaThrustQuant(IStrategy):
    """
    AlphaThrustQuant: Institutional-Grade BTC Intraday Futures Strategy
    """

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "15m"

    minimal_roi = {"0": 10.0}
    stoploss = -0.10
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count: int = 150

    # -------------------------------------------------------------
    # Hyperopt Parameter Spaces
    # -------------------------------------------------------------
    # Buy / Indicator Spaces
    kama_len = IntParameter(10, 40, default=20, space="buy", optimize=True)
    chop_max = DecimalParameter(45.0, 60.0, default=52.0, decimals=1, space="buy", optimize=True)
    cmf_len = IntParameter(14, 28, default=20, space="buy", optimize=False)
    cmf_threshold = DecimalParameter(-0.02, 0.12, default=0.03, decimals=2, space="buy", optimize=True)
    displacement_ratio = DecimalParameter(0.55, 0.80, default=0.65, decimals=2, space="buy", optimize=True)
    displacement_atr_mult = DecimalParameter(0.8, 1.8, default=1.2, decimals=2, space="buy", optimize=True)
    sqz_length = IntParameter(14, 26, default=20, space="buy", optimize=False)

    # Sell / Risk Management Spaces
    sl_atr_mult = DecimalParameter(1.8, 3.2, default=2.4, decimals=2, space="sell", optimize=True)
    tp1_r_mult = DecimalParameter(1.0, 2.0, default=1.4, decimals=2, space="sell", optimize=True)
    tp2_r_mult = DecimalParameter(2.2, 4.0, default=2.8, decimals=2, space="sell", optimize=True)
    tp3_r_mult = DecimalParameter(4.0, 8.0, default=5.0, decimals=2, space="sell", optimize=True)
    trail_atr_mult = DecimalParameter(1.0, 2.5, default=1.5, decimals=2, space="sell", optimize=True)
    enable_momentum_exhaustion_exit = BooleanParameter(default=True, space="sell", optimize=False)

    plot_config = {
        "main_plot": {
            "kama": {"color": "#00e5ff"},
            "kama_1h": {"color": "#ff007f"},
            "bb_upper": {"color": "#78909c"},
            "bb_lower": {"color": "#78909c"},
            "kc_upper": {"color": "#ffca28"},
            "kc_lower": {"color": "#ffca28"},
        },
        "subplots": {
            "Squeeze Momentum": {
                "sqz_mom": {"color": "#00e676"},
            },
            "Choppiness & CMF": {
                "chop": {"color": "#ab47bc"},
                "cmf": {"color": "#29b6f6"},
            },
        },
    }

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate 1h Higher Timeframe KAMA and Trend Direction"""
        dataframe["kama"] = ta.KAMA(dataframe["close"], timeperiod=25)
        dataframe["trend_up"] = (dataframe["close"] > dataframe["kama"]).astype(int)
        dataframe["trend_dn"] = (dataframe["close"] < dataframe["kama"]).astype(int)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate Leading KAMA, Squeeze Momentum, Choppiness Index, CMF, and Market Structure"""
        # 1. Kaufman's Adaptive Moving Average (KAMA)
        dataframe["kama"] = ta.KAMA(dataframe["close"], timeperiod=int(self.kama_len.value))
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=14)

        # 2. TTM Squeeze Momentum (Bollinger Bands vs Keltner Channels)
        sqz_len = int(self.sqz_length.value)
        basis = ta.SMA(dataframe["close"], timeperiod=sqz_len)
        std_dev = dataframe["close"].rolling(sqz_len).std()

        dataframe["bb_upper"] = basis + (2.0 * std_dev)
        dataframe["bb_lower"] = basis - (2.0 * std_dev)

        keltner_atr = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=sqz_len)
        dataframe["kc_upper"] = basis + (1.5 * keltner_atr)
        dataframe["kc_lower"] = basis - (1.5 * keltner_atr)

        dataframe["squeeze_on"] = (dataframe["bb_upper"] < dataframe["kc_upper"]) & (dataframe["bb_lower"] > dataframe["kc_lower"])

        # Linear Regression Momentum Wave
        donchian_mid = (dataframe["high"].rolling(sqz_len).max() + dataframe["low"].rolling(sqz_len).min()) / 2.0
        delta = dataframe["close"] - ((donchian_mid + basis) / 2.0)
        dataframe["sqz_mom"] = ta.LINEARREG(delta, timeperiod=sqz_len)

        dataframe["sqz_mom_rising"] = dataframe["sqz_mom"] > dataframe["sqz_mom"].shift(1)
        dataframe["sqz_mom_falling"] = dataframe["sqz_mom"] < dataframe["sqz_mom"].shift(1)

        # 3. Choppiness Index (CHOP)
        tr1 = pd.Series(ta.TRANGE(dataframe["high"], dataframe["low"], dataframe["close"]), index=dataframe.index)
        sum_tr = tr1.rolling(14).sum()
        high_low_diff = dataframe["high"].rolling(14).max() - dataframe["low"].rolling(14).min()
        ratio = np.where(high_low_diff > 0, sum_tr / high_low_diff, 1.0)
        ratio = np.where(ratio > 0, ratio, 1.0)
        dataframe["chop"] = 100.0 * np.log10(ratio) / np.log10(14.0)

        # 4. Chaikin Money Flow (CMF)
        cmf_p = int(self.cmf_len.value)
        hl_diff = dataframe["high"] - dataframe["low"]
        mfm = np.where(hl_diff > 0, ((dataframe["close"] - dataframe["low"]) - (dataframe["high"] - dataframe["close"])) / hl_diff, 0.0)
        mfv = mfm * dataframe["volume"]
        vol_sum = dataframe["volume"].rolling(cmf_p).sum()
        dataframe["cmf"] = np.where(vol_sum > 0, mfv.rolling(cmf_p).sum() / vol_sum, 0.0)

        # 5. Price Action Displacement & Liquidity Pin-Bar Sweeps
        candle_body = (dataframe["close"] - dataframe["open"]).abs()
        candle_range = dataframe["high"] - dataframe["low"]

        # Institutional Displacement (Strong decisive body expansion)
        dataframe["is_displacement_long"] = (
            (dataframe["close"] > dataframe["open"])
            & (candle_body >= (self.displacement_ratio.value * candle_range))
            & (candle_body >= (self.displacement_atr_mult.value * dataframe["atr"]))
        )
        dataframe["is_displacement_short"] = (
            (dataframe["close"] < dataframe["open"])
            & (candle_body >= (self.displacement_ratio.value * candle_range))
            & (candle_body >= (self.displacement_atr_mult.value * dataframe["atr"]))
        )

        # Liquidity Sweep Pin-Bar (Wick swept swing extreme, but closed with strong rejection)
        swing_low_10 = dataframe["low"].shift(1).rolling(10).min()
        swing_high_10 = dataframe["high"].shift(1).rolling(10).max()
        lower_wick = dataframe[["open", "close"]].min(axis=1) - dataframe["low"]
        upper_wick = dataframe["high"] - dataframe[["open", "close"]].max(axis=1)

        dataframe["is_sweep_long"] = (
            (dataframe["low"] < swing_low_10)
            & (dataframe["close"] > swing_low_10)
            & (dataframe["close"] >= dataframe["open"])
            & (lower_wick >= (0.35 * candle_range))
        )
        dataframe["is_sweep_short"] = (
            (dataframe["high"] > swing_high_10)
            & (dataframe["close"] < swing_high_10)
            & (dataframe["close"] <= dataframe["open"])
            & (upper_wick >= (0.35 * candle_range))
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Trigger high-conviction entries aligned with 1h KAMA trend"""
        chop_limit = self.chop_max.value
        cmf_thresh = self.cmf_threshold.value

        # HTF 1h Trend Alignment
        htf_bull = dataframe.get("trend_up_1h", 1) == 1
        htf_bear = dataframe.get("trend_dn_1h", 1) == 1

        # Long Entry: 1h Bullish + 15m CHOP Filter + 15m CMF Flow + KAMA Reclaim + Green Accelerating Squeeze Momentum + Price Action Confirmation
        long_condition = (
            htf_bull
            & (dataframe["chop"] < chop_limit)
            & (dataframe["cmf"] >= cmf_thresh)
            & (dataframe["close"] > dataframe["kama"])
            & (dataframe["sqz_mom"] > 0)
            & (dataframe["sqz_mom_rising"])
            & (dataframe["is_displacement_long"] | dataframe["is_sweep_long"])
        )

        dataframe.loc[long_condition & ~long_condition.shift(1).fillna(False), "enter_long"] = 1

        # Short Entry: 1h Bearish + 15m CHOP Filter + 15m CMF Outflow + KAMA Rejection + Red Accelerating Squeeze Momentum + Price Action Confirmation
        short_condition = (
            htf_bear
            & (dataframe["chop"] < chop_limit)
            & (dataframe["cmf"] <= -cmf_thresh)
            & (dataframe["close"] < dataframe["kama"])
            & (dataframe["sqz_mom"] < 0)
            & (dataframe["sqz_mom_falling"])
            & (dataframe["is_displacement_short"] | dataframe["is_sweep_short"])
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
        Multi-Tier Asymmetric Risk Management:
        1. Initial SL = entry - sl_atr_mult * ATR (Wider structural cushion)
        2. TP1 reached (+1.4R) -> Move SL to Breakeven (+0.1% fee cover)
        3. TP2 reached (+2.8R) -> Lock in +1.4R & Activate dynamic Chandelier ATR Trailing Stop
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        # Find entry ATR
        entry_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        entry_atr = entry_candle["atr"].iloc[-1] if not entry_candle.empty else dataframe["atr"].iloc[-1]

        if np.isnan(entry_atr) or entry_atr <= 0:
            return None

        atr_ratio = entry_atr / trade.open_rate
        initial_sl_ratio = float(self.sl_atr_mult.value) * atr_ratio
        tp1_ratio = float(self.tp1_r_mult.value) * initial_sl_ratio
        tp2_ratio = float(self.tp2_r_mult.value) * initial_sl_ratio

        # Tier 3: Profit reached TP2 (+2.8R) -> Chandelier ATR Trailing Stop
        if current_profit >= tp2_ratio:
            current_atr = dataframe["atr"].iloc[-1]
            trail_dist_ratio = (float(self.trail_atr_mult.value) * current_atr) / current_rate
            target_stop_ratio = max(tp1_ratio, current_profit - trail_dist_ratio)
            return stoploss_from_open(target_stop_ratio, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Tier 2: Profit reached TP1 (+1.4R) -> Move to Breakeven
        if current_profit >= tp1_ratio:
            return stoploss_from_open(0.001, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # Tier 1: Initial Stop Loss
        return stoploss_from_open(-initial_sl_ratio, current_profit, is_short=trade.is_short, leverage=trade.leverage)

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
        1. TP3 Profit Target (+5.0R)
        2. Momentum Exhaustion Exit: Squeeze momentum rolls over in profit
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        entry_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        entry_atr = entry_candle["atr"].iloc[-1] if not entry_candle.empty else dataframe["atr"].iloc[-1]

        if not np.isnan(entry_atr) and entry_atr > 0:
            initial_sl_ratio = float(self.sl_atr_mult.value) * (entry_atr / trade.open_rate)
            tp3_ratio = float(self.tp3_r_mult.value) * initial_sl_ratio
            if current_profit >= tp3_ratio:
                return "tp3_target_exit"

        # Momentum Exhaustion Exit (Lock in profit on leading momentum rollover)
        if self.enable_momentum_exhaustion_exit.value and current_profit >= 0.006:
            last_candle = dataframe.iloc[-1]
            sqz_mom = last_candle.get("sqz_mom", 0.0)
            sqz_mom_prev = dataframe["sqz_mom"].iloc[-2] if len(dataframe) > 1 else sqz_mom

            if not trade.is_short and sqz_mom < sqz_mom_prev and sqz_mom_prev > 0:
                return "mom_exhaustion_exit_long"
            elif trade.is_short and sqz_mom > sqz_mom_prev and sqz_mom_prev < 0:
                return "mom_exhaustion_exit_short"

        return None
