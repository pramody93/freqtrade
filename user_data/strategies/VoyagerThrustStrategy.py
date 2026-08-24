# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file

"""
Voyager Thrust Strategy
Translated faithfully from TradingView Pine Script v6 (Voyager Thrust Strategy / OrionAlgo).
Supports Spot & Futures (Long & Short) with Multi-Timeframe Thrust Oscillator,
ADX momentum filter, SMA trend alignment, and ATR-based multi-tier Stoploss/TP management.
"""

from datetime import datetime, timezone
import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.strategy import (
    IStrategy,
    Trade,
    Order,
    PairLocks,
    informative,
    stoploss_from_open,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    RealParameter,
)


def calc_thrust_pulse(close_series: pd.Series) -> pd.Series:
    """Core Thruster Pulse calculation: (EMA(12) - EMA(26) - EMA(Signal, 9)) * 3"""
    fast_len = 12
    slow_len = 26
    signal_len = 9
    thrust_core = ta.EMA(close_series, timeperiod=fast_len) - ta.EMA(close_series, timeperiod=slow_len)
    thrust_signal = ta.EMA(thrust_core, timeperiod=signal_len)
    pulse = (thrust_core - thrust_signal) * 3
    if isinstance(pulse, np.ndarray):
        return pd.Series(pulse, index=close_series.index)
    return pulse


def calc_thrust_trend_direction(pulse: pd.Series | np.ndarray, index: pd.Index | None = None) -> pd.Series:
    """Calculates the stateful bullish (1) / bearish (-1) phase of the thrust oscillator."""
    if isinstance(pulse, pd.Series):
        p = pulse.to_numpy()
        idx = pulse.index
    else:
        p = np.asarray(pulse)
        idx = index

    n = len(p)
    trend = np.ones(n, dtype=int)
    bearish = False

    for i in range(1, n):
        if np.isnan(p[i]) or np.isnan(p[i - 1]):
            continue
        is_falling = p[i] < p[i - 1]
        is_rising = p[i] > p[i - 1]
        bearish = is_falling or (bearish and not is_rising)
        trend[i] = -1 if bearish else 1

    return pd.Series(trend, index=idx) if idx is not None else pd.Series(trend)


class VoyagerThrustStrategy(IStrategy):
    """
    Voyager Thrust Strategy (Pine Script Port)
    """

    INTERFACE_VERSION = 3

    # Enable Futures Long & Short trading
    can_short: bool = True

    # Base Timeframe
    timeframe: str = "5m"

    # Multi-timeframe Informatives
    inf_tf_1: str = "15m"  # Primary Thrust TF
    inf_tf_2: str = "30m"  # Secondary Thrust Alt TF
    inf_tf_htf: str = "1h"  # HTF Base Trend TF (SMA 50 + HTF Thrust)

    # Strategy Settings
    process_only_new_candles = True
    startup_candle_count: int = 100

    # Minimal ROI (Managed by dynamic ATR Take Profits in custom_exit / custom_stoploss)
    minimal_roi = {
        "0": 10.0  # Kept high so custom_exit and custom_stoploss control exits
    }

    # Hard Stoploss fallback (-10%)
    stoploss = -0.10

    # Risk Management Multipliers (Hyperoptable)
    sl_atr_mult = DecimalParameter(1.0, 3.0, default=1.5, decimals=2, space="sell", optimize=True)
    tp1_atr_mult = DecimalParameter(1.5, 3.5, default=2.0, decimals=2, space="sell", optimize=True)
    tp2_atr_mult = DecimalParameter(3.0, 5.5, default=4.0, decimals=2, space="sell", optimize=True)
    tp3_atr_mult = DecimalParameter(5.0, 8.0, default=6.0, decimals=2, space="sell", optimize=True)

    use_trail_after_tp2 = BooleanParameter(default=False, space="sell", optimize=False)
    trail_atr_mult = DecimalParameter(1.0, 4.0, default=2.0, decimals=2, space="sell", optimize=False)

    # Indicators Lengths
    sma_len = IntParameter(20, 100, default=50, space="buy", optimize=False)
    atr_len = IntParameter(7, 21, default=14, space="buy", optimize=False)
    adx_len = IntParameter(7, 21, default=14, space="buy", optimize=False)
    adx_min = DecimalParameter(10.0, 35.0, default=20.0, decimals=1, space="buy", optimize=True)
    use_adx_filter = BooleanParameter(default=True, space="buy", optimize=False)

    use_custom_stoploss = True

    # FreqUI Chart Visualizations
    plot_config = {
        "main_plot": {
            "sma_chart": {"color": "#ffa500"},
            f"sma_htf_{inf_tf_htf}": {"color": "#2196f3"},
        },
        "subplots": {
            "Thrust Oscillator": {
                f"thrust_{inf_tf_1}": {"color": "#00e676"},
                f"thrust_{inf_tf_2}": {"color": "#00bcd4"},
            },
            "ADX Trend Strength": {
                "adx": {"color": "#e91e63"},
            },
        },
    }

    @informative("15m")
    def populate_indicators_15m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pulse = calc_thrust_pulse(dataframe["close"])
        dataframe["thrust"] = pulse
        dataframe["thrust_trend"] = calc_thrust_trend_direction(pulse)
        return dataframe

    @informative("30m")
    def populate_indicators_30m(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pulse = calc_thrust_pulse(dataframe["close"])
        dataframe["thrust"] = pulse
        dataframe["thrust_trend"] = calc_thrust_trend_direction(pulse)
        return dataframe

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["sma_htf"] = ta.SMA(dataframe["close"], timeperiod=int(self.sma_len.value))
        pulse = calc_thrust_pulse(dataframe["close"])
        dataframe["thrust"] = pulse
        dataframe["thrust_trend"] = calc_thrust_trend_direction(pulse)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate Base Timeframe Indicators (SMA 50, ATR 14, ADX 14)"""
        # Chart SMA
        dataframe["sma_chart"] = ta.SMA(dataframe["close"], timeperiod=int(self.sma_len.value))

        # ATR & ADX
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=int(self.atr_len.value))
        dataframe["adx"] = ta.ADX(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=int(self.adx_len.value))

        # ADX condition
        if self.use_adx_filter.value:
            dataframe["adx_ok"] = dataframe["adx"] >= self.adx_min.value
        else:
            dataframe["adx_ok"] = True

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Entry Conditions:
        Long: close > smaChart & close > smaHtf & thrust15m > 0 & thrustTrend15m == 1 & thrustTrend30m == 1 & adx >= 20
        Short: close < smaChart & close < smaHtf & thrust15m < 0 & thrustTrend15m == -1 & thrustTrend30m == -1 & adx >= 20
        """
        # Column names merged from informative decorators
        thrust_15m = dataframe["thrust_15m"]
        thrust_trend_15m = dataframe["thrust_trend_15m"]
        thrust_trend_30m = dataframe["thrust_trend_30m"]
        sma_htf = dataframe[f"sma_htf_{self.inf_tf_htf}"]

        # Long Condition
        long_condition = (
            (dataframe["close"] > dataframe["sma_chart"])
            & (dataframe["close"] > sma_htf)
            & (thrust_trend_15m == 1)
            & (thrust_trend_30m == 1)
            & (thrust_15m > 0)
            & (dataframe["adx_ok"])
        )

        # Trigger entry on signal transition
        dataframe.loc[long_condition & ~long_condition.shift(1).fillna(False), "enter_long"] = 1

        # Short Condition
        short_condition = (
            (dataframe["close"] < dataframe["sma_chart"])
            & (dataframe["close"] < sma_htf)
            & (thrust_trend_15m == -1)
            & (thrust_trend_30m == -1)
            & (thrust_15m < 0)
            & (dataframe["adx_ok"])
        )

        # Trigger entry on signal transition
        dataframe.loc[short_condition & ~short_condition.shift(1).fillna(False), "enter_short"] = 1

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Exit trend signals (Managed dynamically in custom_exit)"""
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
        Dynamic ATR-Based Stoploss & Take-Profit Ratchet:
        - Initial SL = entry - 1.5x ATR
        - TP1 reached (+2.0x ATR) -> Move SL to Breakeven (0.0% profit)
        - TP2 reached (+4.0x ATR) -> Move SL to TP1 level (+2.0x ATR profit lock)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        # Find ATR at entry candle
        entry_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if entry_candle.empty:
            entry_atr = dataframe["atr"].iloc[-1]
        else:
            entry_atr = entry_candle["atr"].iloc[-1]

        if np.isnan(entry_atr) or entry_atr <= 0 or trade.open_rate <= 0:
            return None

        # Convert ATR distances to profit ratio relative to open_rate
        atr_ratio = entry_atr / trade.open_rate
        sl_ratio = float(self.sl_atr_mult.value) * atr_ratio
        tp1_ratio = float(self.tp1_atr_mult.value) * atr_ratio
        tp2_ratio = float(self.tp2_atr_mult.value) * atr_ratio

        # 1. If current profit reached TP2: Lock stoploss at TP1 profit level
        if current_profit >= tp2_ratio:
            return stoploss_from_open(tp1_ratio, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # 2. If current profit reached TP1: Move stoploss to Breakeven (0.1% profit to cover fees)
        if current_profit >= tp1_ratio:
            return stoploss_from_open(0.001, current_profit, is_short=trade.is_short, leverage=trade.leverage)

        # 3. Initial stoploss (1.5x ATR below entry)
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
        Custom Exit Rules:
        1. TP3 Target Exit (+6.0x ATR)
        2. Trend-Flip Exit:
           - Long: Cross under HTF SMA 50 & HTF Thrust < 0
           - Short: Cross over HTF SMA 50 & HTF Thrust > 0
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty:
            return None

        last_candle = dataframe.iloc[-1]

        # ATR at entry
        entry_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        entry_atr = entry_candle["atr"].iloc[-1] if not entry_candle.empty else dataframe["atr"].iloc[-1]

        if not np.isnan(entry_atr) and entry_atr > 0 and trade.open_rate > 0:
            tp3_ratio = float(self.tp3_atr_mult.value) * (entry_atr / trade.open_rate)
            if current_profit >= tp3_ratio:
                return "tp3_target_exit"

        # Trend-Flip Exit
        sma_htf_col = f"sma_htf_{self.inf_tf_htf}"
        thrust_htf_col = f"thrust_{self.inf_tf_htf}"

        if sma_htf_col in last_candle and thrust_htf_col in last_candle:
            sma_htf_val = last_candle[sma_htf_col]
            thrust_htf_val = last_candle[thrust_htf_col]

            if not trade.is_short:
                if current_rate < sma_htf_val and thrust_htf_val < 0:
                    return "trend_flip_exit_long"
            else:
                if current_rate > sma_htf_val and thrust_htf_val > 0:
                    return "trend_flip_exit_short"

        return None
