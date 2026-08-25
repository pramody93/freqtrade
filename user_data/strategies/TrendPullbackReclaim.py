"""
Trend Pullback Reclaim Strategy
Based on "Systematic Strategy Research: BTC Perpetual Futures Rulebook" (Strategy 1)

Concept: Trend + Pullback + Price-Action Continuation
Timeframe: 15 minutes
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


class TrendPullbackReclaim(IStrategy):
    """
    Trend Pullback Reclaim Strategy (15m)
    - Structural ATR-based Stop Loss
    - Fixed 2.00R Take Profit Target
    - EMA(50) Trend-Failure Exit
    - 24-Bar Time Exit
    """

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "15m"

    # Minimal ROI placeholder - Exits handled dynamically in custom_exit
    minimal_roi = {"0": 10.0}

    # Hard emergency stoploss (overridden by custom_stoploss)
    stoploss = -0.10
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count: int = 100

    # -------------------------------------------------------------
    # Hyperoptable Parameters
    # -------------------------------------------------------------
    # Buy / Indicator Spaces
    ema_fast_len = IntParameter(15, 25, default=20, space="buy", optimize=True)
    ema_slow_len = IntParameter(40, 60, default=50, space="buy", optimize=True)
    atr_len = IntParameter(10, 20, default=14, space="buy", optimize=False)
    slope_lookback = IntParameter(3, 8, default=5, space="buy", optimize=False)
    ema_sep_mult = DecimalParameter(0.15, 0.40, default=0.25, decimals=2, space="buy", optimize=True)
    max_signal_range_mult = DecimalParameter(1.2, 2.0, default=1.5, decimals=2, space="buy", optimize=False)
    min_risk_mult = DecimalParameter(0.3, 0.7, default=0.5, decimals=2, space="buy", optimize=False)
    max_risk_mult = DecimalParameter(1.2, 2.0, default=1.5, decimals=2, space="buy", optimize=False)

    # Sell / Exit Spaces
    target_r_mult = DecimalParameter(1.5, 3.0, default=2.0, decimals=2, space="sell", optimize=True)
    stop_buffer_mult = DecimalParameter(0.05, 0.20, default=0.10, decimals=2, space="sell", optimize=False)
    max_holding_bars = IntParameter(16, 32, default=24, space="sell", optimize=False)
    enable_trend_failure_exit = BooleanParameter(default=True, space="sell", optimize=False)

    plot_config = {
        "main_plot": {
            "ema_fast": {"color": "#2962ff"},
            "ema_slow": {"color": "#ff6d00"},
        },
        "subplots": {
            "ATR": {
                "atr": {"color": "#00e676"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate EMA(20), EMA(50), ATR(14), and structural high/low rolling metrics"""
        # EMAs
        dataframe["ema_fast"] = ta.EMA(dataframe["close"], timeperiod=int(self.ema_fast_len.value))
        dataframe["ema_slow"] = ta.EMA(dataframe["close"], timeperiod=int(self.ema_slow_len.value))

        # ATR
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=int(self.atr_len.value))

        # EMA(50) slope: compare with 5 bars earlier
        slope_lb = int(self.slope_lookback.value)
        dataframe["ema_slow_slope_up"] = dataframe["ema_slow"] > dataframe["ema_slow"].shift(slope_lb)
        dataframe["ema_slow_slope_down"] = dataframe["ema_slow"] < dataframe["ema_slow"].shift(slope_lb)

        # EMA separation
        dataframe["ema_sep"] = (dataframe["ema_fast"] - dataframe["ema_slow"]).abs()
        dataframe["ema_sep_ok"] = dataframe["ema_sep"] >= (self.ema_sep_mult.value * dataframe["atr"])

        # Pullback conditions on bars t-3, t-2, t-1:
        # Check if at least one low <= ema_fast and all 3 closed above ema_slow (for Long)
        p1_touch_long = dataframe["low"].shift(1) <= dataframe["ema_fast"].shift(1)
        p2_touch_long = dataframe["low"].shift(2) <= dataframe["ema_fast"].shift(2)
        p3_touch_long = dataframe["low"].shift(3) <= dataframe["ema_fast"].shift(3)
        dataframe["pullback_touch_long"] = p1_touch_long | p2_touch_long | p3_touch_long

        p1_above_slow = dataframe["close"].shift(1) > dataframe["ema_slow"].shift(1)
        p2_above_slow = dataframe["close"].shift(2) > dataframe["ema_slow"].shift(2)
        p3_above_slow = dataframe["close"].shift(3) > dataframe["ema_slow"].shift(3)
        dataframe["pullback_above_slow"] = p1_above_slow & p2_above_slow & p3_above_slow

        # Check if at least one high >= ema_fast and all 3 closed below ema_slow (for Short)
        p1_touch_short = dataframe["high"].shift(1) >= dataframe["ema_fast"].shift(1)
        p2_touch_short = dataframe["high"].shift(2) >= dataframe["ema_fast"].shift(2)
        p3_touch_short = dataframe["high"].shift(3) >= dataframe["ema_fast"].shift(3)
        dataframe["pullback_touch_short"] = p1_touch_short | p2_touch_short | p3_touch_short

        p1_below_slow = dataframe["close"].shift(1) < dataframe["ema_slow"].shift(1)
        p2_below_slow = dataframe["close"].shift(2) < dataframe["ema_slow"].shift(2)
        p3_below_slow = dataframe["close"].shift(3) < dataframe["ema_slow"].shift(3)
        dataframe["pullback_below_slow"] = p1_below_slow & p2_below_slow & p3_below_slow

        # Signal bar range constraint
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["range_ok"] = dataframe["candle_range"] <= (self.max_signal_range_mult.value * dataframe["atr"])

        # 4-bar structural lowest low / highest high (3 pullback bars + signal bar)
        dataframe["struct_low_4"] = dataframe["low"].rolling(4).min()
        dataframe["struct_high_4"] = dataframe["high"].rolling(4).max()

        # Stop calculations & Risk checks
        buffer = self.stop_buffer_mult.value * dataframe["atr"]
        dataframe["stop_long_price"] = dataframe["struct_low_4"] - buffer
        dataframe["stop_short_price"] = dataframe["struct_high_4"] + buffer

        dataframe["risk_long"] = dataframe["close"] - dataframe["stop_long_price"]
        dataframe["risk_short"] = dataframe["stop_short_price"] - dataframe["close"]

        min_r = self.min_risk_mult.value * dataframe["atr"]
        max_r = self.max_risk_mult.value * dataframe["atr"]
        dataframe["risk_ok_long"] = (dataframe["risk_long"] >= min_r) & (dataframe["risk_long"] <= max_r)
        dataframe["risk_ok_short"] = (dataframe["risk_short"] >= min_r) & (dataframe["risk_short"] <= max_r)

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Evaluate exact rulebook conditions on completed signal bar"""
        # Long Conditions
        long_condition = (
            (dataframe["ema_fast"] > dataframe["ema_slow"])
            & (dataframe["ema_slow_slope_up"])
            & (dataframe["ema_sep_ok"])
            & (dataframe["pullback_touch_long"])
            & (dataframe["pullback_above_slow"])
            & (dataframe["close"] > dataframe["high"].shift(1))
            & (dataframe["close"] > dataframe["ema_fast"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["range_ok"])
            & (dataframe["risk_ok_long"])
        )

        dataframe.loc[long_condition & ~long_condition.shift(1).fillna(False), "enter_long"] = 1

        # Short Conditions
        short_condition = (
            (dataframe["ema_fast"] < dataframe["ema_slow"])
            & (dataframe["ema_slow_slope_down"])
            & (dataframe["ema_sep_ok"])
            & (dataframe["pullback_touch_short"])
            & (dataframe["pullback_below_slow"])
            & (dataframe["close"] < dataframe["low"].shift(1))
            & (dataframe["close"] < dataframe["ema_fast"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["range_ok"])
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
        """Fixed initial stop based on 4-bar structural low/high minus ATR buffer"""
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        # Find signal bar (candle immediately preceding or at trade open)
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
        1. Fixed Take Profit at 2.00R
        2. Trend-Failure Exit: Close through EMA(50)
        3. Time Exit: After 24 completed bars (6 hours)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        # Determine Initial Risk R
        signal_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if signal_candle.empty:
            return None

        last_signal = signal_candle.iloc[-1]

        if not trade.is_short:
            stop_price = last_signal.get("stop_long_price", 0.0)
            if stop_price > 0 and stop_price < trade.open_rate:
                r_ratio = (trade.open_rate - stop_price) / trade.open_rate
                target_ratio = float(self.target_r_mult.value) * r_ratio
                if current_profit >= target_ratio:
                    return f"target_{self.target_r_mult.value}r_long"
        else:
            stop_price = last_signal.get("stop_short_price", 0.0)
            if stop_price > 0 and stop_price > trade.open_rate:
                r_ratio = (stop_price - trade.open_rate) / trade.open_rate
                target_ratio = float(self.target_r_mult.value) * r_ratio
                if current_profit >= target_ratio:
                    return f"target_{self.target_r_mult.value}r_short"

        # 2. Trend-Failure Exit on completed bar
        if self.enable_trend_failure_exit.value:
            current_bar = dataframe.iloc[-1]
            ema_slow_val = current_bar.get("ema_slow", 0.0)
            if ema_slow_val > 0:
                if not trade.is_short and current_bar["close"] < ema_slow_val:
                    return "trend_failure_exit_long"
                elif trade.is_short and current_bar["close"] > ema_slow_val:
                    return "trend_failure_exit_short"

        # 3. Time Exit: 24 completed bars (24 * 15m = 360 min)
        elapsed_seconds = (current_time - trade.open_date_utc).total_seconds()
        max_duration_seconds = int(self.max_holding_bars.value) * 15 * 60
        if elapsed_seconds >= max_duration_seconds:
            return f"time_exit_{self.max_holding_bars.value}_bars"

        return None
