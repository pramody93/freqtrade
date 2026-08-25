"""
Compression Box Expansion Strategy
Based on "Systematic Strategy Research: BTC Perpetual Futures Rulebook" (Strategy 2)

Concept: Volatility Compression + Box Range Breakout + Volume Confirmation
Timeframe: 30 minutes
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


class CompressionBoxExpansion(IStrategy):
    """
    Compression Box Expansion Strategy (30m)
    - 12-Bar Volatility Box (U, L, W)
    - 1.5x Volume Confirmation
    - Fixed 3.00R Target + +1.00R Activation Trailing Stop
    - 3-Bar Breakout Failure Exit & 16-Bar Time Exit
    """

    INTERFACE_VERSION = 3
    can_short: bool = True
    timeframe = "30m"

    minimal_roi = {"0": 10.0}
    stoploss = -0.10
    use_custom_stoploss = True

    process_only_new_candles = True
    startup_candle_count: int = 100

    # -------------------------------------------------------------
    # Hyperoptable Parameters
    # -------------------------------------------------------------
    box_len = IntParameter(10, 16, default=12, space="buy", optimize=True)
    compression_ratio = DecimalParameter(0.70, 0.90, default=0.80, decimals=2, space="buy", optimize=True)
    min_box_width_mult = DecimalParameter(1.0, 1.5, default=1.25, decimals=2, space="buy", optimize=False)
    max_box_width_mult = DecimalParameter(2.0, 3.0, default=2.50, decimals=2, space="buy", optimize=False)
    vol_mult = DecimalParameter(1.3, 1.8, default=1.5, decimals=2, space="buy", optimize=True)

    # Exits
    target_r_mult = DecimalParameter(2.0, 4.0, default=3.0, decimals=2, space="sell", optimize=True)
    trail_activation_r = DecimalParameter(0.8, 1.5, default=1.0, decimals=2, space="sell", optimize=False)
    trail_atr_mult = DecimalParameter(1.2, 2.0, default=1.5, decimals=2, space="sell", optimize=True)
    max_holding_bars = IntParameter(12, 24, default=16, space="sell", optimize=False)

    plot_config = {
        "main_plot": {
            "box_u": {"color": "#00e676"},
            "box_l": {"color": "#ff1744"},
        },
        "subplots": {
            "Volume & Vol SMA": {
                "volume": {"color": "#42a5f5"},
                "vol_sma": {"color": "#ffca28"},
            },
        },
    }

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Calculate ATR(5), ATR(20), Volume SMA(20), and 12-bar price box (U, L, W)"""
        # ATRs
        dataframe["atr_5"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=5)
        dataframe["atr_20"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=20)

        # Volume SMA
        dataframe["vol_sma"] = ta.SMA(dataframe["volume"], timeperiod=20)

        # 12-Bar Box over t-12 through t-1 (excluding current bar t)
        bl = int(self.box_len.value)
        dataframe["box_u"] = dataframe["high"].shift(1).rolling(bl).max()
        dataframe["box_l"] = dataframe["low"].shift(1).rolling(bl).min()
        dataframe["box_w"] = dataframe["box_u"] - dataframe["box_l"]

        # Shifted ATR(20) and ATR(5) for conditions referencing t-1
        dataframe["atr_20_prev"] = dataframe["atr_20"].shift(1)
        dataframe["atr_5_prev"] = dataframe["atr_5"].shift(1)
        dataframe["vol_sma_prev"] = dataframe["vol_sma"].shift(1)

        # Compression check on t-1
        dataframe["is_compressed"] = dataframe["atr_5_prev"] <= (self.compression_ratio.value * dataframe["atr_20_prev"])
        dataframe["box_width_ok"] = (
            (dataframe["box_w"] >= (self.min_box_width_mult.value * dataframe["atr_20_prev"]))
            & (dataframe["box_w"] <= (self.max_box_width_mult.value * dataframe["atr_20_prev"]))
        )

        # Signal bar checks
        dataframe["open_in_box"] = (dataframe["open"] >= dataframe["box_l"]) & (dataframe["open"] <= dataframe["box_u"])
        dataframe["candle_range"] = dataframe["high"] - dataframe["low"]
        dataframe["range_ok"] = (
            (dataframe["candle_range"] >= (0.80 * dataframe["atr_20_prev"]))
            & (dataframe["candle_range"] <= (2.00 * dataframe["atr_20_prev"]))
        )

        # Closing Location = (close - low) / (high - low)
        hl_diff = dataframe["high"] - dataframe["low"]
        dataframe["closing_loc"] = np.where(hl_diff > 0, (dataframe["close"] - dataframe["low"]) / hl_diff, 0.5)

        # Volume confirmation
        dataframe["vol_ok"] = dataframe["volume"] >= (self.vol_mult.value * dataframe["vol_sma_prev"])

        # Stop and Risk calculations
        dataframe["stop_long_price"] = dataframe["box_u"] - (0.75 * dataframe["atr_20_prev"])
        dataframe["stop_short_price"] = dataframe["box_l"] + (0.75 * dataframe["atr_20_prev"])

        dataframe["risk_long"] = dataframe["close"] - dataframe["stop_long_price"]
        dataframe["risk_short"] = dataframe["stop_short_price"] - dataframe["close"]

        dataframe["risk_ok_long"] = (
            (dataframe["risk_long"] >= (0.75 * dataframe["atr_20_prev"]))
            & (dataframe["risk_long"] <= (1.75 * dataframe["atr_20_prev"]))
        )
        dataframe["risk_ok_short"] = (
            (dataframe["risk_short"] >= (0.75 * dataframe["atr_20_prev"]))
            & (dataframe["risk_short"] <= (1.75 * dataframe["atr_20_prev"]))
        )

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Evaluate Compression Box Breakout on completed signal candle"""
        # Long Breakout
        long_condition = (
            (dataframe["is_compressed"])
            & (dataframe["box_width_ok"])
            & (dataframe["open_in_box"])
            & (dataframe["close"] >= (dataframe["box_u"] + 0.10 * dataframe["atr_20_prev"]))
            & (dataframe["range_ok"])
            & (dataframe["closing_loc"] >= 0.75)
            & (dataframe["vol_ok"])
            & (dataframe["risk_ok_long"])
        )

        dataframe.loc[long_condition & ~long_condition.shift(1).fillna(False), "enter_long"] = 1

        # Short Breakout
        short_condition = (
            (dataframe["is_compressed"])
            & (dataframe["box_width_ok"])
            & (dataframe["open_in_box"])
            & (dataframe["close"] <= (dataframe["box_l"] - 0.10 * dataframe["atr_20_prev"]))
            & (dataframe["range_ok"])
            & (dataframe["closing_loc"] <= 0.25)
            & (dataframe["vol_ok"])
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
        """
        Stop Loss Controller:
        - Initial Stop: U - 0.75 * ATR(20) (Long) / L + 0.75 * ATR(20) (Short)
        - Dynamic Trailing Stop: Activated after +1.00R profit
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        signal_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if signal_candle.empty:
            return None

        last_signal = signal_candle.iloc[-1]

        if not trade.is_short:
            stop_price = last_signal.get("stop_long_price", 0.0)
            if stop_price <= 0 or stop_price >= trade.open_rate:
                return None
            initial_r_ratio = (trade.open_rate - stop_price) / trade.open_rate

            # Trailing stop after +1.00R profit
            if current_profit >= float(self.trail_activation_r.value) * initial_r_ratio:
                # Find highest close since entry
                candles_in_trade = dataframe.loc[dataframe["date"] >= trade.open_date_utc]
                highest_close = candles_in_trade["close"].max() if not candles_in_trade.empty else current_rate
                current_atr = dataframe["atr_20"].iloc[-1]
                trail_stop_price = highest_close - (float(self.trail_atr_mult.value) * current_atr)
                dynamic_stop = max(stop_price, trail_stop_price)
                dynamic_ratio = (dynamic_stop - trade.open_rate) / trade.open_rate
                return stoploss_from_open(dynamic_ratio, current_profit, is_short=False, leverage=trade.leverage)

            initial_stop_ratio = (stop_price - trade.open_rate) / trade.open_rate
            return stoploss_from_open(initial_stop_ratio, current_profit, is_short=False, leverage=trade.leverage)
        else:
            stop_price = last_signal.get("stop_short_price", 0.0)
            if stop_price <= 0 or stop_price <= trade.open_rate:
                return None
            initial_r_ratio = (stop_price - trade.open_rate) / trade.open_rate

            # Trailing stop after +1.00R profit
            if current_profit >= float(self.trail_activation_r.value) * initial_r_ratio:
                candles_in_trade = dataframe.loc[dataframe["date"] >= trade.open_date_utc]
                lowest_close = candles_in_trade["close"].min() if not candles_in_trade.empty else current_rate
                current_atr = dataframe["atr_20"].iloc[-1]
                trail_stop_price = lowest_close + (float(self.trail_atr_mult.value) * current_atr)
                dynamic_stop = min(stop_price, trail_stop_price)
                dynamic_ratio = (trade.open_rate - dynamic_stop) / trade.open_rate
                return stoploss_from_open(dynamic_ratio, current_profit, is_short=True, leverage=trade.leverage)

            initial_stop_ratio = (trade.open_rate - stop_price) / trade.open_rate
            return stoploss_from_open(initial_stop_ratio, current_profit, is_short=True, leverage=trade.leverage)

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
        1. Fixed Target at 3.00R
        2. Breakout-Failure Exit: During first 3 completed bars after entry, if close <= U (Long) or close >= L (Short)
        3. Time Exit: After 16 completed 30m bars (8 hours)
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe.empty or trade.open_rate <= 0:
            return None

        signal_candle = dataframe.loc[dataframe["date"] <= trade.open_date_utc]
        if signal_candle.empty:
            return None

        last_signal = signal_candle.iloc[-1]
        box_u = last_signal.get("box_u", 0.0)
        box_l = last_signal.get("box_l", 0.0)

        # 1. 3.00R Profit Target
        if not trade.is_short:
            stop_price = last_signal.get("stop_long_price", 0.0)
            if stop_price > 0 and stop_price < trade.open_rate:
                r_ratio = (trade.open_rate - stop_price) / trade.open_rate
                if current_profit >= float(self.target_r_mult.value) * r_ratio:
                    return f"target_{self.target_r_mult.value}r_long"
        else:
            stop_price = last_signal.get("stop_short_price", 0.0)
            if stop_price > 0 and stop_price > trade.open_rate:
                r_ratio = (stop_price - trade.open_rate) / trade.open_rate
                if current_profit >= float(self.target_r_mult.value) * r_ratio:
                    return f"target_{self.target_r_mult.value}r_short"

        # 2. Breakout-Failure Exit (First 3 completed bars)
        candles_in_trade = dataframe.loc[dataframe["date"] >= trade.open_date_utc]
        bars_elapsed = len(candles_in_trade)
        if 1 <= bars_elapsed <= 3 and not candles_in_trade.empty:
            latest_bar = candles_in_trade.iloc[-1]
            if not trade.is_short and box_u > 0 and latest_bar["close"] <= box_u:
                return "breakout_failure_exit_long"
            elif trade.is_short and box_l > 0 and latest_bar["close"] >= box_l:
                return "breakout_failure_exit_short"

        # 3. Time Exit: 16 completed 30m bars (16 * 30m = 480 min)
        elapsed_seconds = (current_time - trade.open_date_utc).total_seconds()
        max_duration_seconds = int(self.max_holding_bars.value) * 30 * 60
        if elapsed_seconds >= max_duration_seconds:
            return f"time_exit_{self.max_holding_bars.value}_bars"

        return None
