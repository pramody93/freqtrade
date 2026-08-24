"""Delta Exchange and Delta Exchange India subclass"""

import logging

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Exchange
from freqtrade.exchange.exchange_types import FtHas


logger = logging.getLogger(__name__)


class Delta(Exchange):
    """Delta Exchange class.

    Contains adjustments needed for Freqtrade to work with Delta Exchange
    and Delta Exchange India.
    """

    _ft_has: FtHas = {
        "order_time_in_force": ["GTC", "IOC", "FOK", "PO"],
        "stoploss_on_exchange": False,
        "marketOrderRequiresPrice": False,
        "ohlcv_has_history": True,
        "trades_has_history": False,
        "ws_enabled": False,
        "exchange_has_overrides": {
            "fetchFundingRateHistory": False,
            "fetchFundingRates": False,
        },
    }

    _ft_has_futures: FtHas = {
        "uses_leverage_tiers": False,
        "funding_fee_candle_limit": 500,
        "mark_ohlcv_price": "mark",
    }

    _supported_trading_mode_margin_pairs: list[tuple[TradingMode, MarginMode]] = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        (TradingMode.FUTURES, MarginMode.CROSS),
    ]

    @property
    def _ccxt_config(self) -> dict:
        config = super()._ccxt_config
        ex_name = self._config.get("exchange", {}).get("name", "").lower()
        if ex_name in ("deltaindia", "delta_india", "delta-india", "deltain"):
            urls = config.setdefault("urls", {})
            api_urls = urls.setdefault("api", {})
            api_urls.setdefault("public", "https://api.india.delta.exchange")
            api_urls.setdefault("private", "https://api.india.delta.exchange")
        return config


class DeltaIndia(Delta):
    """Delta Exchange India subclass.

    Pre-configures endpoints for the India-specific Delta Exchange platform (india.delta.exchange).
    """

    @property
    def _ccxt_config(self) -> dict:
        config = super()._ccxt_config
        urls = config.setdefault("urls", {})
        api_urls = urls.setdefault("api", {})
        api_urls.setdefault("public", "https://api.india.delta.exchange")
        api_urls.setdefault("private", "https://api.india.delta.exchange")
        return config


Deltaindia = DeltaIndia
