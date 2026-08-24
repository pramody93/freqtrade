from unittest.mock import MagicMock

import pytest

from freqtrade.enums import MarginMode, TradingMode
from freqtrade.exchange import Delta, DeltaIndia, Deltaindia, is_exchange_known_ccxt, validate_exchange
from freqtrade.resolvers import ExchangeResolver
from tests.conftest import get_patched_exchange


def test_delta_exchange_resolution(default_conf, mocker):
    conf_delta = default_conf.copy()
    conf_delta["exchange"]["name"] = "delta"
    exchange = get_patched_exchange(mocker, conf_delta, exchange="delta")
    assert isinstance(exchange, Delta)

    conf_india = default_conf.copy()
    conf_india["exchange"]["name"] = "deltaindia"
    exchange_india = get_patched_exchange(mocker, conf_india, exchange="deltaindia")
    assert isinstance(exchange_india, Delta)


def test_delta_supported_trading_modes():
    expected_modes = [
        (TradingMode.SPOT, MarginMode.NONE),
        (TradingMode.FUTURES, MarginMode.ISOLATED),
        (TradingMode.FUTURES, MarginMode.CROSS),
    ]
    assert Delta._supported_trading_mode_margin_pairs == expected_modes
    assert DeltaIndia._supported_trading_mode_margin_pairs == expected_modes
    assert Deltaindia._supported_trading_mode_margin_pairs == expected_modes


def test_delta_ft_has(default_conf, mocker):
    exchange = get_patched_exchange(mocker, default_conf, exchange="delta")
    assert exchange._ft_has["order_time_in_force"] == ["GTC", "IOC", "FOK", "PO"]
    assert exchange._ft_has["stoploss_on_exchange"] is False
    assert exchange._ft_has["ohlcv_has_history"] is True
    assert exchange._ft_has["trades_has_history"] is False

    conf_fut = default_conf.copy()
    conf_fut["trading_mode"] = TradingMode.FUTURES
    conf_fut["margin_mode"] = MarginMode.ISOLATED
    exchange_fut = get_patched_exchange(mocker, conf_fut, exchange="delta")
    assert exchange_fut._ft_has["uses_leverage_tiers"] is False
    assert exchange_fut._ft_has["funding_fee_candle_limit"] == 500
    assert exchange_fut._ft_has["mark_ohlcv_price"] == "mark"


def test_delta_india_urls(default_conf, mocker):
    conf_india = default_conf.copy()
    conf_india["exchange"]["name"] = "deltaindia"
    exchange_india = get_patched_exchange(mocker, conf_india, exchange="deltaindia")

    ccxt_config = exchange_india._ccxt_config
    assert "urls" in ccxt_config
    assert ccxt_config["urls"]["api"]["public"] == "https://api.india.delta.exchange"
    assert ccxt_config["urls"]["api"]["private"] == "https://api.india.delta.exchange"


def test_delta_validation():
    assert is_exchange_known_ccxt("delta") is True
    assert is_exchange_known_ccxt("deltaindia") is True
    assert is_exchange_known_ccxt("delta-india") is True

    valid_delta, reason_delta, _, _ = validate_exchange("delta")
    assert valid_delta is True

    valid_india, reason_india, _, _ = validate_exchange("deltaindia")
    assert valid_india is True
