from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import pytest

from quantagent.contracts.errors import ProviderUnavailableError, UnknownTickerError
from quantagent.data.providers.fundamentals import (
    FUNDAMENTALS_CACHE_TTL_S,
    YFinanceFundamentalsProvider,
)

_AAPL_INFO = {
    "quoteType": "EQUITY",
    "symbol": "AAPL",
    "shortName": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "totalRevenue": 466822987776,
    "profitMargins": 0.27618998,
    "trailingPE": 35.579792,
    "mostRecentQuarter": 1782518400,
}

_SPY_INFO = {
    "quoteType": "ETF",
    "symbol": "SPY",
    "shortName": "SPDR S&P 500",
    "sector": None,
    "industry": None,
    "totalRevenue": None,
    "profitMargins": None,
    "trailingPE": None,
    "mostRecentQuarter": None,
}


async def test_get_fundamentals_maps_real_response_shape(monkeypatch) -> None:
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: _AAPL_INFO)
    )
    provider = YFinanceFundamentalsProvider()

    result = await provider.get_fundamentals("AAPL")

    assert result.sector == "Technology"
    assert result.industry == "Consumer Electronics"
    assert result.revenue_ttm == pytest.approx(466822987776)
    assert result.net_margin == pytest.approx(0.27618998)
    assert result.pe_ratio == pytest.approx(35.579792)
    assert result.as_of == datetime.fromtimestamp(1782518400, tz=UTC).date()


async def test_get_fundamentals_falls_back_to_today_when_no_quarter(monkeypatch) -> None:
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: _SPY_INFO)
    )
    provider = YFinanceFundamentalsProvider()

    result = await provider.get_fundamentals("SPY")

    assert result.as_of == date.today()
    assert result.sector is None
    assert result.revenue_ttm is None


async def test_get_fundamentals_raises_unknown_ticker_for_near_empty_response(monkeypatch) -> None:
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: {})
    )
    provider = YFinanceFundamentalsProvider()

    with pytest.raises(UnknownTickerError):
        await provider.get_fundamentals("NOPE")


async def test_get_fundamentals_wraps_fetch_failure_as_provider_unavailable(monkeypatch) -> None:
    def _raise(ticker: str) -> dict:
        raise ConnectionError("rate limited")

    monkeypatch.setattr(YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(_raise))
    provider = YFinanceFundamentalsProvider()

    with pytest.raises(ProviderUnavailableError):
        await provider.get_fundamentals("AAPL")


async def test_non_numeric_financial_fields_become_none_not_a_crash(monkeypatch) -> None:
    info = dict(_AAPL_INFO, totalRevenue="not a number", profitMargins=True)
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: info)
    )
    provider = YFinanceFundamentalsProvider()

    result = await provider.get_fundamentals("AAPL")

    assert result.revenue_ttm is None
    assert result.net_margin is None  # bool is an int subclass, explicitly rejected


async def test_get_fundamentals_uses_cache_on_hit_without_calling_fetch(monkeypatch) -> None:
    fetch_calls: list[str] = []
    monkeypatch.setattr(
        YFinanceFundamentalsProvider,
        "_fetch_info_sync",
        staticmethod(lambda t: fetch_calls.append(t) or _AAPL_INFO),
    )
    cache = AsyncMock()
    cache.get.return_value = None
    provider = YFinanceFundamentalsProvider(cache=cache)

    first = await provider.get_fundamentals("AAPL")
    assert len(fetch_calls) == 1
    cached_bytes = cache.set.call_args.args[1]

    cache.get.return_value = cached_bytes
    second = await provider.get_fundamentals("AAPL")

    assert len(fetch_calls) == 1
    assert second == first


async def test_get_fundamentals_caches_with_configured_ttl(monkeypatch) -> None:
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: _AAPL_INFO)
    )
    cache = AsyncMock()
    cache.get.return_value = None
    provider = YFinanceFundamentalsProvider(cache=cache)

    await provider.get_fundamentals("AAPL")

    cache.set.assert_awaited_once()
    assert cache.set.call_args.kwargs["ttl_s"] == FUNDAMENTALS_CACHE_TTL_S


async def test_batch_resolves_available_and_reports_the_rest_unresolved(monkeypatch) -> None:
    info_by_ticker = {"AAPL": _AAPL_INFO, "SPY": _SPY_INFO, "NOPE": {}}
    monkeypatch.setattr(
        YFinanceFundamentalsProvider,
        "_fetch_info_sync",
        staticmethod(lambda t: info_by_ticker[t]),
    )
    provider = YFinanceFundamentalsProvider()

    panel = await provider.get_fundamentals_batch(["AAPL", "SPY", "NOPE"])

    assert set(panel.tickers) == {"AAPL", "SPY"}
    assert panel.unresolved_tickers == ["NOPE"]
    assert any("NOPE" in w for w in panel.warnings)


async def test_batch_raises_unknown_ticker_when_nothing_resolves(monkeypatch) -> None:
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: {})
    )
    provider = YFinanceFundamentalsProvider()

    with pytest.raises(UnknownTickerError):
        await provider.get_fundamentals_batch(["NOPE", "ALSO_NOPE"])


async def test_batch_deduplicates_repeated_tickers(monkeypatch) -> None:
    fetch_calls: list[str] = []
    monkeypatch.setattr(
        YFinanceFundamentalsProvider,
        "_fetch_info_sync",
        staticmethod(lambda t: fetch_calls.append(t) or _AAPL_INFO),
    )
    provider = YFinanceFundamentalsProvider()

    panel = await provider.get_fundamentals_batch(["AAPL", "AAPL", "AAPL"])

    assert len(fetch_calls) == 1
    assert panel.tickers == ["AAPL"]


async def test_batch_propagates_provider_unavailable_error(monkeypatch) -> None:
    def _raise(ticker: str) -> dict:
        raise ConnectionError("rate limited")

    monkeypatch.setattr(YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(_raise))
    provider = YFinanceFundamentalsProvider()

    with pytest.raises(ProviderUnavailableError):
        await provider.get_fundamentals_batch(["AAPL", "MSFT"])
