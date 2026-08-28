from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from quantagent.contracts.errors import ProviderUnavailableError, UnknownTickerError
from quantagent.data.providers.prices import PRICE_CACHE_TTL_S, YFinancePriceProvider


def _multiindex_frame(closes_by_ticker: dict[str, list[float | None]], n: int = 5) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=n)
    frames = {}
    for ticker, closes in closes_by_ticker.items():
        frames[ticker] = pd.DataFrame(
            {
                "Open": closes,
                "High": closes,
                "Low": closes,
                "Close": closes,
                "Volume": [100] * n,
            },
            index=index,
        )
    return pd.concat(frames, axis=1)


async def test_get_prices_resolves_available_tickers_and_flags_unresolved(monkeypatch) -> None:
    frame = _multiindex_frame({"AAPL": [100.0, 101.0, 102.0, 103.0, 104.0], "NOPE": [None] * 5})
    monkeypatch.setattr(
        YFinancePriceProvider, "_download_sync", staticmethod(lambda *a, **k: frame)
    )
    provider = YFinancePriceProvider()

    panel = await provider.get_prices(["AAPL", "NOPE"], date(2024, 1, 2), date(2024, 1, 8))

    assert panel.tickers == ["AAPL"]
    assert panel.unresolved_tickers == ["NOPE"]
    assert any("NOPE" in w for w in panel.warnings)
    assert panel.n_obs == 5


async def test_get_prices_raises_unknown_ticker_when_nothing_resolves(monkeypatch) -> None:
    frame = _multiindex_frame({"NOPE": [None] * 5})
    monkeypatch.setattr(
        YFinancePriceProvider, "_download_sync", staticmethod(lambda *a, **k: frame)
    )
    provider = YFinancePriceProvider()

    with pytest.raises(UnknownTickerError):
        await provider.get_prices(["NOPE"], date(2024, 1, 2), date(2024, 1, 8))


async def test_get_prices_wraps_download_failure_as_provider_unavailable(monkeypatch) -> None:
    def _raise(*args: object, **kwargs: object) -> pd.DataFrame:
        raise ConnectionError("network down")

    monkeypatch.setattr(YFinancePriceProvider, "_download_sync", staticmethod(_raise))
    provider = YFinancePriceProvider()

    with pytest.raises(ProviderUnavailableError):
        await provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))


async def test_get_prices_uses_cache_on_hit_without_calling_download(monkeypatch) -> None:
    download_calls: list[int] = []
    frame = _multiindex_frame({"AAPL": [100.0] * 5})
    monkeypatch.setattr(
        YFinancePriceProvider,
        "_download_sync",
        staticmethod(lambda *a, **k: download_calls.append(1) or frame),
    )
    cache = AsyncMock()
    cache.get.return_value = None
    provider = YFinancePriceProvider(cache=cache)

    first_panel = await provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))
    assert len(download_calls) == 1
    cached_bytes = cache.set.call_args.args[1]

    cache.get.return_value = cached_bytes
    second_panel = await provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))

    assert len(download_calls) == 1  # not called again on cache hit
    assert second_panel.tickers == first_panel.tickers


async def test_get_prices_caches_with_configured_ttl(monkeypatch) -> None:
    frame = _multiindex_frame({"AAPL": [100.0] * 5})
    monkeypatch.setattr(
        YFinancePriceProvider, "_download_sync", staticmethod(lambda *a, **k: frame)
    )
    cache = AsyncMock()
    cache.get.return_value = None
    provider = YFinancePriceProvider(cache=cache)

    await provider.get_prices(["AAPL"], date(2024, 1, 2), date(2024, 1, 8))

    cache.set.assert_awaited_once()
    assert cache.set.call_args.kwargs["ttl_s"] == PRICE_CACHE_TTL_S
