from __future__ import annotations

import asyncio
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, cast, runtime_checkable

import pandas as pd
import yfinance as yf

from quantagent.contracts.errors import (
    InsufficientDataError,
    ProviderUnavailableError,
    UnknownTickerError,
)
from quantagent.data.cache import CacheClient, compute_inputs_hash

# architecture.md §13.2: "prices 15 min intra-day / EOD until next close"
PRICE_CACHE_TTL_S = 900


@dataclass(frozen=True, slots=True)
class PricePanel:
    """A wide, float64, calendar-indexed adjusted-close price panel."""

    prices: pd.DataFrame
    tickers: list[str]
    unresolved_tickers: list[str]
    as_of: date
    source: str
    warnings: list[str]
    n_obs: int


@runtime_checkable
class PriceProvider(Protocol):
    async def get_prices(
        self, tickers: Sequence[str], start: date, end: date, *, adjusted: bool = True
    ) -> PricePanel: ...


class YFinancePriceProvider:
    """`PriceProvider` backed by yfinance, with Redis cache-aside."""

    def __init__(self, cache: CacheClient | None = None) -> None:
        self._cache = cache

    async def get_prices(
        self, tickers: Sequence[str], start: date, end: date, *, adjusted: bool = True
    ) -> PricePanel:
        key = self._cache_key(tickers, start, end, adjusted)
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return _panel_from_cache_bytes(cached)

        panel = await self._fetch(tickers, start, end, adjusted)

        if self._cache is not None:
            await self._cache.set(key, _panel_to_cache_bytes(panel), ttl_s=PRICE_CACHE_TTL_S)
        return panel

    @staticmethod
    def _cache_key(tickers: Sequence[str], start: date, end: date, adjusted: bool) -> str:
        inputs_hash = compute_inputs_hash(
            tickers=sorted(tickers),
            start=start.isoformat(),
            end=end.isoformat(),
            adjusted=adjusted,
            source="yfinance",
        )
        return f"quantagent:v1:prices:{inputs_hash}"

    async def _fetch(
        self, tickers: Sequence[str], start: date, end: date, adjusted: bool
    ) -> PricePanel:
        ticker_list = list(tickers)
        try:
            raw = await asyncio.to_thread(self._download_sync, ticker_list, start, end, adjusted)
        except Exception as exc:
            raise ProviderUnavailableError(f"yfinance request failed: {exc}") from exc

        close_panel = _extract_close_panel(raw, ticker_list)
        resolved = [t for t in ticker_list if close_panel[t].notna().any()]
        unresolved = [t for t in ticker_list if t not in resolved]
        if not resolved:
            raise UnknownTickerError(f"none of the requested tickers resolved: {ticker_list}")

        panel = close_panel[resolved].dropna(how="all")
        if panel.empty:
            raise InsufficientDataError(f"no price data returned for tickers {ticker_list}")

        warnings = [f"unresolved ticker: {t}" for t in unresolved]
        return PricePanel(
            prices=panel.astype("float64"),
            tickers=resolved,
            unresolved_tickers=unresolved,
            as_of=panel.index.max().date(),
            source="yfinance",
            warnings=warnings,
            n_obs=len(panel),
        )

    @staticmethod
    def _download_sync(tickers: list[str], start: date, end: date, adjusted: bool) -> pd.DataFrame:
        """Blocking yfinance call, wrapped in `asyncio.to_thread` by the caller.

        yfinance's underlying HTTP client (`requests`) is synchronous; a real
        async HTTP client for the same data would mean reimplementing
        yfinance's parsing layer for no benefit at M1's scale, so the
        provider boundary here is a thin sync-to-async shim rather than a
        native async client.
        """
        result = yf.download(
            tickers, start=start, end=end, auto_adjust=adjusted, progress=False, group_by="ticker"
        )
        return cast(pd.DataFrame, result)


def _extract_close_panel(raw: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Flatten yfinance's `(ticker, field)` MultiIndex columns to a wide
    ticker -> adjusted-close DataFrame, one column per requested ticker
    (all-NaN for any ticker yfinance didn't resolve).
    """
    available = set(raw.columns.get_level_values(0))
    series = {
        ticker: (raw[ticker]["Close"] if ticker in available else pd.Series(dtype="float64"))
        for ticker in tickers
    }
    return pd.concat(series, axis=1)


def _panel_to_cache_bytes(panel: PricePanel) -> bytes:
    payload = {
        "prices": panel.prices.to_json(orient="split", date_format="iso"),
        "tickers": panel.tickers,
        "unresolved_tickers": panel.unresolved_tickers,
        "as_of": panel.as_of.isoformat(),
        "source": panel.source,
        "warnings": panel.warnings,
        "n_obs": panel.n_obs,
    }
    return json.dumps(payload).encode()


def _panel_from_cache_bytes(data: bytes) -> PricePanel:
    payload = json.loads(data.decode())
    prices = pd.read_json(io.StringIO(payload["prices"]), orient="split")
    return PricePanel(
        prices=prices.astype("float64"),
        tickers=payload["tickers"],
        unresolved_tickers=payload["unresolved_tickers"],
        as_of=date.fromisoformat(payload["as_of"]),
        source=payload["source"],
        warnings=payload["warnings"],
        n_obs=payload["n_obs"],
    )
