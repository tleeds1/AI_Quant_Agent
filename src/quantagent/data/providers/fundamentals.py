from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, cast, runtime_checkable

import yfinance as yf

from quantagent.contracts.errors import ProviderUnavailableError, UnknownTickerError
from quantagent.data.cache import CacheClient, compute_inputs_hash

# architecture.md §13.2 pins prices/filings/news but not fundamentals; 24h is a
# judgment call sized to how fast the data actually moves (quarterly reporting,
# with sector/industry effectively static).
FUNDAMENTALS_CACHE_TTL_S = 86_400

# Yahoo rate-limits aggressively, so batch fan-out is capped rather than unbounded.
MAX_CONCURRENT_FUNDAMENTALS_FETCHES = 8

_RESOLUTION_MARKER_KEYS = ("quoteType", "symbol", "shortName", "longName")


@dataclass(frozen=True, slots=True)
class Fundamentals:
    ticker: str
    as_of: date
    sector: str | None
    industry: str | None
    revenue_ttm: float | None
    net_margin: float | None
    pe_ratio: float | None
    source: str


@dataclass(frozen=True, slots=True)
class FundamentalsPanel:
    """Batch result, mirroring `PricePanel`'s resolved/unresolved shape."""

    fundamentals: dict[str, Fundamentals]
    tickers: list[str]
    unresolved_tickers: list[str]
    as_of: date
    source: str
    warnings: list[str]


@runtime_checkable
class FundamentalsProvider(Protocol):
    async def get_fundamentals(self, ticker: str) -> Fundamentals: ...

    async def get_fundamentals_batch(self, tickers: Sequence[str]) -> FundamentalsPanel: ...


class YFinanceFundamentalsProvider:
    """`FundamentalsProvider` backed by yfinance, with Redis cache-aside."""

    def __init__(self, cache: CacheClient | None = None) -> None:
        self._cache = cache
        self._fetch_semaphore = asyncio.Semaphore(MAX_CONCURRENT_FUNDAMENTALS_FETCHES)

    async def get_fundamentals(self, ticker: str) -> Fundamentals:
        key = self._cache_key(ticker)
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return _fundamentals_from_cache_bytes(cached)

        fundamentals = await self._fetch(ticker)

        if self._cache is not None:
            await self._cache.set(
                key, _fundamentals_to_cache_bytes(fundamentals), ttl_s=FUNDAMENTALS_CACHE_TTL_S
            )
        return fundamentals

    async def get_fundamentals_batch(self, tickers: Sequence[str]) -> FundamentalsPanel:
        """One cached, concurrency-capped `get_fundamentals` per unique ticker.

        Yahoo exposes no multi-symbol `quoteSummary` endpoint, so batching
        here buys concurrency and partial-failure tolerance, not fewer round
        trips. A ticker Yahoo doesn't know lands in `unresolved_tickers`
        instead of failing the call, because `get_sector_exposure` must
        still answer for the holdings that did resolve; a provider outage
        still propagates, since a silently missing sector is worse than a
        visible error.
        """
        requested = list(dict.fromkeys(tickers))
        results = await asyncio.gather(*(self._resolve_or_none(t) for t in requested))
        resolved = {t: f for t, f in zip(requested, results, strict=True) if f is not None}
        unresolved = [t for t in requested if t not in resolved]
        if not resolved:
            raise UnknownTickerError(f"none of the requested tickers resolved: {requested}")

        return FundamentalsPanel(
            fundamentals=resolved,
            tickers=list(resolved),
            unresolved_tickers=unresolved,
            as_of=max(f.as_of for f in resolved.values()),
            source="yfinance",
            warnings=[f"unresolved ticker: {t}" for t in unresolved],
        )

    async def _resolve_or_none(self, ticker: str) -> Fundamentals | None:
        async with self._fetch_semaphore:
            try:
                return await self.get_fundamentals(ticker)
            except UnknownTickerError:
                return None

    @staticmethod
    def _cache_key(ticker: str) -> str:
        inputs_hash = compute_inputs_hash(ticker=ticker, source="yfinance")
        return f"quantagent:v1:fundamentals:{inputs_hash}"

    async def _fetch(self, ticker: str) -> Fundamentals:
        try:
            info = await asyncio.to_thread(self._fetch_info_sync, ticker)
        except Exception as exc:
            raise ProviderUnavailableError(f"yfinance request failed for {ticker}: {exc}") from exc

        if not _is_resolved(info):
            raise UnknownTickerError(f"ticker does not resolve in yfinance: {ticker}")
        return _fundamentals_from_info(ticker, info)

    @staticmethod
    def _fetch_info_sync(ticker: str) -> dict[str, Any]:
        """Blocking yfinance call, wrapped in `asyncio.to_thread` by the caller.

        `get_info()` rather than `fast_info`: `fast_info` is limited to
        quote fields (price, market cap, day range) and carries none of
        `sector`, `industry`, `totalRevenue`, `profitMargins` or
        `trailingPE` -- which are the entire reason this provider exists.
        """
        return cast(dict[str, Any], yf.Ticker(ticker).get_info())


def _is_resolved(info: dict[str, Any]) -> bool:
    """yfinance returns a near-empty dict rather than raising for a symbol
    Yahoo doesn't know, so the presence of any identity field is the signal.
    """
    return any(info.get(key) for key in _RESOLUTION_MARKER_KEYS)


def _fundamentals_from_info(ticker: str, info: dict[str, Any]) -> Fundamentals:
    return Fundamentals(
        ticker=ticker,
        as_of=_as_of_from_info(info),
        sector=_optional_str(info.get("sector")),
        industry=_optional_str(info.get("industry")),
        revenue_ttm=_optional_float(info.get("totalRevenue")),
        net_margin=_optional_float(info.get("profitMargins")),
        pe_ratio=_optional_float(info.get("trailingPE")),
        source="yfinance",
    )


def _as_of_from_info(info: dict[str, Any]) -> date:
    """`mostRecentQuarter` (epoch seconds) is the fiscal date the TTM figures
    are true as of; fall back to the fetch date for instruments (ETFs,
    indices) that report no quarter.
    """
    epoch = info.get("mostRecentQuarter")
    if isinstance(epoch, (int, float)) and not isinstance(epoch, bool):
        return datetime.fromtimestamp(float(epoch), tz=UTC).date()
    return date.today()


def _optional_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _optional_str(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _fundamentals_to_cache_bytes(fundamentals: Fundamentals) -> bytes:
    payload = {
        "ticker": fundamentals.ticker,
        "as_of": fundamentals.as_of.isoformat(),
        "sector": fundamentals.sector,
        "industry": fundamentals.industry,
        "revenue_ttm": fundamentals.revenue_ttm,
        "net_margin": fundamentals.net_margin,
        "pe_ratio": fundamentals.pe_ratio,
        "source": fundamentals.source,
    }
    return json.dumps(payload).encode()


def _fundamentals_from_cache_bytes(data: bytes) -> Fundamentals:
    payload = json.loads(data.decode())
    return Fundamentals(
        ticker=payload["ticker"],
        as_of=date.fromisoformat(payload["as_of"]),
        sector=payload["sector"],
        industry=payload["industry"],
        revenue_ttm=payload["revenue_ttm"],
        net_margin=payload["net_margin"],
        pe_ratio=payload["pe_ratio"],
        source=payload["source"],
    )
