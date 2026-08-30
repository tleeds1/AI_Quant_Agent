"""data/providers/edgar.py -- SEC EDGAR filings provider (architecture.md
§4.7).

Direct HTTP against EDGAR's public JSON/HTML endpoints, not
`sec-edgar-downloader`: httpx is respx-mockable in tests, matching
`data/providers/factors.py`'s pattern (yfinance's curl_cffi transport is
not respx-mockable, hence that provider's different test strategy).

SEC's fair-access policy requires every request to carry a descriptive
`User-Agent` identifying the requester -- enforced at construction, not
left to the caller to remember.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any, Protocol, runtime_checkable

import httpx
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

from quantagent.config import settings
from quantagent.contracts.errors import (
    InsufficientDataError,
    ProviderUnavailableError,
    UnknownTickerError,
)
from quantagent.data.cache import CacheClient, compute_inputs_hash

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_BASE_URL = "https://data.sec.gov/submissions"
SEC_ARCHIVES_BASE_URL = "https://www.sec.gov/Archives/edgar/data"

TICKER_MAP_CACHE_TTL_S = 7 * 86_400
SUBMISSIONS_CACHE_TTL_S = 86_400
# A filed accession's primary document never changes once published --
# cached with a long TTL so replaying a trace (architecture.md §9.2) reads
# the same bytes, not a re-fetch that could theoretically differ.
FILING_DOCUMENT_CACHE_TTL_S = 30 * 86_400

_HTTP_TIMEOUT_S = 30.0
_MAX_ATTEMPTS = 3
_DEFAULT_LIST_LIMIT = 20


def _is_transient_http_error(exc: BaseException) -> bool:
    """Only rate-limit (429) and server errors (5xx) are retried -- a 404
    (unknown CIK/document) is permanent and retrying it would be the
    guideline.md §13 anti-pattern "retrying non-transient errors."
    """
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code == 429 or exc.response.status_code >= 500
    return isinstance(exc, httpx.TransportError)


@dataclass(frozen=True, slots=True)
class FilingRef:
    cik: str  # zero-padded to 10 digits
    accession_no: str  # e.g. "0000320193-24-000123"
    form_type: str
    filed_at: date
    period_of_report: date | None
    primary_document: str
    company_name: str

    @property
    def source_url(self) -> str:
        accession_no_no_dashes = self.accession_no.replace("-", "")
        cik_no_leading_zeros = str(int(self.cik))
        return (
            f"{SEC_ARCHIVES_BASE_URL}/{cik_no_leading_zeros}/{accession_no_no_dashes}/"
            f"{self.primary_document}"
        )


@dataclass(frozen=True, slots=True)
class FilingDocument:
    ref: FilingRef
    html: str
    fetched_at: datetime


@runtime_checkable
class FilingsProvider(Protocol):
    async def resolve_cik(self, ticker: str) -> str: ...

    async def list_filings(
        self, ticker: str, form_types: list[str], since: date | None = None, limit: int = 20
    ) -> list[FilingRef]: ...

    async def fetch_primary_document(self, ref: FilingRef) -> FilingDocument: ...


class EdgarFilingsProvider:
    """`FilingsProvider` backed by SEC EDGAR's public JSON/HTML endpoints."""

    def __init__(
        self,
        cache: CacheClient | None = None,
        user_agent: str | None = None,
        tickers_url: str = SEC_TICKERS_URL,
        submissions_base_url: str = SEC_SUBMISSIONS_BASE_URL,
    ) -> None:
        resolved_user_agent = user_agent if user_agent is not None else settings.sec_user_agent
        if not resolved_user_agent.strip():
            raise ValueError(
                "EdgarFilingsProvider requires a non-blank SEC_USER_AGENT (SEC's fair-access "
                "policy mandates a descriptive contact string on every request)"
            )
        self._cache = cache
        self._user_agent = resolved_user_agent
        self._tickers_url = tickers_url
        self._submissions_base_url = submissions_base_url.rstrip("/")

    async def resolve_cik(self, ticker: str) -> str:
        ticker_map = await self._load_ticker_map()
        cik = ticker_map.get(ticker.strip().upper())
        if cik is None:
            raise UnknownTickerError(f"ticker {ticker!r} not found in SEC company_tickers.json")
        return cik

    async def list_filings(
        self,
        ticker: str,
        form_types: list[str],
        since: date | None = None,
        limit: int = _DEFAULT_LIST_LIMIT,
    ) -> list[FilingRef]:
        cik = await self.resolve_cik(ticker)
        submissions = await self._load_submissions(cik)
        company_name = str(submissions.get("name", ticker))
        recent = submissions.get("filings", {}).get("recent", {})
        refs = _parse_recent_filings(recent, cik=cik, company_name=company_name)

        wanted_forms = {form.upper() for form in form_types}
        matching = [
            ref
            for ref in refs
            if ref.form_type.upper() in wanted_forms and (since is None or ref.filed_at >= since)
        ]
        if not matching:
            raise InsufficientDataError(
                f"no {sorted(wanted_forms)} filings for {ticker} "
                f"{f'since {since}' if since else '(any date)'} in EDGAR submissions"
            )
        matching.sort(key=lambda ref: ref.filed_at, reverse=True)
        return matching[:limit]

    async def fetch_primary_document(self, ref: FilingRef) -> FilingDocument:
        key = _document_cache_key(ref)
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return FilingDocument(
                    ref=ref, html=cached.decode("utf-8"), fetched_at=datetime.now(UTC)
                )
        html = await self._get_text(ref.source_url)
        if self._cache is not None:
            await self._cache.set(key, html.encode("utf-8"), ttl_s=FILING_DOCUMENT_CACHE_TTL_S)
        return FilingDocument(ref=ref, html=html, fetched_at=datetime.now(UTC))

    async def _load_ticker_map(self) -> dict[str, str]:
        key = _ticker_map_cache_key()
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return dict(json.loads(cached.decode("utf-8")))

        raw = await self._get_json(self._tickers_url)
        ticker_map = {
            str(entry["ticker"]).upper(): f"{int(entry['cik_str']):010d}" for entry in raw.values()
        }
        if self._cache is not None:
            await self._cache.set(
                key, json.dumps(ticker_map).encode("utf-8"), ttl_s=TICKER_MAP_CACHE_TTL_S
            )
        return ticker_map

    async def _load_submissions(self, cik: str) -> dict[str, Any]:
        key = _submissions_cache_key(cik)
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return dict(json.loads(cached.decode("utf-8")))

        raw = await self._get_json(f"{self._submissions_base_url}/CIK{cik}.json")
        if self._cache is not None:
            await self._cache.set(
                key, json.dumps(raw).encode("utf-8"), ttl_s=SUBMISSIONS_CACHE_TTL_S
            )
        return dict(raw)

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._user_agent}

    @retry(
        retry=retry_if_exception(_is_transient_http_error),
        wait=wait_exponential(multiplier=0.5, max=5),
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        reraise=True,
    )
    async def _request(self, url: str) -> httpx.Response:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S) as client:
            response = await client.get(url, headers=self._headers())
            response.raise_for_status()
            return response

    async def _get_json(self, url: str) -> Any:
        try:
            response = await self._request(url)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"EDGAR request failed for {url}: {exc}") from exc
        return response.json()

    async def _get_text(self, url: str) -> str:
        try:
            response = await self._request(url)
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(f"EDGAR request failed for {url}: {exc}") from exc
        return response.text


def _parse_recent_filings(
    recent: dict[str, Any], *, cik: str, company_name: str
) -> list[FilingRef]:
    """EDGAR's submissions JSON stores `filings.recent` as parallel arrays
    (one list per field, same index = same filing), not a list of objects.
    """
    forms: list[str] = recent.get("form", [])
    filing_dates: list[str] = recent.get("filingDate", [])
    report_dates: list[str] = recent.get("reportDate", [])
    accession_numbers: list[str] = recent.get("accessionNumber", [])
    primary_documents: list[str] = recent.get("primaryDocument", [])

    refs = []
    for i in range(len(forms)):
        report_date_str = report_dates[i] if i < len(report_dates) else ""
        refs.append(
            FilingRef(
                cik=cik,
                accession_no=accession_numbers[i],
                form_type=forms[i],
                filed_at=date.fromisoformat(filing_dates[i]),
                period_of_report=date.fromisoformat(report_date_str) if report_date_str else None,
                primary_document=primary_documents[i],
                company_name=company_name,
            )
        )
    return refs


def _ticker_map_cache_key() -> str:
    inputs_hash = compute_inputs_hash(source="edgar_company_tickers")
    return f"quantagent:v1:edgar:ticker_map:{inputs_hash}"


def _submissions_cache_key(cik: str) -> str:
    inputs_hash = compute_inputs_hash(source="edgar_submissions", cik=cik)
    return f"quantagent:v1:edgar:submissions:{inputs_hash}"


def _document_cache_key(ref: FilingRef) -> str:
    inputs_hash = compute_inputs_hash(
        source="edgar_document",
        accession_no=ref.accession_no,
        primary_document=ref.primary_document,
    )
    return f"quantagent:v1:edgar:document:{inputs_hash}"
