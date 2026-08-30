"""tests/unit/data/providers/test_edgar_provider.py"""

from __future__ import annotations

from datetime import date

import httpx
import pytest
import respx

from quantagent.contracts.errors import (
    InsufficientDataError,
    ProviderUnavailableError,
    UnknownTickerError,
)
from quantagent.data.providers.edgar import EdgarFilingsProvider

_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions"
_TEST_USER_AGENT = "Test Agent test@example.com"

_TICKERS_JSON = {
    "0": {"cik_str": 1045810, "ticker": "NVDA", "title": "NVIDIA CORP"},
    "1": {"cik_str": 320193, "ticker": "AAPL", "title": "APPLE INC"},
}

_SUBMISSIONS_JSON = {
    "cik": "1045810",
    "name": "NVIDIA CORP",
    "filings": {
        "recent": {
            "accessionNumber": [
                "0001045810-24-000123",
                "0001045810-23-000099",
                "0001045810-24-000200",
            ],
            "filingDate": ["2024-02-21", "2023-02-22", "2024-05-22"],
            "reportDate": ["2024-01-28", "2023-01-29", "2024-04-28"],
            "form": ["10-K", "10-K", "10-Q"],
            "primaryDocument": ["nvda-20240128.htm", "nvda-20230129.htm", "nvda-20240428.htm"],
        }
    },
}


def _mock_tickers() -> None:
    respx.get(_TICKERS_URL).mock(return_value=httpx.Response(200, json=_TICKERS_JSON))


def _mock_submissions() -> None:
    respx.get(f"{_SUBMISSIONS_URL}/CIK0001045810.json").mock(
        return_value=httpx.Response(200, json=_SUBMISSIONS_JSON)
    )


def _provider() -> EdgarFilingsProvider:
    return EdgarFilingsProvider(user_agent=_TEST_USER_AGENT)


def test_requires_a_non_blank_user_agent() -> None:
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        EdgarFilingsProvider(user_agent="")


@respx.mock
async def test_resolve_cik_zero_pads_to_ten_digits() -> None:
    _mock_tickers()
    cik = await _provider().resolve_cik("nvda")
    assert cik == "0001045810"


@respx.mock
async def test_resolve_cik_unknown_ticker_raises() -> None:
    _mock_tickers()
    with pytest.raises(UnknownTickerError):
        await _provider().resolve_cik("NOPE")


@respx.mock
async def test_list_filings_filters_by_form_type_and_sorts_newest_first() -> None:
    _mock_tickers()
    _mock_submissions()
    refs = await _provider().list_filings("NVDA", ["10-K"])
    assert [r.accession_no for r in refs] == ["0001045810-24-000123", "0001045810-23-000099"]
    assert refs[0].filed_at == date(2024, 2, 21)
    assert refs[0].period_of_report == date(2024, 1, 28)
    assert refs[0].company_name == "NVIDIA CORP"


@respx.mock
async def test_list_filings_filters_by_since_date() -> None:
    _mock_tickers()
    _mock_submissions()
    refs = await _provider().list_filings("NVDA", ["10-K"], since=date(2024, 1, 1))
    assert [r.accession_no for r in refs] == ["0001045810-24-000123"]


@respx.mock
async def test_list_filings_respects_limit() -> None:
    _mock_tickers()
    _mock_submissions()
    refs = await _provider().list_filings("NVDA", ["10-K", "10-Q"], limit=1)
    assert len(refs) == 1
    assert refs[0].accession_no == "0001045810-24-000200"  # newest overall


@respx.mock
async def test_list_filings_no_match_raises_insufficient_data() -> None:
    _mock_tickers()
    _mock_submissions()
    with pytest.raises(InsufficientDataError):
        await _provider().list_filings("NVDA", ["8-K"])


@respx.mock
async def test_filing_ref_source_url_strips_dashes_and_leading_zeros() -> None:
    _mock_tickers()
    _mock_submissions()
    refs = await _provider().list_filings("NVDA", ["10-K"])
    ref = refs[0]
    assert ref.source_url == (
        "https://www.sec.gov/Archives/edgar/data/1045810/000104581024000123/nvda-20240128.htm"
    )


@respx.mock
async def test_fetch_primary_document_returns_html() -> None:
    _mock_tickers()
    _mock_submissions()
    ref = (await _provider().list_filings("NVDA", ["10-K"]))[0]
    respx.get(ref.source_url).mock(return_value=httpx.Response(200, text="<html>filing</html>"))
    document = await _provider().fetch_primary_document(ref)
    assert document.html == "<html>filing</html>"
    assert document.ref == ref


@respx.mock
async def test_persistent_5xx_converts_to_provider_unavailable() -> None:
    respx.get(_TICKERS_URL).mock(return_value=httpx.Response(500))
    with pytest.raises(ProviderUnavailableError):
        await _provider().resolve_cik("NVDA")


@respx.mock
async def test_404_is_not_retried_and_converts_to_provider_unavailable() -> None:
    route = respx.get(_TICKERS_URL).mock(return_value=httpx.Response(404))
    with pytest.raises(ProviderUnavailableError):
        await _provider().resolve_cik("NVDA")
    assert route.call_count == 1  # a permanent error must not be retried


@respx.mock
async def test_transient_5xx_then_success_recovers_via_retry() -> None:
    route = respx.get(_TICKERS_URL).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=_TICKERS_JSON)]
    )
    cik = await _provider().resolve_cik("NVDA")
    assert cik == "0001045810"
    assert route.call_count == 2


@respx.mock
async def test_requests_carry_the_configured_user_agent() -> None:
    _mock_tickers()
    await _provider().resolve_cik("NVDA")
    request = respx.calls.last.request
    assert request.headers["User-Agent"] == _TEST_USER_AGENT
