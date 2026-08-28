"""Integration test for `YFinanceFundamentalsProvider` against a REAL captured
Yahoo Finance `get_info()` response -- no live network call.

Same rationale as `test_yfinance_price_provider_fixtures.py` (M1): yfinance
prefers `curl_cffi` for TLS/browser impersonation, which vcrpy cannot
intercept (it patches `requests`/`urllib3`, not libcurl bindings). A static
JSON fixture of a real, previously-captured `yf.Ticker(...).get_info()`
response achieves the same goal -- the provider's field-mapping logic runs
against a real response shape, with no live network dependency.

To refresh a fixture: re-run `yf.Ticker("AAPL").get_info()` /
`yf.Ticker("SPY").get_info()` and overwrite the JSON (see git history for the
one-off capture script used).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> dict[str, Any]:
    with (FIXTURES_DIR / name).open(encoding="utf-8") as f:
        return dict(json.load(f))


async def test_get_fundamentals_parses_real_equity_response_shape(monkeypatch) -> None:
    info = _load_fixture("get_fundamentals_aapl.json")
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: info)
    )
    provider = YFinanceFundamentalsProvider()

    result = await provider.get_fundamentals("AAPL")

    assert result.sector == "Technology"
    assert result.industry
    assert result.revenue_ttm is not None and result.revenue_ttm > 0
    assert result.net_margin is not None and -1 < result.net_margin < 1
    assert result.pe_ratio is not None and result.pe_ratio > 0


async def test_get_fundamentals_parses_real_etf_response_shape(monkeypatch) -> None:
    info = _load_fixture("get_fundamentals_etf_spy.json")
    monkeypatch.setattr(
        YFinanceFundamentalsProvider, "_fetch_info_sync", staticmethod(lambda t: info)
    )
    provider = YFinanceFundamentalsProvider()

    result = await provider.get_fundamentals("SPY")

    assert result.sector is None
    assert result.revenue_ttm is None
