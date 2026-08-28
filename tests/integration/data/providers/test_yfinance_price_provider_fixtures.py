"""Integration test for `YFinancePriceProvider` against REAL captured Yahoo
Finance response shapes -- no live network call.

Deviation from the original plan (recorded here per guideline.md §0 --
"if a library is unavailable ... state the conflict explicitly"): the plan
called for vcrpy HTTP-level cassettes. yfinance now prefers `curl_cffi` for
TLS/browser impersonation (see `yfinance/_http.py`), which talks to Yahoo via
libcurl bindings rather than `requests`/`urllib3` -- vcrpy patches the latter,
not the former, so it cannot intercept yfinance's actual HTTP calls (verified:
`vcr.use_cassette()` recorded no cassette file while a real network call
silently went through). Static CSV fixtures of real, previously-captured
`yf.download(..., group_by="ticker")` output achieve the same goal -- the
provider's parsing/resolution logic runs against a real response shape, and
no test touches a live market data API (guideline.md §10.3) -- without
depending on an HTTP-mocking layer that doesn't actually work here.

To refresh a fixture: re-run the two `yf.download(...)` calls used to
generate `fixtures/get_prices_happy_path.csv` and
`fixtures/get_prices_partial_unresolved.csv` (see git history for the
one-off script used) and overwrite the CSV.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from quantagent.data.providers.prices import YFinancePriceProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load_fixture(name: str) -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / name, header=[0, 1], index_col=0, parse_dates=True)


async def test_get_prices_parses_real_happy_path_response_shape(monkeypatch) -> None:
    frame = _load_fixture("get_prices_happy_path.csv")
    monkeypatch.setattr(
        YFinancePriceProvider, "_download_sync", staticmethod(lambda *a, **k: frame)
    )
    provider = YFinancePriceProvider()

    panel = await provider.get_prices(["AAPL", "MSFT"], date(2024, 1, 2), date(2024, 1, 12))

    assert panel.tickers == ["AAPL", "MSFT"]
    assert panel.unresolved_tickers == []
    assert panel.n_obs == 8
    assert panel.prices.dtypes.eq("float64").all()


async def test_get_prices_parses_real_partial_unresolved_response_shape(monkeypatch) -> None:
    frame = _load_fixture("get_prices_partial_unresolved.csv")
    monkeypatch.setattr(
        YFinancePriceProvider, "_download_sync", staticmethod(lambda *a, **k: frame)
    )
    provider = YFinancePriceProvider()

    panel = await provider.get_prices(
        ["AAPL", "NOTATICKERXYZ123"], date(2024, 1, 2), date(2024, 1, 12)
    )

    assert panel.tickers == ["AAPL"]
    assert panel.unresolved_tickers == ["NOTATICKERXYZ123"]
    assert any("NOTATICKERXYZ123" in w for w in panel.warnings)
