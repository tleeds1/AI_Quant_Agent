"""Integration test for `KenFrenchFactorDataProvider` against REAL, trimmed
Ken French archives (served via `respx` -- no live network call).

The trimmed fixtures were downloaded from the real, live URLs and truncated
to their last ~400 data rows (see git history for the one-off capture
script used) -- preamble, header, and copyright footer are untouched.

The magnitude assertion below is a deliberate tripwire: if Ken French ever
changed their unit convention (percent -> decimal) upstream, this test
would fail loudly instead of every factor beta silently coming back ~100x
off with no exception anywhere.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import respx

from quantagent.data.providers.factors import KEN_FRENCH_BASE_URL, KenFrenchFactorDataProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"
_FF5_URL = f"{KEN_FRENCH_BASE_URL}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM_URL = f"{KEN_FRENCH_BASE_URL}/F-F_Momentum_Factor_daily_CSV.zip"


@respx.mock
async def test_real_trimmed_archives_parse_into_a_decimal_valued_panel() -> None:
    ff5_bytes = (FIXTURES_DIR / "ff5_daily_trimmed.zip").read_bytes()
    mom_bytes = (FIXTURES_DIR / "momentum_daily_trimmed.zip").read_bytes()
    respx.get(_FF5_URL).mock(return_value=httpx.Response(200, content=ff5_bytes))
    respx.get(_MOM_URL).mock(return_value=httpx.Response(200, content=mom_bytes))
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2024, 1, 1), date(2027, 1, 1))

    assert panel.n_obs > 0
    assert list(panel.returns.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    # Tripwire: real daily factor returns are always well under 50% in magnitude.
    # A value >= 0.5 here means the percent->decimal conversion silently broke.
    assert panel.returns.abs().max().max() < 0.5
    assert panel.risk_free.abs().max() < 0.5
