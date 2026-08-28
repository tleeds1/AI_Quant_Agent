"""Integration smoke test for the M2-refactored CLI: `_run()` now does zero
business logic of its own (three `registry.invoke(...)` calls + render), so
this test exists to prove that orchestration -- and the DB wiring -- work
together against real Postgres, not to re-verify tool-level math (that's
`tests/integration/tools/test_tools_end_to_end.py`'s job).
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from scripts.print_risk_report import DEFAULT_LOOKBACK_DAYS, _parse_args, _run
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository

TICKERS = ["AAPL", "MSFT", "NVDA"]
BENCHMARK = "SPY"


def _synthetic_yfinance_frame(n: int, seed: int = 42) -> pd.DataFrame:
    """A MultiIndex(ticker, field) frame matching yfinance's real
    `group_by="ticker"` shape, filled with a synthetic random-walk price
    path -- exercises the real pipeline (repository -> provider -> registry
    -> tool adapters -> render) end-to-end without a live network call
    (guideline.md §10.3).
    """
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=date.today(), periods=n)
    frames = {}
    for ticker in [*TICKERS, BENCHMARK]:
        closes = 100.0 * (1.0 + rng.normal(0.0003, 0.015, size=n)).cumprod()
        frames[ticker] = pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
            index=index,
        )
    return pd.concat(frames, axis=1)


async def test_print_risk_report_full_pipeline_against_real_postgres(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    monkeypatch.setattr(
        YFinancePriceProvider,
        "_download_sync",
        staticmethod(lambda *a, **k: _synthetic_yfinance_frame(DEFAULT_LOOKBACK_DAYS)),
    )

    portfolios = PortfolioRepository(session_factory)
    await portfolios.upsert_portfolio(
        id="pf_smoke",
        tenant_id="tenant_smoke",
        name="Smoke Test Portfolio",
        base_currency="USD",
        benchmark_ticker=BENCHMARK,
        mandate_constraints={},
    )
    await portfolios.upsert_holdings(
        "pf_smoke",
        "tenant_smoke",
        date.today() - timedelta(days=1),
        [("AAPL", 10.0, 150.0), ("MSFT", 5.0, 280.0), ("NVDA", 8.0, 90.0)],
    )

    args = _parse_args(["--portfolio-id", "pf_smoke", "--tenant-id", "tenant_smoke"])
    output = await _run(args)

    assert "Smoke Test Portfolio" in output
    assert "portfolio_var_95_1d" in output
    assert "max_drawdown" in output
    assert "beta" in output
    for ticker in TICKERS:
        assert ticker in output
