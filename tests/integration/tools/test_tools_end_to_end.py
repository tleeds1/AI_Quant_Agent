"""Integration test: every registered tool invoked through the REAL registry,
REAL `ToolContext`, REAL `PortfolioRepository` (against Dockerized Postgres),
REAL `YFinancePriceProvider`/`YFinanceFundamentalsProvider`/
`KenFrenchFactorDataProvider`, and REAL `CacheClient` (against Dockerized
Redis) -- with only the network-hitting sync fetch methods monkeypatched to
return real-shaped synthetic data.

Same rationale as M1's `tests/integration/scripts/test_print_risk_report_smoke.py`
and the M2 provider integration tests: guideline.md §10.3 forbids any test
(unit or integration) from touching a live market-data endpoint, so this
proves the real wiring end-to-end (repository -> providers -> registry ->
tool adapters) without a live network call.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

import quantagent.tools  # noqa: F401 -- import side effect: populates the registry
from quantagent.data.cache import CacheClient
from quantagent.data.providers.factors import FACTOR_COLUMNS, KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import (
    PortfolioRepository,
    TransactionInput,
)
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry

PORTFOLIO_ID = "pf_tools_e2e"
TENANT_ID = "tenant_tools_e2e"
TICKERS = ["AAPL", "MSFT", "NVDA"]
BENCHMARK = "SPY"
LOOKBACK_DAYS = 700
N_OBS = 500


def _synthetic_price_frame(n: int, seed: int = 42) -> pd.DataFrame:
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


def _synthetic_fundamentals_info(sector: str) -> dict[str, object]:
    return {
        "quoteType": "EQUITY",
        "symbol": "FAKE",
        "shortName": "Fake Co",
        "sector": sector,
        "industry": "Software",
        "totalRevenue": 1_000_000_000,
        "profitMargins": 0.2,
        "trailingPE": 22.5,
        "mostRecentQuarter": None,
    }


def _synthetic_factor_frame(n: int, seed: int = 99) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=date.today(), periods=n)
    data = rng.normal(0.0, 0.008, size=(n, len(FACTOR_COLUMNS) + 1))  # +1 for RF
    return pd.DataFrame(data, index=index, columns=[*FACTOR_COLUMNS, "risk_free"])


async def _seeded_context(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> ToolContext:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id=PORTFOLIO_ID,
        tenant_id=TENANT_ID,
        name="Tools E2E Portfolio",
        base_currency="USD",
        benchmark_ticker=BENCHMARK,
        mandate_constraints={},
    )
    await repository.upsert_holdings(
        PORTFOLIO_ID,
        TENANT_ID,
        date.today() - timedelta(days=1),
        [("AAPL", 10.0, 150.0), ("MSFT", 5.0, 280.0), ("NVDA", 8.0, 90.0)],
    )
    await repository.replace_transactions(
        PORTFOLIO_ID,
        TENANT_ID,
        [TransactionInput("AAPL", "buy", 10.0, 150.0, date.today() - timedelta(days=1))],
    )

    price_frame = _synthetic_price_frame(LOOKBACK_DAYS)
    monkeypatch.setattr(
        YFinancePriceProvider, "_download_sync", staticmethod(lambda *a, **k: price_frame)
    )
    monkeypatch.setattr(
        YFinanceFundamentalsProvider,
        "_fetch_info_sync",
        staticmethod(lambda ticker: _synthetic_fundamentals_info("Technology")),
    )
    factor_frame = _synthetic_factor_frame(N_OBS)

    async def _fake_get_factor_returns(self: object, start: date, end: date):
        from quantagent.data.providers.factors import FactorReturnPanel

        return FactorReturnPanel(
            returns=factor_frame[FACTOR_COLUMNS],
            risk_free=factor_frame["risk_free"],
            factors=list(FACTOR_COLUMNS),
            as_of=factor_frame.index.max().date(),
            source="fake_ken_french",
            warnings=[],
            n_obs=len(factor_frame),
        )

    monkeypatch.setattr(KenFrenchFactorDataProvider, "get_factor_returns", _fake_get_factor_returns)

    cache = CacheClient.from_settings()
    return ToolContext(
        tenant_id=TENANT_ID,
        portfolios=repository,
        prices=YFinancePriceProvider(cache=cache),
        fundamentals=YFinanceFundamentalsProvider(cache=cache),
        factors=KenFrenchFactorDataProvider(cache=cache),
        cache=cache,
    )


_ARGS_BY_TOOL = {
    "get_portfolio": {"portfolio_id": PORTFOLIO_ID},
    "get_holdings": {"portfolio_id": PORTFOLIO_ID},
    "get_transactions": {
        "portfolio_id": PORTFOLIO_ID,
        "start": "2000-01-01",
        "end": "2100-01-01",
    },
    "get_prices": {
        "tickers": TICKERS,
        "start": (date.today() - timedelta(days=30)).isoformat(),
        "end": date.today().isoformat(),
    },
    "get_returns": {"tickers": TICKERS, "lookback_days": 200},
    "get_fundamentals": {"ticker": "AAPL"},
    "get_sector_exposure": {"portfolio_id": PORTFOLIO_ID},
    "get_factor_exposure": {"portfolio_id": PORTFOLIO_ID},
    "get_correlation_matrix": {"tickers": TICKERS, "lookback_days": 200},
    "get_concentration_metrics": {"portfolio_id": PORTFOLIO_ID},
    "calculate_portfolio_var": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
    "calculate_cvar": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
    "calculate_component_var": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
    "calculate_max_drawdown": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
    "get_portfolio_beta": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
    "calculate_tracking_error": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
    "compute_expression": {"expr": "a / b - 1", "refs": {"a": 1.05, "b": 1.0}},
    "generate_risk_report": {"portfolio_id": PORTFOLIO_ID, "lookback_days": 400},
}


async def test_every_tool_succeeds_against_real_postgres_and_real_provider_classes(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    ctx = await _seeded_context(monkeypatch, session_factory)

    assert set(_ARGS_BY_TOOL) == {spec.name for spec in registry.list_tools()}
    for tool_name, args in _ARGS_BY_TOOL.items():
        result = await registry.invoke(tool_name, args, ctx)
        assert result is not None, f"{tool_name} returned None"
