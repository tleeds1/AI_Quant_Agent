"""MCP server smoke test: constructs the real server and calls its
`list_tools`/`call_tool` handlers directly (bypassing the stdio transport,
which is I/O plumbing owned by the `mcp` SDK, not our code) against real
Postgres and real provider classes with monkeypatched network fetches --
satisfies the M2 DoD's "the MCP server is usable from an MCP client" without
a flaky subprocess-based test.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from mcp import types
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quantagent.data.cache import CacheClient
from quantagent.data.providers.factors import KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.tools.context import ToolContext
from quantagent.tools.mcp_server import build_call_tool_handler, build_server, handle_list_tools
from quantagent.tools.registry import registry

PORTFOLIO_ID = "pf_mcp_smoke"
TENANT_ID = "tenant_mcp_smoke"
TICKERS = ["AAPL", "MSFT"]
BENCHMARK = "SPY"


def _synthetic_price_frame(n: int, seed: int = 7) -> pd.DataFrame:
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


async def _seeded_context(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> ToolContext:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id=PORTFOLIO_ID,
        tenant_id=TENANT_ID,
        name="MCP Smoke Portfolio",
        base_currency="USD",
        benchmark_ticker=BENCHMARK,
        mandate_constraints={},
    )
    await repository.upsert_holdings(
        PORTFOLIO_ID,
        TENANT_ID,
        date.today() - timedelta(days=1),
        [("AAPL", 10.0, 150.0), ("MSFT", 5.0, 280.0)],
    )
    monkeypatch.setattr(
        YFinancePriceProvider,
        "_download_sync",
        staticmethod(lambda *a, **k: _synthetic_price_frame(700)),
    )
    cache = CacheClient.from_settings()
    return ToolContext(
        tenant_id=TENANT_ID,
        portfolios=repository,
        prices=YFinancePriceProvider(cache=cache),
        fundamentals=YFinanceFundamentalsProvider(cache=cache),
        factors=KenFrenchFactorDataProvider(cache=cache),
        cache=cache,
    )


async def test_list_tools_returns_every_registered_tool_with_a_valid_schema(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    ctx = await _seeded_context(monkeypatch, session_factory)
    build_server(ctx)  # constructs without raising

    result = await handle_list_tools(None, None)  # type: ignore[arg-type]

    names = {t.name for t in result.tools}
    assert names == {spec.name for spec in registry.list_tools()}
    for tool in result.tools:
        assert "properties" in tool.input_schema


async def test_call_tool_returns_real_data_for_a_valid_call(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    ctx = await _seeded_context(monkeypatch, session_factory)
    handler = build_call_tool_handler(ctx)
    params = types.CallToolRequestParams(
        name="get_portfolio", arguments={"portfolio_id": PORTFOLIO_ID}
    )

    result = await handler(None, params)  # type: ignore[arg-type]

    assert result.is_error is not True
    assert "MCP Smoke Portfolio" in result.content[0].text


async def test_call_tool_reports_an_mcp_error_for_an_unknown_portfolio(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    ctx = await _seeded_context(monkeypatch, session_factory)
    handler = build_call_tool_handler(ctx)
    params = types.CallToolRequestParams(
        name="get_portfolio", arguments={"portfolio_id": "does_not_exist"}
    )

    result = await handler(None, params)  # type: ignore[arg-type]

    assert result.is_error is True


async def test_call_tool_reports_an_mcp_error_for_an_unknown_tool_name(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    ctx = await _seeded_context(monkeypatch, session_factory)
    handler = build_call_tool_handler(ctx)
    params = types.CallToolRequestParams(name="not_a_real_tool", arguments={})

    result = await handler(None, params)  # type: ignore[arg-type]

    assert result.is_error is True


async def test_call_tool_computes_a_real_risk_metric_end_to_end(
    monkeypatch, session_factory: async_sessionmaker[AsyncSession]
) -> None:
    ctx = await _seeded_context(monkeypatch, session_factory)
    handler = build_call_tool_handler(ctx)
    params = types.CallToolRequestParams(
        name="calculate_portfolio_var", arguments={"portfolio_id": PORTFOLIO_ID}
    )

    result = await handler(None, params)  # type: ignore[arg-type]

    assert result.is_error is not True
    assert '"metric_id":"portfolio_var' in result.content[0].text
