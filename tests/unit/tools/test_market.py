from __future__ import annotations

import dataclasses
from datetime import date

import pytest

from quantagent.contracts.tools import GetFundamentalsInput, GetPricesInput, GetReturnsInput
from quantagent.tools.market import get_fundamentals, get_prices, get_returns
from tests.unit.tools.builders import build_tool_context


async def test_get_prices_returns_one_observation_per_trading_day() -> None:
    ctx = build_tool_context().for_call(tool_name="get_prices", inputs_hash="h")

    result = await get_prices(
        GetPricesInput(tickers=["AAA", "BBB"], start=date(2024, 1, 1), end=date(2024, 12, 31)),
        ctx,
    )

    assert result.tickers == ["AAA", "BBB"]
    assert len(result.observations) > 0
    assert set(result.observations[0].prices.keys()) == {"AAA", "BBB"}


async def test_get_returns_produces_simple_returns_by_default() -> None:
    ctx = build_tool_context().for_call(tool_name="get_returns", inputs_hash="h")

    returns = await get_returns(GetReturnsInput(tickers=["AAA"], lookback_days=100), ctx)

    assert returns.n_obs > 0
    assert returns.kind == "simple"
    assert returns.kind == "simple"


async def test_get_returns_log_kind_is_respected() -> None:
    ctx = build_tool_context().for_call(tool_name="get_returns", inputs_hash="h")

    result = await get_returns(GetReturnsInput(tickers=["AAA"], kind="log"), ctx)

    assert result.kind == "log"


async def test_get_fundamentals_wraps_numeric_fields_as_metric_values() -> None:
    ctx = build_tool_context(sector_by_ticker={"AAA": "Healthcare"}).for_call(
        tool_name="get_fundamentals", inputs_hash="h"
    )

    result = await get_fundamentals(GetFundamentalsInput(ticker="AAA"), ctx)

    assert result.sector == "Healthcare"
    assert result.revenue_ttm is not None
    assert result.revenue_ttm.unit == "usd"
    assert result.net_margin is not None
    assert result.net_margin.unit == "ratio"
    assert result.pe_ratio is not None


async def test_get_fundamentals_omits_metric_when_value_is_none(monkeypatch) -> None:
    ctx = build_tool_context()
    original = ctx.fundamentals.get_fundamentals

    async def _no_revenue(ticker: str):
        fundamentals = await original(ticker)
        return dataclasses.replace(fundamentals, revenue_ttm=None)

    monkeypatch.setattr(ctx.fundamentals, "get_fundamentals", _no_revenue)
    bound = ctx.for_call(tool_name="get_fundamentals", inputs_hash="h")

    result = await get_fundamentals(GetFundamentalsInput(ticker="AAA"), bound)

    assert result.revenue_ttm is None
    assert result.net_margin is not None
    assert result.net_margin.value == pytest.approx(0.2)
