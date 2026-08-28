from __future__ import annotations

from datetime import date

import pytest

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import GetHoldingsInput, GetPortfolioInput, GetTransactionsInput
from quantagent.tools.portfolio import get_holdings, get_portfolio, get_transactions
from tests.unit.tools.builders import (
    DEFAULT_PORTFOLIO_ID,
    build_holding,
    build_tool_context,
    build_transaction,
)


async def test_get_portfolio_returns_metadata() -> None:
    ctx = build_tool_context().for_call(tool_name="get_portfolio", inputs_hash="h")

    result = await get_portfolio(GetPortfolioInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)

    assert result.portfolio_id == DEFAULT_PORTFOLIO_ID
    assert result.benchmark_ticker == "SPY"
    assert result.provenance.tool_name == "get_portfolio"


async def test_get_portfolio_raises_for_unknown_portfolio() -> None:
    ctx = build_tool_context().for_call(tool_name="get_portfolio", inputs_hash="h")

    with pytest.raises(ToolValidationError):
        await get_portfolio(GetPortfolioInput(portfolio_id="nope"), ctx)


async def test_get_holdings_reports_weights_summing_to_one() -> None:
    ctx = build_tool_context(
        holdings=[
            build_holding(ticker="AAA", quantity=10.0, cost_basis=100.0),
            build_holding(ticker="BBB", quantity=5.0, cost_basis=200.0),
        ]
    ).for_call(tool_name="get_holdings", inputs_hash="h")

    result = await get_holdings(GetHoldingsInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)

    assert {h.ticker for h in result.holdings} == {"AAA", "BBB"}
    total_weight = sum(h.weight.value for h in result.holdings)
    assert total_weight == pytest.approx(1.0)


async def test_get_holdings_single_holding_has_full_weight_and_usd_market_value() -> None:
    ctx = build_tool_context(
        holdings=[build_holding(ticker="AAA", quantity=10.0, cost_basis=100.0)]
    ).for_call(tool_name="get_holdings", inputs_hash="h")

    result = await get_holdings(GetHoldingsInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)

    holding = result.holdings[0]
    assert holding.weight.value == pytest.approx(1.0)
    assert holding.market_value.value > 0
    assert holding.market_value.unit == "usd"
    assert holding.quantity == pytest.approx(10.0)
    assert holding.cost_basis_usd == pytest.approx(100.0)


async def test_get_holdings_raises_for_empty_holdings() -> None:
    ctx = build_tool_context(holdings=[]).for_call(tool_name="get_holdings", inputs_hash="h")

    with pytest.raises(ToolValidationError):
        await get_holdings(GetHoldingsInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)


async def test_get_transactions_filters_by_date_range() -> None:
    ctx = build_tool_context(
        transactions=[
            build_transaction(ticker="AAA", trade_date=date(2024, 1, 1)),
            build_transaction(ticker="BBB", trade_date=date(2025, 6, 1)),
        ]
    ).for_call(tool_name="get_transactions", inputs_hash="h")

    result = await get_transactions(
        GetTransactionsInput(
            portfolio_id=DEFAULT_PORTFOLIO_ID, start=date(2024, 1, 1), end=date(2024, 12, 31)
        ),
        ctx,
    )

    assert [t.ticker for t in result.transactions] == ["AAA"]


async def test_get_transactions_returns_empty_list_when_none_match() -> None:
    ctx = build_tool_context(transactions=[]).for_call(
        tool_name="get_transactions", inputs_hash="h"
    )

    result = await get_transactions(
        GetTransactionsInput(
            portfolio_id=DEFAULT_PORTFOLIO_ID, start=date(2024, 1, 1), end=date(2024, 12, 31)
        ),
        ctx,
    )

    assert result.transactions == []
