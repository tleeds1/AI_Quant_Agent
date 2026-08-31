from __future__ import annotations

import pytest

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import OptimizePortfolioInput, SimulateTradeImpactInput
from quantagent.tools.optimization_tools import optimize_portfolio, simulate_trade_impact
from tests.unit.tools.builders import (
    DEFAULT_PORTFOLIO_ID,
    build_holding,
    build_tool_context,
)


async def test_optimize_portfolio_success() -> None:
    ctx = build_tool_context(
        holdings=[
            build_holding(ticker="AAPL", quantity=100.0, cost_basis=150.0),
            build_holding(ticker="MSFT", quantity=50.0, cost_basis=250.0),
            build_holding(ticker="GOOG", quantity=80.0, cost_basis=120.0),
        ]
    ).for_call(tool_name="optimize_portfolio", inputs_hash="hash_opt")

    inp = OptimizePortfolioInput(
        portfolio_id=DEFAULT_PORTFOLIO_ID,
        objective="min_variance",
        max_concentration=0.5,
    )

    result = await optimize_portfolio(inp, ctx)

    assert result.portfolio_id == DEFAULT_PORTFOLIO_ID
    assert result.objective == "min_variance"
    assert len(result.trades) == 3
    assert result.current_volatility.value > 0
    assert result.target_volatility.value > 0

    # Check that constraints are respected on proposed trades
    for trade in result.trades:
        assert trade.target_weight <= 0.50001
        assert trade.action in ("BUY", "SELL", "HOLD")


async def test_optimize_portfolio_raises_on_empty_holdings() -> None:
    ctx = build_tool_context(holdings=[]).for_call(
        tool_name="optimize_portfolio", inputs_hash="hash_opt"
    )

    inp = OptimizePortfolioInput(
        portfolio_id=DEFAULT_PORTFOLIO_ID,
        objective="min_variance",
    )

    with pytest.raises(ToolValidationError):
        await optimize_portfolio(inp, ctx)


async def test_simulate_trade_impact_success() -> None:
    ctx = build_tool_context(
        holdings=[
            build_holding(ticker="AAPL", quantity=100.0, cost_basis=150.0),
            build_holding(ticker="MSFT", quantity=50.0, cost_basis=250.0),
        ]
    ).for_call(tool_name="simulate_trade_impact", inputs_hash="hash_sim")

    inp = SimulateTradeImpactInput(
        portfolio_id=DEFAULT_PORTFOLIO_ID,
        target_weights={"AAPL": 0.6, "MSFT": 0.4},
    )

    result = await simulate_trade_impact(inp, ctx)

    assert result.portfolio_id == DEFAULT_PORTFOLIO_ID
    assert result.total_trade_value_usd.value >= 0.0
    assert result.estimated_cost_usd.value >= 0.0
    assert result.turnover_pct.value >= 0.0
