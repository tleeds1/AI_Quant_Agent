from __future__ import annotations

import pytest

from quantagent.contracts.tools import (
    CalculateComponentVarInput,
    CalculateCvarInput,
    CalculateMaxDrawdownInput,
    CalculatePortfolioVarInput,
    CalculateTrackingErrorInput,
    GetPortfolioBetaInput,
)
from quantagent.tools.risk import (
    calculate_component_var,
    calculate_cvar,
    calculate_max_drawdown,
    calculate_portfolio_var,
    calculate_tracking_error,
    get_portfolio_beta,
)
from tests.unit.tools.builders import DEFAULT_PORTFOLIO_ID, build_holding, build_tool_context

_TWO_HOLDINGS = [
    build_holding(ticker="AAA", quantity=10.0, cost_basis=100.0),
    build_holding(ticker="BBB", quantity=5.0, cost_basis=200.0),
]


async def test_calculate_portfolio_var_returns_a_positive_ratio_metric() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="calculate_portfolio_var", inputs_hash="h"
    )

    result = await calculate_portfolio_var(
        CalculatePortfolioVarInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    assert result.value > 0
    assert result.unit == "ratio"
    assert result.provenance.tool_name == "calculate_portfolio_var"


async def test_calculate_portfolio_var_horizon_scales_with_sqrt_h() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="calculate_portfolio_var", inputs_hash="h"
    )

    var_1d = await calculate_portfolio_var(
        CalculatePortfolioVarInput(portfolio_id=DEFAULT_PORTFOLIO_ID, horizon_days=1), ctx
    )
    var_4d = await calculate_portfolio_var(
        CalculatePortfolioVarInput(portfolio_id=DEFAULT_PORTFOLIO_ID, horizon_days=4), ctx
    )

    assert var_4d.value == pytest.approx(var_1d.value * 2.0, abs=1e-9)


async def test_calculate_cvar_is_at_least_var() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="calculate_cvar", inputs_hash="h"
    )

    var_result = await calculate_portfolio_var(
        CalculatePortfolioVarInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )
    cvar_result = await calculate_cvar(CalculateCvarInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)

    assert cvar_result.value >= var_result.value


async def test_calculate_component_var_sums_to_portfolio_var() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="calculate_component_var", inputs_hash="h"
    )

    result = await calculate_component_var(
        CalculateComponentVarInput(portfolio_id=DEFAULT_PORTFOLIO_ID, method="parametric"), ctx
    )

    total = sum(c.contribution.value for c in result.components)
    assert total == pytest.approx(result.portfolio_var.value, abs=1e-6)


async def test_calculate_max_drawdown_is_non_positive() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="calculate_max_drawdown", inputs_hash="h"
    )

    result = await calculate_max_drawdown(
        CalculateMaxDrawdownInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    assert result.drawdown.value <= 0.0
    assert result.trough_date >= result.peak_date


async def test_get_portfolio_beta_reports_beta_and_downside_beta() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="get_portfolio_beta", inputs_hash="h"
    )

    result = await get_portfolio_beta(GetPortfolioBetaInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)

    assert result.benchmark_ticker == "SPY"
    assert result.beta.unit == "ratio"
    assert result.downside_beta.unit == "ratio"


async def test_get_portfolio_beta_honors_explicit_benchmark_override() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="get_portfolio_beta", inputs_hash="h"
    )

    result = await get_portfolio_beta(
        GetPortfolioBetaInput(portfolio_id=DEFAULT_PORTFOLIO_ID, benchmark_ticker="QQQ"), ctx
    )

    assert result.benchmark_ticker == "QQQ"


async def test_calculate_tracking_error_is_non_negative() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="calculate_tracking_error", inputs_hash="h"
    )

    result = await calculate_tracking_error(
        CalculateTrackingErrorInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    assert result.value >= 0.0
