from __future__ import annotations

import pytest

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import (
    GetConcentrationMetricsInput,
    GetCorrelationMatrixInput,
    GetFactorExposureInput,
    GetSectorExposureInput,
)
from quantagent.tools.exposure import (
    get_concentration_metrics,
    get_correlation_matrix,
    get_factor_exposure,
    get_sector_exposure,
)
from tests.unit.tools.builders import DEFAULT_PORTFOLIO_ID, build_holding, build_tool_context


async def test_get_sector_exposure_groups_holdings_by_sector() -> None:
    ctx = build_tool_context(
        holdings=[
            build_holding(ticker="AAA", quantity=10.0, cost_basis=100.0),
            build_holding(ticker="BBB", quantity=5.0, cost_basis=200.0),
        ],
        sector_by_ticker={"AAA": "Technology", "BBB": "Healthcare"},
    ).for_call(tool_name="get_sector_exposure", inputs_hash="h")

    result = await get_sector_exposure(
        GetSectorExposureInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    labels = {b.label for b in result.buckets}
    assert labels == {"Technology", "Healthcare"}
    total_weight = sum(b.weight.value for b in result.buckets)
    assert total_weight == pytest.approx(1.0)


async def test_get_sector_exposure_raises_for_empty_holdings() -> None:
    ctx = build_tool_context(holdings=[]).for_call(tool_name="get_sector_exposure", inputs_hash="h")

    with pytest.raises(ToolValidationError):
        await get_sector_exposure(GetSectorExposureInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx)


async def test_get_factor_exposure_reports_all_six_ff5_mom_factors() -> None:
    ctx = build_tool_context(
        holdings=[build_holding(ticker="AAA", quantity=10.0, cost_basis=100.0)]
    ).for_call(tool_name="get_factor_exposure", inputs_hash="h")

    result = await get_factor_exposure(
        GetFactorExposureInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    assert {loading.factor for loading in result.loadings} == {
        "mkt_rf",
        "smb",
        "hml",
        "rmw",
        "cma",
        "mom",
    }
    assert result.r_squared.unit == "ratio"


async def test_get_correlation_matrix_diagonal_is_one() -> None:
    ctx = build_tool_context().for_call(tool_name="get_correlation_matrix", inputs_hash="h")

    result = await get_correlation_matrix(GetCorrelationMatrixInput(tickers=["AAA", "BBB"]), ctx)

    row_by_ticker = {row.ticker: row for row in result.rows}
    assert row_by_ticker["AAA"].correlations["AAA"] == pytest.approx(1.0)
    assert row_by_ticker["BBB"].correlations["BBB"] == pytest.approx(1.0)


async def test_get_concentration_metrics_hhi_and_top_holdings() -> None:
    ctx = build_tool_context(
        holdings=[
            build_holding(ticker="AAA", quantity=90.0, cost_basis=1.0),
            build_holding(ticker="BBB", quantity=10.0, cost_basis=1.0),
        ]
    ).for_call(tool_name="get_concentration_metrics", inputs_hash="h")

    result = await get_concentration_metrics(
        GetConcentrationMetricsInput(portfolio_id=DEFAULT_PORTFOLIO_ID, top_n=1), ctx
    )

    assert 0.0 < result.hhi.value <= 1.0
    assert len(result.top_holdings) == 1


async def test_get_concentration_metrics_raises_for_empty_holdings() -> None:
    ctx = build_tool_context(holdings=[]).for_call(
        tool_name="get_concentration_metrics", inputs_hash="h"
    )

    with pytest.raises(ToolValidationError):
        await get_concentration_metrics(
            GetConcentrationMetricsInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
        )
