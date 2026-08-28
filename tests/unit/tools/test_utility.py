from __future__ import annotations

from quantagent.contracts.tools import GenerateRiskReportInput
from quantagent.tools.utility import generate_risk_report
from tests.unit.tools.builders import DEFAULT_PORTFOLIO_ID, build_holding, build_tool_context

_TWO_HOLDINGS = [
    build_holding(ticker="AAA", quantity=10.0, cost_basis=100.0),
    build_holding(ticker="BBB", quantity=5.0, cost_basis=200.0),
]


async def test_generate_risk_report_includes_every_default_section() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="generate_risk_report", inputs_hash="h"
    )

    result = await generate_risk_report(
        GenerateRiskReportInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    section_ids = {s.section_id for s in result.sections}
    assert section_ids == {
        "var",
        "cvar",
        "component_var",
        "drawdown",
        "beta",
        "tracking_error",
        "concentration",
    }


async def test_generate_risk_report_honors_a_restricted_section_list() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="generate_risk_report", inputs_hash="h"
    )

    result = await generate_risk_report(
        GenerateRiskReportInput(portfolio_id=DEFAULT_PORTFOLIO_ID, sections=["var", "beta"]), ctx
    )

    assert [s.section_id for s in result.sections] == ["var", "beta"]


async def test_generate_risk_report_every_section_has_at_least_one_metric() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="generate_risk_report", inputs_hash="h"
    )

    result = await generate_risk_report(
        GenerateRiskReportInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    for section in result.sections:
        assert len(section.metrics) > 0


async def test_generate_risk_report_skips_benchmark_fetch_when_not_requested() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="generate_risk_report", inputs_hash="h"
    )

    result = await generate_risk_report(
        GenerateRiskReportInput(portfolio_id=DEFAULT_PORTFOLIO_ID, sections=["var"]), ctx
    )

    assert result.sections[0].section_id == "var"


async def test_generate_risk_report_carries_top_level_provenance() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="generate_risk_report", inputs_hash="h"
    )

    result = await generate_risk_report(
        GenerateRiskReportInput(portfolio_id=DEFAULT_PORTFOLIO_ID), ctx
    )

    assert result.provenance.tool_name == "generate_risk_report"
    assert result.provenance.data_sources == ["fake"]


async def test_generate_risk_report_var_and_cvar_are_internally_consistent() -> None:
    ctx = build_tool_context(holdings=_TWO_HOLDINGS).for_call(
        tool_name="generate_risk_report", inputs_hash="h"
    )

    result = await generate_risk_report(
        GenerateRiskReportInput(portfolio_id=DEFAULT_PORTFOLIO_ID, sections=["var", "cvar"]), ctx
    )

    var_value = next(s for s in result.sections if s.section_id == "var").metrics[0].value
    cvar_value = next(s for s in result.sections if s.section_id == "cvar").metrics[0].value
    assert cvar_value >= var_value
