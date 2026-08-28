from __future__ import annotations

from datetime import date, datetime

from scripts.print_risk_report import render_report

from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.provenance import Provenance
from quantagent.contracts.tools import (
    GetHoldingsOutput,
    HoldingRecord,
    PortfolioOutput,
    ReportArtifact,
    ReportSection,
)


def _build_provenance(**overrides: object) -> Provenance:
    defaults: dict[str, object] = {
        "tool_call_id": "tc_1",
        "tool_name": "print_risk_report",
        "as_of": date(2026, 8, 22),
        "computed_at": datetime(2026, 8, 22, 21, 0, 0),
        "inputs_hash": "a" * 64,
        "data_sources": ["yfinance"],
    }
    defaults.update(overrides)
    return Provenance(**defaults)  # type: ignore[arg-type]


def _build_metric(**overrides: object) -> MetricValue:
    defaults: dict[str, object] = {
        "metric_id": "portfolio_var_95_1d",
        "value": 0.0243,
        "unit": "ratio",
        "method": "historical",
        "provenance": _build_provenance(),
    }
    defaults.update(overrides)
    return MetricValue(**defaults)  # type: ignore[arg-type]


def _build_portfolio(**overrides: object) -> PortfolioOutput:
    defaults: dict[str, object] = {
        "portfolio_id": "pf_demo",
        "name": "Demo Growth Portfolio",
        "base_currency": "USD",
        "benchmark_ticker": "SPY",
        "mandate_constraints": {},
        "provenance": _build_provenance(),
    }
    defaults.update(overrides)
    return PortfolioOutput(**defaults)  # type: ignore[arg-type]


def _build_holdings(**overrides: object) -> GetHoldingsOutput:
    holding_kwargs: dict[str, object] = {
        "ticker": "AAPL",
        "quantity": 50.0,
        "cost_basis_usd": 150.0,
        "market_value": _build_metric(metric_id="market_value_AAPL", value=7500.0, unit="usd"),
        "weight": _build_metric(metric_id="weight_AAPL", value=0.6),
        "as_of": date(2026, 8, 22),
    }
    other_kwargs = dict(holding_kwargs, ticker="MSFT", weight=_build_metric(value=0.4))
    defaults: dict[str, object] = {
        "portfolio_id": "pf_demo",
        "as_of": date(2026, 8, 22),
        "holdings": [HoldingRecord(**holding_kwargs), HoldingRecord(**other_kwargs)],  # type: ignore[arg-type]
        "provenance": _build_provenance(),
    }
    defaults.update(overrides)
    return GetHoldingsOutput(**defaults)  # type: ignore[arg-type]


def _build_report(**overrides: object) -> ReportArtifact:
    defaults: dict[str, object] = {
        "portfolio_id": "pf_demo",
        "as_of": date(2026, 8, 22),
        "generated_at": datetime(2026, 8, 22, 21, 0, 0),
        "sections": [ReportSection(section_id="var", metrics=[_build_metric()])],
        "warnings": [],
        "provenance": _build_provenance(),
    }
    defaults.update(overrides)
    return ReportArtifact(**defaults)  # type: ignore[arg-type]


def test_render_report_includes_portfolio_name_and_id() -> None:
    output = render_report(_build_portfolio(), _build_holdings(), _build_report())

    assert "Demo Growth Portfolio" in output
    assert "pf_demo" in output


def test_render_report_lists_holdings_with_weights() -> None:
    output = render_report(_build_portfolio(), _build_holdings(), _build_report())

    assert "AAPL" in output
    assert "60.00%" in output


def test_render_report_includes_metric_values() -> None:
    output = render_report(_build_portfolio(), _build_holdings(), _build_report())

    assert "portfolio_var_95_1d" in output
    assert "0.0243" in output


def test_render_report_includes_warnings_section_when_present() -> None:
    report = _build_report(warnings=["unresolved ticker: NOPE"])

    output = render_report(_build_portfolio(), _build_holdings(), report)

    assert "Warnings:" in output
    assert "NOPE" in output


def test_render_report_omits_warnings_section_when_absent() -> None:
    output = render_report(_build_portfolio(), _build_holdings(), _build_report())

    assert "Warnings:" not in output
