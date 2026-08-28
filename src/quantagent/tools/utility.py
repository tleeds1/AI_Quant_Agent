from __future__ import annotations

from datetime import datetime

import pandas as pd

from quantagent.contracts.tools import (
    GenerateRiskReportInput,
    ReportArtifact,
    ReportSection,
    ReportSectionId,
)
from quantagent.data.providers.prices import PricePanel
from quantagent.quant import beta as beta_mod
from quantagent.quant import component_var as component_var_mod
from quantagent.quant import concentration as concentration_mod
from quantagent.quant import cvar as cvar_mod
from quantagent.quant import drawdown as drawdown_mod
from quantagent.quant import tracking_error as tracking_error_mod
from quantagent.quant import var as var_mod
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry
from quantagent.tools.risk import _load_asset_returns, _load_benchmark_returns

_BENCHMARK_SECTIONS = {"beta", "tracking_error"}


def _var_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    ctx: ToolContext,
    panel: PricePanel,
    inp: GenerateRiskReportInput,
) -> ReportSection:
    result = var_mod.portfolio_var(
        weights, asset_returns, inp.alpha, horizon_days=inp.horizon_days, method=inp.method
    )
    metric = ctx.wrap_metric(
        f"portfolio_var_{int(inp.alpha * 100)}_{inp.horizon_days}d",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        sample_size=result.sample_size,
        seed=result.seed,
        warnings=result.warnings,
        data_sources=[panel.source],
    )
    return ReportSection(section_id="var", metrics=[metric])


def _cvar_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    ctx: ToolContext,
    panel: PricePanel,
    inp: GenerateRiskReportInput,
) -> ReportSection:
    result = cvar_mod.portfolio_cvar(weights, asset_returns, inp.alpha)
    metric = ctx.wrap_metric(
        f"portfolio_cvar_{int(inp.alpha * 100)}",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        sample_size=result.sample_size,
        data_sources=[panel.source],
    )
    return ReportSection(section_id="cvar", metrics=[metric])


def _component_var_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    ctx: ToolContext,
    panel: PricePanel,
    inp: GenerateRiskReportInput,
) -> ReportSection:
    fn = (
        component_var_mod.parametric_component_var
        if inp.method == "parametric"
        else component_var_mod.historical_component_var
    )
    result = fn(weights, asset_returns, inp.alpha)
    metrics = [
        ctx.wrap_metric(
            f"component_var_{ticker}",
            value,
            "ratio",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=[panel.source],
        )
        for ticker, value in result.components.items()
    ]
    return ReportSection(section_id="component_var", metrics=metrics)


def _drawdown_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    ctx: ToolContext,
    panel: PricePanel,
    inp: GenerateRiskReportInput,
) -> ReportSection:
    result = drawdown_mod.max_drawdown(weights, asset_returns)
    metric = ctx.wrap_metric(
        "max_drawdown",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        sample_size=result.sample_size,
        data_sources=[panel.source],
    )
    return ReportSection(section_id="drawdown", metrics=[metric])


def _concentration_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    ctx: ToolContext,
    panel: PricePanel,
    inp: GenerateRiskReportInput,
) -> ReportSection:
    result = concentration_mod.portfolio_concentration(weights)
    metrics = [
        ctx.wrap_metric(
            "hhi",
            result.hhi,
            "ratio",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=[panel.source],
        ),
        ctx.wrap_metric(
            "effective_holdings",
            result.effective_holdings,
            "count",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=[panel.source],
        ),
    ]
    return ReportSection(section_id="concentration", metrics=metrics)


def _beta_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    ctx: ToolContext,
    panel: PricePanel,
) -> ReportSection:
    common = asset_returns.index.intersection(benchmark_returns.index)
    result = beta_mod.beta(weights, asset_returns.loc[common], benchmark_returns.loc[common])
    metric = ctx.wrap_metric(
        "beta",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        sample_size=result.sample_size,
        data_sources=[panel.source],
    )
    return ReportSection(section_id="beta", metrics=[metric])


def _tracking_error_section(
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    ctx: ToolContext,
    panel: PricePanel,
) -> ReportSection:
    common = asset_returns.index.intersection(benchmark_returns.index)
    result = tracking_error_mod.tracking_error(
        weights, asset_returns.loc[common], benchmark_returns.loc[common]
    )
    metric = ctx.wrap_metric(
        "tracking_error",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        sample_size=result.sample_size,
        data_sources=[panel.source],
    )
    return ReportSection(section_id="tracking_error", metrics=[metric])


def _build_section(
    section_id: ReportSectionId,
    weights: pd.Series,
    asset_returns: pd.DataFrame,
    benchmark_returns: pd.Series | None,
    ctx: ToolContext,
    panel: PricePanel,
    inp: GenerateRiskReportInput,
) -> ReportSection:
    if section_id == "beta":
        assert benchmark_returns is not None  # guaranteed by the pre-fetch below
        return _beta_section(weights, asset_returns, benchmark_returns, ctx, panel)
    if section_id == "tracking_error":
        assert benchmark_returns is not None
        return _tracking_error_section(weights, asset_returns, benchmark_returns, ctx, panel)
    section_builders = {
        "var": _var_section,
        "cvar": _cvar_section,
        "component_var": _component_var_section,
        "drawdown": _drawdown_section,
        "concentration": _concentration_section,
    }
    return section_builders[section_id](weights, asset_returns, ctx, panel, inp)


@registry.tool(
    name="generate_risk_report",
    description=(
        "Assembles a full deterministic risk report (VaR, CVaR, component VaR, drawdown, "
        "beta, tracking error, concentration) for a portfolio in one call. Use when the "
        "user wants a broad risk overview rather than one specific metric -- saves N "
        "separate tool calls. Do NOT use when only one metric is needed (call that tool "
        "directly; this one is slower and does more work than necessary)."
    ),
    p95_latency_ms=1500,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def generate_risk_report(inp: GenerateRiskReportInput, ctx: ToolContext) -> ReportArtifact:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )

    benchmark_returns: pd.Series | None = None
    if _BENCHMARK_SECTIONS & set(inp.sections):
        _, benchmark_returns, _ = await _load_benchmark_returns(ctx, inp.portfolio_id, None, panel)

    sections = [
        _build_section(section_id, weights, asset_returns, benchmark_returns, ctx, panel, inp)
        for section_id in inp.sections
    ]
    return ReportArtifact(
        portfolio_id=inp.portfolio_id,
        as_of=panel.as_of,
        generated_at=datetime.now(),
        sections=sections,
        warnings=list(panel.warnings),
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=[panel.source]),
    )
