from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.tools import (
    CalculateComponentVarInput,
    CalculateComponentVarOutput,
    CalculateCvarInput,
    CalculateMaxDrawdownInput,
    CalculateMaxDrawdownOutput,
    CalculatePortfolioVarInput,
    CalculateTrackingErrorInput,
    ComponentVarEntry,
    GetPortfolioBetaInput,
    GetPortfolioBetaOutput,
)
from quantagent.data.providers.prices import PricePanel
from quantagent.quant import beta as beta_mod
from quantagent.quant import calendar as calendar_mod
from quantagent.quant import component_var as component_var_mod
from quantagent.quant import cvar as cvar_mod
from quantagent.quant import drawdown as drawdown_mod
from quantagent.quant import returns as returns_mod
from quantagent.quant import tracking_error as tracking_error_mod
from quantagent.quant import var as var_mod
from quantagent.tools._shared import fetch_priced_weights
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry


async def _load_asset_returns(
    ctx: ToolContext, portfolio_id: str, lookback_days: int
) -> tuple[pd.Series, pd.DataFrame, PricePanel]:
    """Shared fetch-align-return pipeline every risk tool needs. Returns
    `(weights, asset_returns, panel)`.
    """
    holdings = await ctx.portfolios.get_holdings(portfolio_id, tenant_id=ctx.tenant_id)
    weights, _ = await fetch_priced_weights(holdings, ctx.prices)
    end = date.today()
    start = end - timedelta(days=lookback_days)
    panel = await ctx.prices.get_prices(list(weights.index), start, end)
    aligned, _ = calendar_mod.align_calendars(panel.prices)
    asset_returns = returns_mod.simple_returns(aligned)
    return weights, asset_returns, panel


async def _load_benchmark_returns(
    ctx: ToolContext, portfolio_id: str, benchmark_ticker: str | None, panel: PricePanel
) -> tuple[str, pd.Series, PricePanel]:
    """Resolve the benchmark ticker (explicit override or the portfolio's
    own) and fetch its aligned return series over the same window as `panel`.
    """
    portfolio = await ctx.portfolios.get_portfolio(portfolio_id, tenant_id=ctx.tenant_id)
    resolved = benchmark_ticker or (portfolio.benchmark_ticker if portfolio else "SPY")
    bench_panel = await ctx.prices.get_prices([resolved], panel.prices.index[0].date(), panel.as_of)
    bench_aligned, _ = calendar_mod.align_calendars(bench_panel.prices)
    return resolved, returns_mod.simple_returns(bench_aligned)[resolved], bench_panel


@registry.tool(
    name="calculate_portfolio_var",
    description=(
        "Value-at-Risk of an entire portfolio at a given confidence level. Use for "
        "portfolio-level downside risk. Do NOT use for single-position risk (use "
        "calculate_component_var) or for realised losses (use calculate_max_drawdown)."
    ),
    p95_latency_ms=250,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def calculate_portfolio_var(inp: CalculatePortfolioVarInput, ctx: ToolContext) -> MetricValue:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )
    result = var_mod.portfolio_var(
        weights, asset_returns, inp.alpha, horizon_days=inp.horizon_days, method=inp.method
    )
    return ctx.wrap_metric(
        f"portfolio_var_{int(inp.alpha * 100)}_{inp.horizon_days}d",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        window=f"{inp.lookback_days}d",
        sample_size=result.sample_size,
        seed=result.seed,
        warnings=result.warnings,
        data_sources=[panel.source],
        estimator=result.method,
    )


@registry.tool(
    name="calculate_cvar",
    description=(
        "Expected Shortfall (CVaR) at a given confidence level: the average loss beyond "
        "the VaR threshold. Use when VaR alone understates tail severity. Do NOT use for "
        "the VaR threshold itself (use calculate_portfolio_var)."
    ),
    p95_latency_ms=250,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def calculate_cvar(inp: CalculateCvarInput, ctx: ToolContext) -> MetricValue:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )
    result = cvar_mod.portfolio_cvar(weights, asset_returns, inp.alpha)
    return ctx.wrap_metric(
        f"portfolio_cvar_{int(inp.alpha * 100)}",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        window=f"{inp.lookback_days}d",
        sample_size=result.sample_size,
        data_sources=[panel.source],
        estimator=result.method,
    )


@registry.tool(
    name="calculate_component_var",
    description=(
        "Per-position contribution to portfolio VaR. Use to answer 'which holdings are "
        "actually driving my tail risk' -- a small weight can dominate risk contribution. "
        "Do NOT use for portfolio-level VaR alone (use calculate_portfolio_var) or for "
        "static weight concentration (use get_concentration_metrics)."
    ),
    p95_latency_ms=350,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def calculate_component_var(
    inp: CalculateComponentVarInput, ctx: ToolContext
) -> CalculateComponentVarOutput:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )
    fn = (
        component_var_mod.parametric_component_var
        if inp.method == "parametric"
        else component_var_mod.historical_component_var
    )
    result = fn(weights, asset_returns, inp.alpha)

    portfolio_var_metric = ctx.wrap_metric(
        f"portfolio_var_{int(inp.alpha * 100)}",
        result.portfolio_value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        sample_size=result.sample_size,
        data_sources=[panel.source],
    )
    entries = [
        ComponentVarEntry(
            ticker=ticker,
            contribution=ctx.wrap_metric(
                f"component_var_{ticker}",
                value,
                "ratio",
                result.method,
                as_of=panel.as_of,
                sample_size=result.sample_size,
                data_sources=[panel.source],
            ),
            share_of_portfolio_var=(
                value / result.portfolio_value if result.portfolio_value else 0.0
            ),
        )
        for ticker, value in result.components.items()
    ]
    return CalculateComponentVarOutput(
        portfolio_id=inp.portfolio_id,
        alpha=inp.alpha,
        method=inp.method,
        portfolio_var=portfolio_var_metric,
        components=entries,
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=[panel.source]),
    )


@registry.tool(
    name="calculate_max_drawdown",
    description=(
        "Maximum peak-to-trough decline on the portfolio's current-weight equity curve, "
        "with peak/trough/recovery dates. Use for realised historical loss severity. Do "
        "NOT use for forward-looking risk (use calculate_portfolio_var/calculate_cvar)."
    ),
    p95_latency_ms=300,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def calculate_max_drawdown(
    inp: CalculateMaxDrawdownInput, ctx: ToolContext
) -> CalculateMaxDrawdownOutput:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )
    result = drawdown_mod.max_drawdown(weights, asset_returns)
    return CalculateMaxDrawdownOutput(
        portfolio_id=inp.portfolio_id,
        drawdown=ctx.wrap_metric(
            "max_drawdown",
            result.value,
            "ratio",
            result.method,
            as_of=panel.as_of,
            window=f"{inp.lookback_days}d",
            sample_size=result.sample_size,
            data_sources=[panel.source],
        ),
        peak_date=result.peak_date,
        trough_date=result.trough_date,
        recovery_date=result.recovery_date,
        recovery_duration_days=result.recovery_duration_days,
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=[panel.source]),
    )


@registry.tool(
    name="get_portfolio_beta",
    description=(
        "Portfolio beta and downside beta to its benchmark. Use for market sensitivity "
        "questions. Do NOT use for factor-model decomposition (use get_factor_exposure) "
        "or benchmark-relative volatility (use calculate_tracking_error)."
    ),
    p95_latency_ms=300,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def get_portfolio_beta(
    inp: GetPortfolioBetaInput, ctx: ToolContext
) -> GetPortfolioBetaOutput:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )
    benchmark_ticker, market_returns, bench_panel = await _load_benchmark_returns(
        ctx, inp.portfolio_id, inp.benchmark_ticker, panel
    )
    common_index = asset_returns.index.intersection(market_returns.index)
    r = asset_returns.loc[common_index]
    m = market_returns.loc[common_index]

    beta_result = beta_mod.beta(weights, r, m)
    downside_result = beta_mod.downside_beta(weights, r, m)
    sources = [panel.source, bench_panel.source]
    return GetPortfolioBetaOutput(
        portfolio_id=inp.portfolio_id,
        benchmark_ticker=benchmark_ticker,
        beta=ctx.wrap_metric(
            "beta",
            beta_result.value,
            "ratio",
            beta_result.method,
            as_of=panel.as_of,
            sample_size=beta_result.sample_size,
            data_sources=sources,
        ),
        downside_beta=ctx.wrap_metric(
            "downside_beta",
            downside_result.value,
            "ratio",
            downside_result.method,
            as_of=panel.as_of,
            sample_size=downside_result.sample_size,
            data_sources=sources,
        ),
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=sources),
    )


@registry.tool(
    name="calculate_tracking_error",
    description=(
        "Annualised standard deviation of the portfolio's return minus its benchmark's "
        "return. Use for 'how much am I deviating from my benchmark' questions, e.g. for "
        "a mandate that caps active risk. Do NOT use for directional risk (use "
        "get_portfolio_beta) or absolute downside (use calculate_portfolio_var)."
    ),
    p95_latency_ms=300,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def calculate_tracking_error(
    inp: CalculateTrackingErrorInput, ctx: ToolContext
) -> MetricValue:
    weights, asset_returns, panel = await _load_asset_returns(
        ctx, inp.portfolio_id, inp.lookback_days
    )
    _, market_returns, bench_panel = await _load_benchmark_returns(
        ctx, inp.portfolio_id, inp.benchmark_ticker, panel
    )
    common_index = asset_returns.index.intersection(market_returns.index)
    result = tracking_error_mod.tracking_error(
        weights, asset_returns.loc[common_index], market_returns.loc[common_index]
    )
    return ctx.wrap_metric(
        "tracking_error",
        result.value,
        "ratio",
        result.method,
        as_of=panel.as_of,
        window=f"{inp.lookback_days}d",
        sample_size=result.sample_size,
        data_sources=[panel.source, bench_panel.source],
    )
