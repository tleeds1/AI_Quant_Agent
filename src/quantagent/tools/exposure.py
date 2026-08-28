from __future__ import annotations

from datetime import date, timedelta

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import (
    CorrelationRow,
    ExposureBucket,
    FactorLoading,
    GetConcentrationMetricsInput,
    GetConcentrationMetricsOutput,
    GetCorrelationMatrixInput,
    GetCorrelationMatrixOutput,
    GetFactorExposureInput,
    GetFactorExposureOutput,
    GetSectorExposureInput,
    GetSectorExposureOutput,
    TopHolding,
)
from quantagent.quant import calendar as calendar_mod
from quantagent.quant import concentration as concentration_mod
from quantagent.quant import covariance as covariance_mod
from quantagent.quant import factors as factors_mod
from quantagent.quant import returns as returns_mod
from quantagent.tools._shared import fetch_priced_weights
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry


@registry.tool(
    name="get_sector_exposure",
    description=(
        "Portfolio weight grouped by GICS sector, using per-ticker fundamentals lookups. "
        "Use for 'how concentrated am I in tech/healthcare/etc.' questions. Do NOT use for "
        "an AI/thematic breakdown (that needs get_theme_exposure, not available until M3) "
        "or for tail-risk contribution by sector (use calculate_component_var)."
    ),
    p95_latency_ms=1200,  # concurrency-capped fundamentals batch, one lookup per holding
    est_cost_usd=0.0,
    cache_ttl_s=1800,
    side_effects="READ_ONLY",
)
async def get_sector_exposure(
    inp: GetSectorExposureInput, ctx: ToolContext
) -> GetSectorExposureOutput:
    holdings = await ctx.portfolios.get_holdings(inp.portfolio_id, tenant_id=ctx.tenant_id)
    if not holdings:
        raise ToolValidationError(f"no holdings for portfolio {inp.portfolio_id!r}")
    weights, panel = await fetch_priced_weights(holdings, ctx.prices)
    fundamentals_panel = await ctx.fundamentals.get_fundamentals_batch([h.ticker for h in holdings])

    buckets: dict[str, list[str]] = {}
    for ticker in weights.index:
        fundamentals = fundamentals_panel.fundamentals.get(ticker)
        label = fundamentals.sector if fundamentals and fundamentals.sector else "UNKNOWN"
        buckets.setdefault(label, []).append(ticker)

    exposure_buckets = [
        ExposureBucket(
            label=label,
            weight=ctx.wrap_metric(
                f"sector_weight_{label}",
                float(weights.loc[members].sum()),
                "ratio",
                "market_value_share_by_sector",
                as_of=panel.as_of,
                data_sources=[panel.source, fundamentals_panel.source],
            ),
            tickers=members,
        )
        for label, members in buckets.items()
    ]
    return GetSectorExposureOutput(
        portfolio_id=inp.portfolio_id,
        scheme=inp.scheme,
        buckets=exposure_buckets,
        unresolved_tickers=fundamentals_panel.unresolved_tickers,
        provenance=ctx.build_provenance(
            as_of=panel.as_of,
            data_sources=[panel.source, fundamentals_panel.source],
            warnings=fundamentals_panel.warnings,
        ),
    )


@registry.tool(
    name="get_factor_exposure",
    description=(
        "Fama-French 5 + Momentum factor betas for the portfolio, with HAC (Newey-West) "
        "t-stats, R-squared, and idiosyncratic variance share. Use for 'what's driving my "
        "portfolio's returns' or systematic-risk decomposition. Do NOT use for a single "
        "stock's beta to its benchmark (use get_portfolio_beta) or for correlation between "
        "holdings (use get_correlation_matrix)."
    ),
    p95_latency_ms=800,
    est_cost_usd=0.0,
    cache_ttl_s=1800,
    side_effects="READ_ONLY",
)
async def get_factor_exposure(
    inp: GetFactorExposureInput, ctx: ToolContext
) -> GetFactorExposureOutput:
    holdings = await ctx.portfolios.get_holdings(inp.portfolio_id, tenant_id=ctx.tenant_id)
    if not holdings:
        raise ToolValidationError(f"no holdings for portfolio {inp.portfolio_id!r}")
    end = date.today()
    start = end - timedelta(days=inp.lookback_days)

    weights, _ = await fetch_priced_weights(holdings, ctx.prices)
    panel = await ctx.prices.get_prices(list(weights.index), start, end)
    aligned_prices, _ = calendar_mod.align_calendars(panel.prices)
    asset_returns = returns_mod.simple_returns(aligned_prices)
    r_p = returns_mod.portfolio_returns(weights, asset_returns)

    factor_panel = await ctx.factors.get_factor_returns(start, end)
    common_index = r_p.index.intersection(factor_panel.returns.index)
    excess_r_p = r_p.loc[common_index] - factor_panel.risk_free.loc[common_index]
    factor_returns = factor_panel.returns.loc[common_index]

    result = factors_mod.factor_exposure(excess_r_p, factor_returns)
    sources = [panel.source, factor_panel.source]
    loadings = [
        FactorLoading(
            factor=name,
            beta=ctx.wrap_metric(
                f"factor_beta_{name}",
                result.betas[name],
                "ratio",
                result.method,
                as_of=panel.as_of,
                data_sources=sources,
                sample_size=result.sample_size,
                estimator=result.method,
            ),
            t_stat=result.t_stats[name],
            significant=result.significant[name],
        )
        for name in result.betas
    ]
    return GetFactorExposureOutput(
        portfolio_id=inp.portfolio_id,
        model=inp.model,
        window=f"{inp.lookback_days}d",
        loadings=loadings,
        r_squared=ctx.wrap_metric(
            "factor_r_squared",
            result.r_squared,
            "ratio",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=sources,
        ),
        idiosyncratic_variance_share=ctx.wrap_metric(
            "idiosyncratic_variance_share",
            result.idiosyncratic_variance_share,
            "ratio",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=sources,
        ),
        hac_lags=result.hac_lags,
        provenance=ctx.build_provenance(
            as_of=panel.as_of, data_sources=sources, warnings=factor_panel.warnings
        ),
    )


@registry.tool(
    name="get_correlation_matrix",
    description=(
        "Ledoit-Wolf shrunk correlation matrix across a set of tickers. Use to check "
        "diversification or find hidden co-movement between holdings. Do NOT use for a "
        "single number summarizing portfolio risk (use calculate_portfolio_var) or for "
        "concentration by weight (use get_concentration_metrics)."
    ),
    p95_latency_ms=600,
    est_cost_usd=0.0,
    cache_ttl_s=1800,
    side_effects="READ_ONLY",
)
async def get_correlation_matrix(
    inp: GetCorrelationMatrixInput, ctx: ToolContext
) -> GetCorrelationMatrixOutput:
    end = date.today()
    start = end - timedelta(days=inp.lookback_days)
    panel = await ctx.prices.get_prices(inp.tickers, start, end)
    aligned, _ = calendar_mod.align_calendars(panel.prices)
    asset_returns = returns_mod.simple_returns(aligned)

    cov = covariance_mod.ledoit_wolf_covariance(asset_returns)
    corr = covariance_mod.correlation_from_covariance(cov)
    rows = [
        CorrelationRow(
            ticker=str(ticker),
            correlations={str(k): float(v) for k, v in corr.loc[ticker].items()},
        )
        for ticker in corr.index
    ]
    return GetCorrelationMatrixOutput(
        tickers=panel.tickers,
        method=cov.method,
        sample_size=cov.sample_size,
        rows=rows,
        provenance=ctx.build_provenance(
            as_of=panel.as_of,
            data_sources=[panel.source],
            estimator=cov.method,
            sample_size=cov.sample_size,
        ),
    )


@registry.tool(
    name="get_concentration_metrics",
    description=(
        "HHI, effective number of holdings, and top-N weight for a portfolio. Use for "
        "'how diversified am I' questions -- HHI turns 'I have 30 stocks' into 'you "
        "effectively have 6'. Do NOT use for tail-risk concentration (use "
        "calculate_component_var, which answers 'which positions drive my VaR')."
    ),
    p95_latency_ms=400,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def get_concentration_metrics(
    inp: GetConcentrationMetricsInput, ctx: ToolContext
) -> GetConcentrationMetricsOutput:
    holdings = await ctx.portfolios.get_holdings(inp.portfolio_id, tenant_id=ctx.tenant_id)
    if not holdings:
        raise ToolValidationError(f"no holdings for portfolio {inp.portfolio_id!r}")
    weights, panel = await fetch_priced_weights(holdings, ctx.prices)

    result = concentration_mod.portfolio_concentration(weights, top_n=inp.top_n)
    top_holdings = [
        TopHolding(ticker=str(t), weight=float(weights[t]))
        for t in weights.abs().nlargest(inp.top_n).index
    ]
    return GetConcentrationMetricsOutput(
        portfolio_id=inp.portfolio_id,
        hhi=ctx.wrap_metric(
            "hhi",
            result.hhi,
            "ratio",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=[panel.source],
        ),
        effective_holdings=ctx.wrap_metric(
            "effective_holdings",
            result.effective_holdings,
            "count",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=[panel.source],
        ),
        top_n_weight=ctx.wrap_metric(
            f"top_{inp.top_n}_weight",
            result.top_n_weight,
            "ratio",
            result.method,
            as_of=panel.as_of,
            sample_size=result.sample_size,
            data_sources=[panel.source],
        ),
        top_holdings=top_holdings,
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=[panel.source]),
    )
