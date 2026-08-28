from __future__ import annotations

from datetime import date, timedelta
from typing import cast

import pandas as pd

from quantagent.contracts.tools import (
    FundamentalsOutput,
    GetFundamentalsInput,
    GetPricesInput,
    GetPricesOutput,
    GetReturnsInput,
    GetReturnsOutput,
    PriceObservation,
    ReturnObservation,
)
from quantagent.quant import calendar as calendar_mod
from quantagent.quant import returns as returns_mod
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry


@registry.tool(
    name="get_prices",
    description=(
        "Split/dividend-adjusted daily close prices for one or more tickers over a date "
        "range, calendar-aligned. Use as the raw input to any custom price-based analysis "
        "not already covered by a risk/exposure tool. Do NOT use for returns (use "
        "get_returns) -- it returns levels, not returns."
    ),
    p95_latency_ms=500,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def get_prices(inp: GetPricesInput, ctx: ToolContext) -> GetPricesOutput:
    panel = await ctx.prices.get_prices(inp.tickers, inp.start, inp.end, adjusted=inp.adjusted)
    observations = [
        PriceObservation(
            as_of=cast(pd.Timestamp, idx).date(),
            prices={str(k): float(v) for k, v in row.dropna().items()},
        )
        for idx, row in panel.prices.iterrows()
    ]
    return GetPricesOutput(
        tickers=panel.tickers,
        unresolved_tickers=panel.unresolved_tickers,
        observations=observations,
        source=panel.source,
        provenance=ctx.build_provenance(
            as_of=panel.as_of, data_sources=[panel.source], warnings=panel.warnings
        ),
    )


@registry.tool(
    name="get_returns",
    description=(
        "Daily simple or log returns for one or more tickers over a trailing lookback "
        "window, calendar-aligned (gaps forward-filled up to 5 days, else dropped). Use "
        "simple returns for anything portfolio-aggregated; log only for explicit time-"
        "aggregation. Do NOT use for prices (use get_prices) or for a portfolio-weighted "
        "series (compute that downstream from get_holdings' weights)."
    ),
    p95_latency_ms=500,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def get_returns(inp: GetReturnsInput, ctx: ToolContext) -> GetReturnsOutput:
    end = date.today()
    start = end - timedelta(days=inp.lookback_days)
    panel = await ctx.prices.get_prices(inp.tickers, start, end)
    aligned, warnings = calendar_mod.align_calendars(panel.prices)
    matrix = (
        returns_mod.simple_returns(aligned)
        if inp.kind == "simple"
        else returns_mod.log_returns(aligned)
    )

    observations = [
        ReturnObservation(
            as_of=cast(pd.Timestamp, idx).date(), returns={str(k): float(v) for k, v in row.items()}
        )
        for idx, row in matrix.iterrows()
    ]
    return GetReturnsOutput(
        tickers=panel.tickers,
        kind=inp.kind,
        n_obs=len(matrix),
        observations=observations,
        warnings=[*panel.warnings, *warnings],
        provenance=ctx.build_provenance(
            as_of=panel.as_of, data_sources=[panel.source], sample_size=len(matrix)
        ),
    )


@registry.tool(
    name="get_fundamentals",
    description=(
        "Latest fundamentals for one ticker: sector, industry, TTM revenue, net margin, "
        "P/E. Use for sector classification or valuation context. Do NOT use for time-series "
        "price/return data (use get_prices/get_returns)."
    ),
    p95_latency_ms=400,
    est_cost_usd=0.0,
    cache_ttl_s=3600,
    side_effects="READ_ONLY",
)
async def get_fundamentals(inp: GetFundamentalsInput, ctx: ToolContext) -> FundamentalsOutput:
    fundamentals = await ctx.fundamentals.get_fundamentals(inp.ticker)
    sources = [fundamentals.source]
    revenue = (
        ctx.wrap_metric(
            f"revenue_ttm_{inp.ticker}",
            fundamentals.revenue_ttm,
            "usd",
            "provider_reported",
            as_of=fundamentals.as_of,
            data_sources=sources,
        )
        if fundamentals.revenue_ttm is not None
        else None
    )
    margin = (
        ctx.wrap_metric(
            f"net_margin_{inp.ticker}",
            fundamentals.net_margin,
            "ratio",
            "provider_reported",
            as_of=fundamentals.as_of,
            data_sources=sources,
        )
        if fundamentals.net_margin is not None
        else None
    )
    pe = (
        ctx.wrap_metric(
            f"pe_ratio_{inp.ticker}",
            fundamentals.pe_ratio,
            "ratio",
            "provider_reported",
            as_of=fundamentals.as_of,
            data_sources=sources,
        )
        if fundamentals.pe_ratio is not None
        else None
    )
    return FundamentalsOutput(
        ticker=fundamentals.ticker,
        as_of=fundamentals.as_of,
        sector=fundamentals.sector,
        industry=fundamentals.industry,
        revenue_ttm=revenue,
        net_margin=margin,
        pe_ratio=pe,
        source=fundamentals.source,
        provenance=ctx.build_provenance(as_of=fundamentals.as_of, data_sources=sources),
    )
