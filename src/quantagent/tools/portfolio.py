from __future__ import annotations

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import (
    GetHoldingsInput,
    GetHoldingsOutput,
    GetPortfolioInput,
    GetTransactionsInput,
    GetTransactionsOutput,
    HoldingRecord,
    PortfolioOutput,
    TransactionRecord,
)
from quantagent.tools._shared import fetch_priced_weights
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry


@registry.tool(
    name="get_portfolio",
    description=(
        "Fetch a portfolio's metadata: name, base currency, benchmark ticker, mandate "
        "constraints. Use first in any portfolio analysis to resolve the benchmark and "
        "constraints before calling risk/exposure tools. Do NOT use for positions "
        "(use get_holdings) or trade history (use get_transactions)."
    ),
    p95_latency_ms=50,
    est_cost_usd=0.0,
    cache_ttl_s=300,
    side_effects="READ_ONLY",
)
async def get_portfolio(inp: GetPortfolioInput, ctx: ToolContext) -> PortfolioOutput:
    portfolio = await ctx.portfolios.get_portfolio(inp.portfolio_id, tenant_id=ctx.tenant_id)
    if portfolio is None:
        raise ToolValidationError(f"no portfolio {inp.portfolio_id!r} for this tenant")
    return PortfolioOutput(
        portfolio_id=portfolio.id,
        name=portfolio.name,
        base_currency=portfolio.base_currency,
        benchmark_ticker=portfolio.benchmark_ticker,
        mandate_constraints=portfolio.mandate_constraints,
        provenance=ctx.build_provenance(data_sources=["quantagent:portfolio_repository"]),
    )


@registry.tool(
    name="get_holdings",
    description=(
        "Current positions in a portfolio: ticker, quantity, cost basis, market value and "
        "portfolio weight, each priced off the latest close as of the holdings date. Use "
        "whenever weights or position-level detail are needed. Do NOT use for sector/theme "
        "aggregation (use get_sector_exposure) or historical trades (use get_transactions)."
    ),
    p95_latency_ms=400,
    est_cost_usd=0.0,
    cache_ttl_s=300,
    side_effects="READ_ONLY",
)
async def get_holdings(inp: GetHoldingsInput, ctx: ToolContext) -> GetHoldingsOutput:
    holdings = await ctx.portfolios.get_holdings(
        inp.portfolio_id, tenant_id=ctx.tenant_id, as_of=inp.as_of
    )
    if not holdings:
        raise ToolValidationError(f"no holdings for portfolio {inp.portfolio_id!r}")
    weights, panel = await fetch_priced_weights(holdings, ctx.prices)
    by_ticker = {h.ticker: h for h in holdings}
    latest = panel.prices.iloc[-1]

    records = [
        HoldingRecord(
            ticker=ticker,
            quantity=by_ticker[ticker].quantity,
            cost_basis_usd=by_ticker[ticker].cost_basis,
            market_value=ctx.wrap_metric(
                f"market_value_{ticker}",
                by_ticker[ticker].quantity * float(latest[ticker]),
                "usd",
                "quantity_times_last_close",
                as_of=panel.as_of,
                data_sources=[panel.source],
            ),
            weight=ctx.wrap_metric(
                f"weight_{ticker}",
                float(weights[ticker]),
                "ratio",
                "market_value_share",
                as_of=panel.as_of,
                data_sources=[panel.source],
            ),
            as_of=by_ticker[ticker].as_of,
        )
        for ticker in weights.index
    ]
    return GetHoldingsOutput(
        portfolio_id=inp.portfolio_id,
        as_of=holdings[0].as_of,
        holdings=records,
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=[panel.source]),
    )


@registry.tool(
    name="get_transactions",
    description=(
        "Trade history for a portfolio between two dates: side, quantity, price. Use for "
        "realised P&L or turnover questions. Do NOT use for current positions (use "
        "get_holdings)."
    ),
    p95_latency_ms=300,
    est_cost_usd=0.0,
    cache_ttl_s=300,
    side_effects="READ_ONLY",
)
async def get_transactions(inp: GetTransactionsInput, ctx: ToolContext) -> GetTransactionsOutput:
    transactions = await ctx.portfolios.get_transactions(
        inp.portfolio_id, tenant_id=ctx.tenant_id, start=inp.start, end=inp.end
    )
    records = [
        TransactionRecord(
            ticker=t.ticker,
            trade_date=t.trade_date,
            side=t.side,
            quantity=t.quantity,
            price_usd=t.price,
        )
        for t in transactions
    ]
    return GetTransactionsOutput(
        portfolio_id=inp.portfolio_id,
        start=inp.start,
        end=inp.end,
        transactions=records,
        provenance=ctx.build_provenance(data_sources=["quantagent:portfolio_repository"]),
    )
