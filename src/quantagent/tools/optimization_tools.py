# mypy: ignore-errors
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from quantagent.contracts.errors import ToolValidationError
from quantagent.contracts.tools import (
    OptimizePortfolioInput,
    OptimizePortfolioOutput,
    RebalanceTrade,
    SimulateTradeImpactInput,
    SimulateTradeImpactOutput,
)
from quantagent.quant import calendar as calendar_mod
from quantagent.quant import returns as returns_mod
from quantagent.quant.covariance import ledoit_wolf_covariance
from quantagent.quant.optimization import optimize_portfolio as quant_optimize_portfolio
from quantagent.tools._shared import fetch_priced_weights
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry


async def _load_prices_and_returns(
    ctx: ToolContext,
    tickers: list[str],
    as_of: date | None,
    lookback_days: int = 504,
) -> tuple[pd.DataFrame, date, str]:
    """Helper to fetch and align price/return history."""
    end = as_of or date.today()
    start = end - timedelta(days=lookback_days)
    panel = await ctx.prices.get_prices(tickers, start, end)
    aligned, _ = calendar_mod.align_calendars(panel.prices)
    asset_returns = returns_mod.simple_returns(aligned)
    return asset_returns, panel.as_of, panel.source


@registry.tool(
    name="optimize_portfolio",
    description=(
        "Compute optimal portfolio weights under constraints using CVXPY. Supports min_variance, "
        "max_utility (mean-variance), and risk_parity objectives. Returns target weights, "
        "ex-ante risk/return statistics, and a list of rebalance trades. Use to design rebalancing "
        "or asset reallocation. Do NOT use for static exposure analysis (use get_sector_exposure)."
    ),
    p95_latency_ms=600,
    est_cost_usd=0.0,
    cache_ttl_s=300,
    side_effects="READ_ONLY",
)
async def optimize_portfolio(
    inp: OptimizePortfolioInput, ctx: ToolContext
) -> OptimizePortfolioOutput:
    holdings = await ctx.portfolios.get_holdings(
        inp.portfolio_id, tenant_id=ctx.tenant_id, as_of=inp.as_of
    )
    if not holdings:
        raise ToolValidationError(f"No holdings found for portfolio {inp.portfolio_id}")

    weights, _ = await fetch_priced_weights(holdings, ctx.prices)
    tickers = list(weights.index)

    # Load returns
    asset_returns, as_of, data_source = await _load_prices_and_returns(ctx, tickers, inp.as_of)

    # Solve optimization
    try:
        target_w = quant_optimize_portfolio(
            returns=asset_returns,
            objective=inp.objective,
            current_weights=weights,
            max_concentration=inp.max_concentration,
            max_turnover=inp.max_turnover,
            risk_aversion=inp.risk_aversion,
            target_return=inp.target_return,
        )
    except Exception as e:
        raise ToolValidationError(f"Portfolio optimization failed: {e}") from e

    # Compute statistics before and after rebalance
    cov_res = ledoit_wolf_covariance(asset_returns)
    sigma = cov_res.matrix.to_numpy() * 252  # Annualized covariance
    mu = asset_returns.mean().to_numpy() * 252  # Annualized returns

    w0 = weights.to_numpy()
    w1 = target_w.reindex(weights.index).fillna(0.0).to_numpy()

    # Current stats
    curr_ret = float(w0 @ mu)
    curr_vol = float(np.sqrt(w0 @ sigma @ w0))
    curr_sharpe = curr_ret / curr_vol if curr_vol > 0 else 0.0

    # Target stats
    targ_ret = float(w1 @ mu)
    targ_vol = float(np.sqrt(w1 @ sigma @ w1))
    targ_sharpe = targ_ret / targ_vol if targ_vol > 0 else 0.0

    delta_risk = targ_vol - curr_vol

    # Calculate total portfolio market value
    panel = await ctx.prices.get_prices(tickers, as_of - timedelta(days=5), as_of)
    latest_close = panel.prices.iloc[-1]

    total_val = sum(h.quantity * float(latest_close[h.ticker]) for h in holdings)

    # Propose trades
    trades = []
    for ticker in tickers:
        w_curr = float(weights.get(ticker, 0.0))
        w_targ = float(target_w.get(ticker, 0.0))
        w_delta = w_targ - w_curr
        price = float(latest_close[ticker])

        trade_val = total_val * w_delta
        trade_qty = trade_val / price if price > 0 else 0.0

        if w_delta > 1e-5:
            action = "BUY"
        elif w_delta < -1e-5:
            action = "SELL"
        else:
            action = "HOLD"

        trades.append(
            RebalanceTrade(
                ticker=ticker,
                current_weight=w_curr,
                target_weight=w_targ,
                weight_delta=w_delta,
                action=action,
                trade_value_usd=abs(trade_val),
                trade_quantity=abs(trade_qty),
            )
        )

    return OptimizePortfolioOutput(
        portfolio_id=inp.portfolio_id,
        objective=inp.objective,
        as_of=as_of,
        current_expected_return=ctx.wrap_metric(
            "current_expected_return",
            curr_ret,
            "ratio",
            "historical_mean",
            as_of=as_of,
            data_sources=[data_source],
        ),
        current_volatility=ctx.wrap_metric(
            "current_volatility",
            curr_vol,
            "ratio",
            "ledoit_wolf_volatility",
            as_of=as_of,
            data_sources=[data_source],
        ),
        current_sharpe=ctx.wrap_metric(
            "current_sharpe",
            curr_sharpe,
            "ratio",
            "sharpe_ratio",
            as_of=as_of,
            data_sources=[data_source],
        ),
        target_expected_return=ctx.wrap_metric(
            "target_expected_return",
            targ_ret,
            "ratio",
            "historical_mean",
            as_of=as_of,
            data_sources=[data_source],
        ),
        target_volatility=ctx.wrap_metric(
            "target_volatility",
            targ_vol,
            "ratio",
            "ledoit_wolf_volatility",
            as_of=as_of,
            data_sources=[data_source],
        ),
        target_sharpe=ctx.wrap_metric(
            "target_sharpe",
            targ_sharpe,
            "ratio",
            "sharpe_ratio",
            as_of=as_of,
            data_sources=[data_source],
        ),
        ex_ante_delta_risk=ctx.wrap_metric(
            "ex_ante_delta_risk",
            delta_risk,
            "ratio",
            "delta_volatility",
            as_of=as_of,
            data_sources=[data_source],
        ),
        trades=trades,
        provenance=ctx.build_provenance(as_of=as_of, data_sources=[data_source]),
    )


@registry.tool(
    name="simulate_trade_impact",
    description=(
        "Simulate transaction costs and market impact when rebalancing to target portfolio "
        "weights. Use to verify if trading friction is within acceptable boundaries before "
        "executing a rebalance. Do NOT use for absolute risk metrics (use calculate_portfolio_var)."
    ),
    p95_latency_ms=250,
    est_cost_usd=0.0,
    cache_ttl_s=300,
    side_effects="READ_ONLY",
)
async def simulate_trade_impact(
    inp: SimulateTradeImpactInput, ctx: ToolContext
) -> SimulateTradeImpactOutput:
    holdings = await ctx.portfolios.get_holdings(
        inp.portfolio_id, tenant_id=ctx.tenant_id, as_of=inp.as_of
    )
    if not holdings:
        raise ToolValidationError(f"No holdings found for portfolio {inp.portfolio_id}")

    weights, _ = await fetch_priced_weights(holdings, ctx.prices)
    tickers = list(weights.index)

    as_of = inp.as_of or date.today()
    panel = await ctx.prices.get_prices(tickers, as_of - timedelta(days=5), as_of)
    latest_close = panel.prices.iloc[-1]

    total_val = sum(h.quantity * float(latest_close[h.ticker]) for h in holdings)

    total_trade_val = 0.0
    turnover = 0.0
    trade_values = []

    for ticker in tickers:
        w_curr = float(weights.get(ticker, 0.0))
        w_targ = float(inp.target_weights.get(ticker, 0.0))
        w_delta = w_targ - w_curr

        trade_val = total_val * abs(w_delta)
        total_trade_val += trade_val
        trade_values.append(trade_val)
        turnover += abs(w_delta)

    # Estimated transaction cost: 10 bps spread + non-linear power law market impact
    # cost = 0.001 * trade_value + 0.00005 * (trade_value ^ 1.2)
    est_cost = 0.001 * total_trade_val
    for tv in trade_values:
        est_cost += 0.00005 * (tv**1.2)

    turnover_pct = 0.5 * turnover

    return SimulateTradeImpactOutput(
        portfolio_id=inp.portfolio_id,
        as_of=panel.as_of,
        total_trade_value_usd=ctx.wrap_metric(
            "total_trade_value_usd",
            total_trade_val,
            "usd",
            "sum_of_trade_values",
            as_of=panel.as_of,
            data_sources=[panel.source],
        ),
        estimated_cost_usd=ctx.wrap_metric(
            "estimated_cost_usd",
            est_cost,
            "usd",
            "market_impact_cost_model",
            as_of=panel.as_of,
            data_sources=[panel.source],
        ),
        turnover_pct=ctx.wrap_metric(
            "turnover_pct",
            turnover_pct,
            "ratio",
            "half_sum_absolute_deltas",
            as_of=panel.as_of,
            data_sources=[panel.source],
        ),
        provenance=ctx.build_provenance(as_of=panel.as_of, data_sources=[panel.source]),
    )
