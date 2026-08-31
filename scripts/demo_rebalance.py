# mypy: ignore-errors
"""CLI: run portfolio optimization and trade simulation for the demo portfolio.

    uv run python scripts/demo_rebalance.py
"""

from __future__ import annotations

import asyncio
from datetime import date

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import quantagent.tools  # noqa: F401
from quantagent.config import settings
from quantagent.contracts.tools import (
    OptimizePortfolioInput,
    OptimizePortfolioOutput,
    SimulateTradeImpactInput,
    SimulateTradeImpactOutput,
)
from quantagent.data.cache import CacheClient
from quantagent.data.providers.factors import KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.unit.tools.builders import build_holding, build_tool_context

DEMO_PORTFOLIO_ID = "pf_demo"
DEMO_TENANT_ID = "tenant_demo"


async def run_rebalance(ctx: ToolContext) -> None:
    print("1. Optimizing portfolio 'pf_demo' using 'min_variance' objective...")
    opt_input = OptimizePortfolioInput(
        portfolio_id=DEMO_PORTFOLIO_ID,
        objective="min_variance",
        max_concentration=0.30,  # Max 30% in any single name
    )
    opt_res: OptimizePortfolioOutput = await registry.invoke("optimize_portfolio", opt_input.model_dump(mode="json"), ctx)

    print("\n--- Portfolio Optimization Results ---")
    print(f"Objective: {opt_res.objective.upper()}")
    print(f"As of: {opt_res.as_of}")
    print(f"Current Expected Return: {opt_res.current_expected_return.value:.2%}")
    print(f"Current Volatility:      {opt_res.current_volatility.value:.2%}")
    print(f"Current Sharpe Ratio:    {opt_res.current_sharpe.value:.2f}")
    print(f"Target Expected Return:  {opt_res.target_expected_return.value:.2%}")
    print(f"Target Volatility:      {opt_res.target_volatility.value:.2%}")
    print(f"Target Sharpe Ratio:    {opt_res.target_sharpe.value:.2f}")
    print(f"Ex-Ante Delta Risk:      {opt_res.ex_ante_delta_risk.value:.2%}")

    print("\nProposed Rebalance Trades:")
    for t in sorted(opt_res.trades, key=lambda tr: tr.weight_delta):
        if t.action == "HOLD":
            continue
        print(
            f"  {t.action:<5} {t.ticker:<5} "
            f"weight: {t.current_weight:6.2%} -> {t.target_weight:6.2%} "
            f"(delta: {t.weight_delta:+6.2%}) | "
            f"value: ${t.trade_value_usd:,.2f} | qty: {t.trade_quantity:,.2f}"
        )

    print("\n2. Simulating trade impact of the proposed weights...")
    target_weights = {t.ticker: t.target_weight for t in opt_res.trades}
    sim_input = SimulateTradeImpactInput(
        portfolio_id=DEMO_PORTFOLIO_ID,
        target_weights=target_weights,
    )
    sim_res: SimulateTradeImpactOutput = await registry.invoke("simulate_trade_impact", sim_input.model_dump(mode="json"), ctx)

    print("\n--- Trade Impact Simulation Results ---")
    print(f"Total Trade Value:   ${sim_res.total_trade_value_usd.value:,.2f}")
    print(f"Estimated Cost:      ${sim_res.estimated_cost_usd.value:,.2f}")
    print(f"Portfolio Turnover:  {sim_res.turnover_pct.value:.2%}")


async def main() -> None:
    try:
        # Try real connection
        engine = create_async_engine(settings.database_url)
        session_factory = async_sessionmaker(engine, expire_on_commit=False)
        cache = CacheClient.from_settings()
        ctx = ToolContext(
            tenant_id=DEMO_TENANT_ID,
            portfolios=PortfolioRepository(session_factory),
            prices=YFinancePriceProvider(cache=cache),
            fundamentals=YFinanceFundamentalsProvider(cache=cache),
            factors=KenFrenchFactorDataProvider(cache=cache),
            cache=cache,
        )
        # Test connection by fetching portfolio
        await ctx.portfolios.get_portfolio(DEMO_PORTFOLIO_ID, tenant_id=DEMO_TENANT_ID)
        await run_rebalance(ctx)
        await engine.dispose()
        await cache.close()
    except Exception as e:
        print(f"\nReal database/cache connection not available: {e}")
        print("Falling back to offline deterministic mode using FakePriceProvider and FakePortfolioRepository...\n")
        
        # Build offline mock context with realistic demo holdings
        from tests.unit.tools.builders import build_portfolio_meta
        portfolio = build_portfolio_meta(id=DEMO_PORTFOLIO_ID, tenant_id=DEMO_TENANT_ID)
        ctx = build_tool_context(
            portfolio=portfolio,
            holdings=[
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="AAPL", quantity=50.0, cost_basis=150.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="MSFT", quantity=40.0, cost_basis=280.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="NVDA", quantity=60.0, cost_basis=90.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="AMD", quantity=80.0, cost_basis=100.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="JPM", quantity=45.0, cost_basis=140.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="JNJ", quantity=55.0, cost_basis=160.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="PG", quantity=50.0, cost_basis=145.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="XOM", quantity=60.0, cost_basis=100.0),
                build_holding(portfolio_id=DEMO_PORTFOLIO_ID, ticker="KO", quantity=100.0, cost_basis=55.0),
            ],
            tenant_id=DEMO_TENANT_ID,
        )
        await run_rebalance(ctx)


if __name__ == "__main__":
    asyncio.run(main())
