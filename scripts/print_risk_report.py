"""CLI: print a full risk report for a seeded portfolio with zero LLM
involvement (docs/guideline.md §11, M1 DoD; M2 refactor).

Thin wrapper over the tool layer -- no business logic of its own, just three
`registry.invoke(...)` calls (`get_portfolio`, `get_holdings`,
`generate_risk_report`) and string formatting. A CLI is itself just another
tool caller (architecture.md §4.3), the same as the future MCP server.

    uv run python scripts/print_risk_report.py
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import quantagent.tools  # noqa: F401 -- import side effect: populates the registry
from quantagent.config import settings
from quantagent.contracts.tools import (
    GenerateRiskReportInput,
    GetHoldingsOutput,
    PortfolioOutput,
    ReportArtifact,
)
from quantagent.data.cache import CacheClient
from quantagent.data.providers.factors import KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry

# Matches scripts/seed_portfolio.py's DEMO_PORTFOLIO_ID/DEMO_TENANT_ID -- not imported
# directly since scripts/ are invoked standalone (`python scripts/foo.py`), not as a
# package, so cross-script imports would depend on the caller's working directory.
DEFAULT_PORTFOLIO_ID = "pf_demo"
DEFAULT_TENANT_ID = "tenant_demo"
DEFAULT_LOOKBACK_DAYS = 700  # calendar days; at alpha=0.95 leaves >=20 tail observations
# (MIN_CVAR_TAIL_OBSERVATIONS) after calendar alignment, comfortably above the
# MIN_VAR_OBSERVATIONS=250 floor too.


def render_report(
    portfolio: PortfolioOutput, holdings: GetHoldingsOutput, report: ReportArtifact
) -> str:
    """Pure formatting -- no I/O, unit-testable in isolation."""
    lines = [
        f"Risk report: {portfolio.name} ({portfolio.portfolio_id})",
        f"Base currency: {portfolio.base_currency}  Benchmark: {portfolio.benchmark_ticker}",
        f"As of: {report.as_of}",
        "",
        "Holdings:",
    ]
    for holding in sorted(holdings.holdings, key=lambda h: -h.weight.value):
        lines.append(f"  {holding.ticker:<6} weight={holding.weight.value:.2%}")
    lines.append("")
    lines.append("Risk metrics:")
    for section in report.sections:
        for metric in section.metrics:
            lines.append(
                f"  {metric.metric_id:<24} {metric.value:.4f} {metric.unit} ({metric.method})"
            )
    if report.warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  - {w}" for w in report.warnings)
    return "\n".join(lines)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-id", default=DEFAULT_PORTFOLIO_ID)
    parser.add_argument("--tenant-id", default=DEFAULT_TENANT_ID)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--lookback-days", type=int, default=DEFAULT_LOOKBACK_DAYS)
    parser.add_argument(
        "--method", choices=["historical", "parametric", "monte_carlo"], default="historical"
    )
    return parser.parse_args(argv)


async def _run(args: argparse.Namespace) -> str:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    cache = CacheClient.from_settings()
    ctx = ToolContext(
        tenant_id=args.tenant_id,
        portfolios=PortfolioRepository(session_factory),
        prices=YFinancePriceProvider(cache=cache),
        fundamentals=YFinanceFundamentalsProvider(cache=cache),
        factors=KenFrenchFactorDataProvider(cache=cache),
        cache=cache,
    )
    try:
        portfolio = await registry.invoke("get_portfolio", {"portfolio_id": args.portfolio_id}, ctx)
        holdings = await registry.invoke("get_holdings", {"portfolio_id": args.portfolio_id}, ctx)
        report = await registry.invoke(
            "generate_risk_report",
            GenerateRiskReportInput(
                portfolio_id=args.portfolio_id,
                alpha=args.alpha,
                horizon_days=args.horizon_days,
                lookback_days=args.lookback_days,
                method=args.method,
            ).model_dump(mode="json"),
            ctx,
        )
    finally:
        await engine.dispose()
        await cache.close()
    assert isinstance(portfolio, PortfolioOutput)
    assert isinstance(holdings, GetHoldingsOutput)
    assert isinstance(report, ReportArtifact)
    return render_report(portfolio, holdings, report)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    print(asyncio.run(_run(args)))


if __name__ == "__main__":
    main()
