"""Seed a demo portfolio for local development and the M1/M2 CLI risk report
and tool layer.

Idempotent: safe to re-run (upserts the portfolio, fully replaces its
holdings as of today, and fully replaces its transaction history).
"""

from __future__ import annotations

import asyncio
from datetime import date

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from quantagent.config import settings
from quantagent.data.repositories.portfolio_repository import (
    PortfolioRepository,
    TransactionInput,
)

DEMO_PORTFOLIO_ID = "pf_demo"
DEMO_TENANT_ID = "tenant_demo"

# ~9 liquid, multi-sector holdings so the resulting risk report is meaningful:
# mega-cap tech, semis, financials, healthcare, and consumer.
DEMO_HOLDINGS: list[tuple[str, float, float]] = [
    ("AAPL", 50.0, 150.0),
    ("MSFT", 40.0, 280.0),
    ("NVDA", 60.0, 90.0),
    ("AMD", 80.0, 100.0),
    ("JPM", 45.0, 140.0),
    ("JNJ", 55.0, 160.0),
    ("PG", 50.0, 145.0),
    ("XOM", 60.0, 100.0),
    ("KO", 100.0, 55.0),
]

# A trade history covering both sides and at least one same-day, same-ticker
# pair (AAPL 2024-03-01), so get_transactions exercises the no-unique-key case.
DEMO_TRANSACTIONS: list[TransactionInput] = [
    TransactionInput("AAPL", "buy", 30.0, 145.0, date(2024, 1, 10)),
    TransactionInput("AAPL", "buy", 15.0, 152.0, date(2024, 3, 1)),
    TransactionInput("AAPL", "buy", 5.0, 149.5, date(2024, 3, 1)),
    TransactionInput("MSFT", "buy", 40.0, 280.0, date(2024, 1, 15)),
    TransactionInput("NVDA", "buy", 40.0, 70.0, date(2024, 1, 20)),
    TransactionInput("NVDA", "buy", 30.0, 95.0, date(2024, 6, 5)),
    TransactionInput("NVDA", "sell", 10.0, 105.0, date(2024, 9, 12)),
    TransactionInput("AMD", "buy", 80.0, 100.0, date(2024, 2, 8)),
    TransactionInput("JPM", "buy", 45.0, 140.0, date(2024, 4, 3)),
    TransactionInput("JNJ", "buy", 55.0, 160.0, date(2024, 5, 22)),
    TransactionInput("PG", "buy", 50.0, 145.0, date(2024, 7, 17)),
    TransactionInput("XOM", "buy", 60.0, 100.0, date(2024, 8, 9)),
    TransactionInput("KO", "buy", 100.0, 55.0, date(2024, 10, 30)),
]


async def main() -> None:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    repository = PortfolioRepository(session_factory)

    await repository.upsert_portfolio(
        id=DEMO_PORTFOLIO_ID,
        tenant_id=DEMO_TENANT_ID,
        name="Demo Growth Portfolio",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={"theme_cap": 0.25},
    )
    await repository.upsert_holdings(DEMO_PORTFOLIO_ID, DEMO_TENANT_ID, date.today(), DEMO_HOLDINGS)
    await repository.replace_transactions(DEMO_PORTFOLIO_ID, DEMO_TENANT_ID, DEMO_TRANSACTIONS)

    print(
        f"Seeded {DEMO_PORTFOLIO_ID!r} with {len(DEMO_HOLDINGS)} holdings and "
        f"{len(DEMO_TRANSACTIONS)} transactions."
    )
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
