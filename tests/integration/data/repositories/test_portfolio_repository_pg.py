from __future__ import annotations

from datetime import date

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quantagent.data.models import Holding as HoldingRow
from quantagent.data.models import Portfolio as PortfolioRow
from quantagent.data.models import Transaction as TransactionRow
from quantagent.data.repositories.portfolio_repository import (
    PortfolioRepository,
    TransactionInput,
)


async def test_full_round_trip_through_real_postgres_including_json_column(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id="pf_pg",
        tenant_id="tenant_pg",
        name="PG Round Trip",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={"theme_cap": 0.25, "sector_caps": {"tech": 0.4}},
    )
    await repository.upsert_holdings(
        "pf_pg", "tenant_pg", date(2026, 8, 22), [("AAPL", 10.0, 150.0)]
    )

    portfolio = await repository.get_portfolio("pf_pg", "tenant_pg")
    holdings = await repository.get_holdings("pf_pg", "tenant_pg")

    assert portfolio is not None
    assert portfolio.mandate_constraints == {"theme_cap": 0.25, "sector_caps": {"tech": 0.4}}
    assert holdings[0].ticker == "AAPL"


async def test_two_tenants_each_only_see_their_own_portfolio(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id="pf_shared_id",
        tenant_id="tenant_x",
        name="Tenant X Portfolio",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={},
    )

    as_tenant_x = await repository.get_portfolio("pf_shared_id", "tenant_x")
    as_tenant_y = await repository.get_portfolio("pf_shared_id", "tenant_y")

    assert as_tenant_x is not None
    assert as_tenant_y is None


async def test_duplicate_holding_for_same_portfolio_ticker_as_of_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id="pf_dup",
        tenant_id="tenant_dup",
        name="Dup Check",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={},
    )
    as_of = date(2026, 8, 22)

    async with session_factory() as session, session.begin():
        await session.execute(
            insert(HoldingRow).values(
                portfolio_id="pf_dup", ticker="AAPL", quantity=1, cost_basis=1, as_of=as_of
            )
        )

    with pytest.raises(IntegrityError):
        async with session_factory() as session, session.begin():
            await session.execute(
                insert(HoldingRow).values(
                    portfolio_id="pf_dup", ticker="AAPL", quantity=2, cost_basis=2, as_of=as_of
                )
            )


async def test_transaction_quantity_and_price_round_trip_without_float_drift(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id="pf_txn",
        tenant_id="tenant_txn",
        name="Transaction Round Trip",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={},
    )
    trade = TransactionInput("AAPL", "buy", 123.456789, 150.123456, date(2026, 8, 22))

    await repository.replace_transactions("pf_txn", "tenant_txn", [trade])
    transactions = await repository.get_transactions(
        "pf_txn", "tenant_txn", date(2026, 8, 1), date(2026, 8, 31)
    )

    assert transactions[0].quantity == pytest.approx(123.456789)
    assert transactions[0].price == pytest.approx(150.123456)


async def test_deleting_portfolio_cascades_transactions(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id="pf_cascade",
        tenant_id="tenant_cascade",
        name="Cascade Check",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={},
    )
    await repository.replace_transactions(
        "pf_cascade",
        "tenant_cascade",
        [TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 8, 22))],
    )

    async with session_factory() as session, session.begin():
        portfolio_row = await session.get(PortfolioRow, "pf_cascade")
        assert portfolio_row is not None
        await session.delete(portfolio_row)

    async with session_factory() as session:
        stmt = select(TransactionRow).where(TransactionRow.portfolio_id == "pf_cascade")
        remaining = (await session.execute(stmt)).scalars().all()
    assert remaining == []


async def test_invalid_transaction_side_raises_integrity_error(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = PortfolioRepository(session_factory)
    await repository.upsert_portfolio(
        id="pf_bad_side",
        tenant_id="tenant_bad_side",
        name="Bad Side Check",
        base_currency="USD",
        benchmark_ticker="SPY",
        mandate_constraints={},
    )

    with pytest.raises(IntegrityError):
        async with session_factory() as session, session.begin():
            await session.execute(
                insert(TransactionRow).values(
                    portfolio_id="pf_bad_side",
                    ticker="AAPL",
                    side="hold",
                    quantity=1,
                    price=1,
                    trade_date=date(2026, 8, 22),
                )
            )
