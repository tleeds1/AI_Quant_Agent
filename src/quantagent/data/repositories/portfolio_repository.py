from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Any, cast

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from quantagent.contracts.errors import DataError
from quantagent.data.models import Holding as HoldingRow
from quantagent.data.models import Portfolio as PortfolioRow
from quantagent.data.models import Transaction as TransactionRow
from quantagent.data.models import TransactionSide
from quantagent.data.repositories.base import RepositoryBase


@dataclass(frozen=True, slots=True)
class PortfolioMeta:
    id: str
    tenant_id: str
    name: str
    base_currency: str
    benchmark_ticker: str
    mandate_constraints: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Holding:
    portfolio_id: str
    ticker: str
    quantity: float
    cost_basis: float
    as_of: date


def _to_meta(row: PortfolioRow) -> PortfolioMeta:
    return PortfolioMeta(
        id=row.id,
        tenant_id=row.tenant_id,
        name=row.name,
        base_currency=row.base_currency,
        benchmark_ticker=row.benchmark_ticker,
        mandate_constraints=row.mandate_constraints,
    )


def _to_holding(row: HoldingRow) -> Holding:
    return Holding(
        portfolio_id=row.portfolio_id,
        ticker=row.ticker,
        quantity=float(row.quantity),
        cost_basis=float(row.cost_basis),
        as_of=row.as_of,
    )


@dataclass(frozen=True, slots=True)
class Transaction:
    portfolio_id: str
    ticker: str
    side: TransactionSide
    quantity: float
    price: float
    trade_date: date


@dataclass(frozen=True, slots=True)
class TransactionInput:
    """Write-side payload for `PortfolioRepository.replace_transactions`.

    A named record rather than a positional tuple: `quantity` and `price`
    are both bare floats, and silently transposing them at a call site
    would corrupt money with no type error.
    """

    ticker: str
    side: TransactionSide
    quantity: float
    price: float
    trade_date: date


def _to_transaction(row: TransactionRow) -> Transaction:
    return Transaction(
        portfolio_id=row.portfolio_id,
        ticker=row.ticker,
        # The `ck_transactions_side` CHECK constraint is the enforcement point;
        # SQLAlchemy types the column as plain `str`.
        side=cast(TransactionSide, row.side),
        quantity=float(row.quantity),
        price=float(row.price),
        trade_date=row.trade_date,
    )


class PortfolioRepository(RepositoryBase):
    """Tenant-scoped reads and idempotent writes for portfolios and holdings.

    Never leaks SQLAlchemy ORM rows across the repository boundary -- every
    public method returns `PortfolioMeta`/`Holding` frozen dataclasses.
    """

    async def get_portfolio(self, portfolio_id: str, tenant_id: str) -> PortfolioMeta | None:
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            stmt = select(PortfolioRow).where(
                PortfolioRow.id == portfolio_id, PortfolioRow.tenant_id == tenant_id
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _to_meta(row) if row is not None else None

    async def get_holdings(
        self, portfolio_id: str, tenant_id: str, as_of: date | None = None
    ) -> list[Holding]:
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            resolved_as_of = as_of
            if resolved_as_of is None:
                resolved_as_of = await self._latest_as_of(session, portfolio_id, tenant_id)
            if resolved_as_of is None:
                return []
            stmt = (
                select(HoldingRow)
                .join(PortfolioRow, HoldingRow.portfolio_id == PortfolioRow.id)
                .where(
                    HoldingRow.portfolio_id == portfolio_id,
                    PortfolioRow.tenant_id == tenant_id,
                    HoldingRow.as_of == resolved_as_of,
                )
                .order_by(HoldingRow.ticker)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_holding(row) for row in rows]

    async def _latest_as_of(
        self, session: AsyncSession, portfolio_id: str, tenant_id: str
    ) -> date | None:
        stmt = (
            select(HoldingRow.as_of)
            .join(PortfolioRow, HoldingRow.portfolio_id == PortfolioRow.id)
            .where(HoldingRow.portfolio_id == portfolio_id, PortfolioRow.tenant_id == tenant_id)
            .order_by(HoldingRow.as_of.desc())
            .limit(1)
        )
        return (await session.execute(stmt)).scalar_one_or_none()

    async def upsert_portfolio(
        self,
        id: str,
        tenant_id: str,
        name: str,
        base_currency: str,
        benchmark_ticker: str,
        mandate_constraints: dict[str, Any],
    ) -> str:
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            existing = await session.get(PortfolioRow, id)
            if existing is not None:
                existing.tenant_id = tenant_id
                existing.name = name
                existing.base_currency = base_currency
                existing.benchmark_ticker = benchmark_ticker
                existing.mandate_constraints = mandate_constraints
            else:
                session.add(
                    PortfolioRow(
                        id=id,
                        tenant_id=tenant_id,
                        name=name,
                        base_currency=base_currency,
                        benchmark_ticker=benchmark_ticker,
                        mandate_constraints=mandate_constraints,
                    )
                )
            await session.commit()
            return id

    async def upsert_holdings(
        self,
        portfolio_id: str,
        tenant_id: str,
        as_of: date,
        holdings: Sequence[tuple[str, float, float]],
    ) -> None:
        """Fully replaces the holding set for `(portfolio_id, as_of)`."""
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            portfolio = await session.get(PortfolioRow, portfolio_id)
            if portfolio is None or portfolio.tenant_id != tenant_id:
                raise DataError(f"portfolio {portfolio_id!r} not found for tenant {tenant_id!r}")

            existing_stmt = select(HoldingRow).where(
                HoldingRow.portfolio_id == portfolio_id, HoldingRow.as_of == as_of
            )
            existing_rows = (await session.execute(existing_stmt)).scalars().all()
            for row in existing_rows:
                await session.delete(row)
            await session.flush()

            for ticker, quantity, cost_basis in holdings:
                session.add(
                    HoldingRow(
                        portfolio_id=portfolio_id,
                        ticker=ticker,
                        quantity=Decimal(str(quantity)),
                        cost_basis=Decimal(str(cost_basis)),
                        as_of=as_of,
                    )
                )
            await session.commit()

    async def get_transactions(
        self, portfolio_id: str, tenant_id: str, start: date, end: date
    ) -> list[Transaction]:
        """Executed trades with `start <= trade_date <= end`, oldest first.

        Both bounds are inclusive. Tenant scope comes from the join to
        `portfolios` -- `transactions` carries no `tenant_id` of its own.
        """
        self._require_tenant(tenant_id)
        if start > end:
            raise DataError(f"start {start} must not be after end {end}")
        async with self._session_factory() as session:
            stmt = (
                select(TransactionRow)
                .join(PortfolioRow, TransactionRow.portfolio_id == PortfolioRow.id)
                .where(
                    TransactionRow.portfolio_id == portfolio_id,
                    PortfolioRow.tenant_id == tenant_id,
                    TransactionRow.trade_date >= start,
                    TransactionRow.trade_date <= end,
                )
                .order_by(TransactionRow.trade_date, TransactionRow.ticker, TransactionRow.id)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_transaction(row) for row in rows]

    async def replace_transactions(
        self, portfolio_id: str, tenant_id: str, transactions: Sequence[TransactionInput]
    ) -> int:
        """Fully replaces the trade history for `portfolio_id`; returns rows written.

        No M2 tool writes transactions -- the tool catalogue is read-only for
        them -- so this exists only so `scripts/seed_portfolio.py` and the
        repository tests can seed a history. Full replacement rather than
        append is what keeps the seed script idempotent: `transactions` has
        no natural key to upsert on (see `models.Transaction`).
        """
        self._require_tenant(tenant_id)
        async with self._session_factory() as session:
            portfolio = await session.get(PortfolioRow, portfolio_id)
            if portfolio is None or portfolio.tenant_id != tenant_id:
                raise DataError(f"portfolio {portfolio_id!r} not found for tenant {tenant_id!r}")

            await session.execute(
                delete(TransactionRow).where(TransactionRow.portfolio_id == portfolio_id)
            )
            session.add_all(
                TransactionRow(
                    portfolio_id=portfolio_id,
                    ticker=trade.ticker,
                    side=trade.side,
                    quantity=Decimal(str(trade.quantity)),
                    price=Decimal(str(trade.price)),
                    trade_date=trade.trade_date,
                )
                for trade in transactions
            )
            await session.commit()
            return len(transactions)
