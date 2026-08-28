from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantagent.contracts.errors import DataError
from quantagent.data.models import Base
from quantagent.data.repositories.portfolio_repository import (
    PortfolioRepository,
    TransactionInput,
)

DEMO_PORTFOLIO_KWARGS = {
    "id": "pf_1",
    "tenant_id": "tenant_a",
    "name": "Demo",
    "base_currency": "USD",
    "benchmark_ticker": "SPY",
    "mandate_constraints": {},
}


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def repository(
    session_factory: async_sessionmaker[AsyncSession],
) -> PortfolioRepository:
    return PortfolioRepository(session_factory)


async def test_get_portfolio_returns_none_for_unknown_id(repository: PortfolioRepository) -> None:
    result = await repository.get_portfolio("nope", "tenant_a")

    assert result is None


async def test_get_portfolio_raises_when_tenant_id_missing(
    repository: PortfolioRepository,
) -> None:
    with pytest.raises(DataError):
        await repository.get_portfolio("pf_1", "")


async def test_get_holdings_raises_when_tenant_id_missing(
    repository: PortfolioRepository,
) -> None:
    with pytest.raises(DataError):
        await repository.get_holdings("pf_1", "")


async def test_upsert_portfolio_then_get_portfolio_round_trips(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)

    result = await repository.get_portfolio("pf_1", "tenant_a")

    assert result is not None
    assert result.name == "Demo"


async def test_get_portfolio_with_wrong_tenant_returns_none(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)

    result = await repository.get_portfolio("pf_1", "tenant_b")

    assert result is None


async def test_upsert_portfolio_is_idempotent(repository: PortfolioRepository) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    updated = dict(DEMO_PORTFOLIO_KWARGS, name="Demo Updated")

    await repository.upsert_portfolio(**updated)
    result = await repository.get_portfolio("pf_1", "tenant_a")

    assert result is not None
    assert result.name == "Demo Updated"


async def test_upsert_holdings_then_get_holdings_round_trips(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    as_of = date(2026, 8, 22)

    await repository.upsert_holdings(
        "pf_1", "tenant_a", as_of, [("AAPL", 10.0, 150.0), ("MSFT", 5.0, 300.0)]
    )
    holdings = await repository.get_holdings("pf_1", "tenant_a")

    assert [h.ticker for h in holdings] == ["AAPL", "MSFT"]
    assert holdings[0].quantity == pytest.approx(10.0)


async def test_upsert_holdings_fully_replaces_existing_set(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    as_of = date(2026, 8, 22)
    await repository.upsert_holdings("pf_1", "tenant_a", as_of, [("AAPL", 10.0, 150.0)])

    await repository.upsert_holdings("pf_1", "tenant_a", as_of, [("MSFT", 5.0, 300.0)])
    holdings = await repository.get_holdings("pf_1", "tenant_a")

    assert [h.ticker for h in holdings] == ["MSFT"]


async def test_get_holdings_resolves_latest_as_of_when_omitted(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    await repository.upsert_holdings("pf_1", "tenant_a", date(2026, 8, 1), [("AAPL", 1.0, 1.0)])
    await repository.upsert_holdings("pf_1", "tenant_a", date(2026, 8, 22), [("MSFT", 1.0, 1.0)])

    holdings = await repository.get_holdings("pf_1", "tenant_a")

    assert [h.ticker for h in holdings] == ["MSFT"]


async def test_get_holdings_returns_empty_list_when_none_exist(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)

    holdings = await repository.get_holdings("pf_1", "tenant_a")

    assert holdings == []


async def test_upsert_holdings_raises_for_unknown_portfolio(
    repository: PortfolioRepository,
) -> None:
    with pytest.raises(DataError):
        await repository.upsert_holdings(
            "nope", "tenant_a", date(2026, 8, 22), [("AAPL", 1.0, 1.0)]
        )


async def test_replace_transactions_then_get_transactions_round_trips(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    trades = [
        TransactionInput("AAPL", "buy", 10.0, 150.0, date(2026, 1, 5)),
        TransactionInput("MSFT", "sell", 3.0, 310.0, date(2026, 2, 1)),
    ]

    written = await repository.replace_transactions("pf_1", "tenant_a", trades)
    transactions = await repository.get_transactions(
        "pf_1", "tenant_a", date(2026, 1, 1), date(2026, 3, 1)
    )

    assert written == 2
    assert [t.ticker for t in transactions] == ["AAPL", "MSFT"]
    assert transactions[0].side == "buy"
    assert transactions[0].quantity == pytest.approx(10.0)


async def test_get_transactions_bounds_are_inclusive(repository: PortfolioRepository) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    trades = [
        TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 1, 1)),
        TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 1, 10)),
        TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 1, 20)),
    ]
    await repository.replace_transactions("pf_1", "tenant_a", trades)

    transactions = await repository.get_transactions(
        "pf_1", "tenant_a", date(2026, 1, 1), date(2026, 1, 10)
    )

    assert [t.trade_date for t in transactions] == [date(2026, 1, 1), date(2026, 1, 10)]


async def test_get_transactions_allows_two_same_day_same_ticker_trades(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    trades = [
        TransactionInput("AAPL", "buy", 5.0, 100.0, date(2026, 1, 5)),
        TransactionInput("AAPL", "buy", 5.0, 101.0, date(2026, 1, 5)),
    ]

    written = await repository.replace_transactions("pf_1", "tenant_a", trades)
    transactions = await repository.get_transactions(
        "pf_1", "tenant_a", date(2026, 1, 5), date(2026, 1, 5)
    )

    assert written == 2
    assert len(transactions) == 2


async def test_replace_transactions_fully_replaces_existing_set(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    await repository.replace_transactions(
        "pf_1", "tenant_a", [TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 1, 1))]
    )

    await repository.replace_transactions(
        "pf_1", "tenant_a", [TransactionInput("MSFT", "buy", 1.0, 1.0, date(2026, 1, 1))]
    )
    transactions = await repository.get_transactions(
        "pf_1", "tenant_a", date(2026, 1, 1), date(2026, 1, 1)
    )

    assert [t.ticker for t in transactions] == ["MSFT"]


async def test_get_transactions_raises_when_tenant_id_missing(
    repository: PortfolioRepository,
) -> None:
    with pytest.raises(DataError):
        await repository.get_transactions("pf_1", "", date(2026, 1, 1), date(2026, 1, 2))


async def test_get_transactions_with_wrong_tenant_returns_empty_list(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)
    await repository.replace_transactions(
        "pf_1", "tenant_a", [TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 1, 1))]
    )

    transactions = await repository.get_transactions(
        "pf_1", "tenant_b", date(2026, 1, 1), date(2026, 1, 2)
    )

    assert transactions == []


async def test_get_transactions_raises_when_start_after_end(
    repository: PortfolioRepository,
) -> None:
    with pytest.raises(DataError):
        await repository.get_transactions("pf_1", "tenant_a", date(2026, 1, 10), date(2026, 1, 1))


async def test_get_transactions_returns_empty_list_when_none_exist(
    repository: PortfolioRepository,
) -> None:
    await repository.upsert_portfolio(**DEMO_PORTFOLIO_KWARGS)

    transactions = await repository.get_transactions(
        "pf_1", "tenant_a", date(2026, 1, 1), date(2026, 12, 31)
    )

    assert transactions == []


async def test_replace_transactions_raises_for_unknown_portfolio(
    repository: PortfolioRepository,
) -> None:
    with pytest.raises(DataError):
        await repository.replace_transactions(
            "nope", "tenant_a", [TransactionInput("AAPL", "buy", 1.0, 1.0, date(2026, 1, 1))]
        )
