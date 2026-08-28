from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantagent.config import settings

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def _migrated_database() -> None:
    """Run Alembic migrations once per test session against the real,
    Dockerized Postgres (see docker-compose.yml). Requires `docker compose
    up -d` to already be running.
    """
    config = Config(str(REPO_ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(REPO_ROOT / "migrations"))
    command.upgrade(config, "head")


@pytest.fixture
async def session_factory(
    _migrated_database: None,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(settings.database_url)
    yield async_sessionmaker(engine, expire_on_commit=False)
    async with engine.begin() as conn:
        await conn.execute(
            text("TRUNCATE TABLE portfolios, holdings, transactions RESTART IDENTITY CASCADE")
        )
    await engine.dispose()
