"""tests/unit/data/repositories/test_filings_repository.py

CRUD methods only -- `search_dense`/`search_bm25` are Postgres-only
(pgvector's `<=>` operator and the migration's generated `tsvector` column
have no sqlite equivalent) and are exercised by
`tests/integration/data/repositories/test_filings_repository_pg.py`
instead.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import replace
from datetime import date

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantagent.contracts.errors import DataError
from quantagent.data.models import Base
from quantagent.data.repositories.filings_repository import FilingMeta, FilingsRepository, NewChunk

_FILING_META = FilingMeta(
    id="0001045810-24-000123",
    cik="0001045810",
    ticker="NVDA",
    company_name="NVIDIA CORP",
    form_type="10-K",
    filed_at=date(2024, 2, 21),
    period_of_report=date(2024, 1, 28),
    primary_document_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
    source_tier="T1",
)


def _chunk(item: str, index: int, text: str = "chunk text") -> NewChunk:
    return NewChunk(
        chunk_id=f"0001045810-24-000123#{item}#{index:04d}",
        item=item,
        section_path=f"10-K#Item {item}",
        text=text,
    )


@pytest.fixture
async def session_factory() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


@pytest.fixture
def repository(session_factory: async_sessionmaker[AsyncSession]) -> FilingsRepository:
    return FilingsRepository(session_factory)


async def test_upsert_filing_then_get_round_trips(repository: FilingsRepository) -> None:
    filing_id = await repository.upsert_filing(_FILING_META, content_sha256="abc123")
    fetched = await repository.get_filing(filing_id)
    assert fetched == _FILING_META


async def test_upsert_filing_is_idempotent_on_same_id(repository: FilingsRepository) -> None:
    await repository.upsert_filing(_FILING_META, content_sha256="abc123")
    updated_meta = replace(_FILING_META, company_name="NVIDIA CORPORATION")
    await repository.upsert_filing(updated_meta, content_sha256="def456")
    fetched = await repository.get_filing(_FILING_META.id)
    assert fetched is not None
    assert fetched.company_name == "NVIDIA CORPORATION"


async def test_get_filing_returns_none_for_unknown_id(repository: FilingsRepository) -> None:
    assert await repository.get_filing("does-not-exist") is None


async def test_list_filings_filters_by_form_type_and_since(
    repository: FilingsRepository,
) -> None:
    await repository.upsert_filing(_FILING_META, content_sha256="abc123")
    older = replace(_FILING_META, id="old-filing", filed_at=date(2020, 1, 1))
    await repository.upsert_filing(older, content_sha256="xyz789")

    results = await repository.list_filings("NVDA", form_types=["10-K"], since=date(2023, 1, 1))
    assert [r.id for r in results] == [_FILING_META.id]


async def test_replace_chunks_writes_and_replaces(repository: FilingsRepository) -> None:
    await repository.upsert_filing(_FILING_META, content_sha256="abc123")
    written = await repository.replace_chunks(
        _FILING_META.id, [_chunk("1", 0), _chunk("1", 1), _chunk("1A", 0)]
    )
    assert written == 3

    chunk = await repository.get_chunk("0001045810-24-000123#1#0000")
    assert chunk is not None
    assert chunk.chunk_text == "chunk text"

    replaced = await repository.replace_chunks(_FILING_META.id, [_chunk("1", 0, text="new text")])
    assert replaced == 1
    assert await repository.get_chunk("0001045810-24-000123#1#0001") is None
    chunk_after = await repository.get_chunk("0001045810-24-000123#1#0000")
    assert chunk_after is not None
    assert chunk_after.chunk_text == "new text"


async def test_get_chunk_returns_none_for_unknown_id(repository: FilingsRepository) -> None:
    assert await repository.get_chunk("does-not-exist") is None


async def test_get_chunks_missing_embeddings_and_set_embedding(
    repository: FilingsRepository,
) -> None:
    await repository.upsert_filing(_FILING_META, content_sha256="abc123")
    await repository.replace_chunks(_FILING_META.id, [_chunk("1", 0), _chunk("1", 1)])

    missing = await repository.get_chunks_missing_embeddings("bge-small-en-v1.5")
    assert {c.chunk_id for c in missing} == {
        "0001045810-24-000123#1#0000",
        "0001045810-24-000123#1#0001",
    }

    await repository.set_embedding(
        "0001045810-24-000123#1#0000", [0.1, 0.2, 0.3], "bge-small-en-v1.5"
    )
    still_missing = await repository.get_chunks_missing_embeddings("bge-small-en-v1.5")
    assert [c.chunk_id for c in still_missing] == ["0001045810-24-000123#1#0001"]


async def test_set_embedding_unknown_chunk_raises(repository: FilingsRepository) -> None:
    with pytest.raises(DataError):
        await repository.set_embedding("does-not-exist", [0.1, 0.2], "model")


# Cascade-delete (filing -> chunks) is NOT tested here: sqlite does not
# enforce foreign keys (including ON DELETE CASCADE) by default, so this
# fixture cannot verify it -- same precedent as
# test_deleting_portfolio_cascades_transactions, which lives only in
# tests/integration/data/repositories/test_portfolio_repository_pg.py.
# See tests/integration/data/repositories/test_filings_repository_pg.py.
