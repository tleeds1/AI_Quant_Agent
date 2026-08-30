"""Integration tests for `FilingsRepository` against real Postgres+pgvector.

`search_dense`/`search_bm25` use operators (`<=>`, `to_tsvector`/`ts_rank`)
sqlite has no equivalent for -- see `data/models.py::FilingChunk`'s
docstring -- so they, and cascade-delete (sqlite doesn't enforce foreign
keys by default), are exercised only here, never by the sqlite unit fixture.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quantagent.data.models import Filing as FilingRow
from quantagent.data.models import FilingChunk as FilingChunkRow
from quantagent.data.providers.embeddings import EMBEDDING_DIM
from quantagent.data.repositories.filings_repository import FilingMeta, FilingsRepository, NewChunk

_MODEL_NAME = "bge-small-en-v1.5"


def _filing_meta(
    filing_id: str, ticker: str = "NVDA", filed_at: date = date(2024, 2, 21)
) -> FilingMeta:
    return FilingMeta(
        id=filing_id,
        cik="0001045810",
        ticker=ticker,
        company_name="NVIDIA CORP",
        form_type="10-K",
        filed_at=filed_at,
        period_of_report=date(2024, 1, 28),
        primary_document_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
        source_tier="T1",
    )


def _chunk(filing_id: str, item: str, index: int, text: str) -> NewChunk:
    return NewChunk(
        chunk_id=f"{filing_id}#{item}#{index:04d}",
        item=item,
        section_path=f"10-K#Item {item}",
        text=text,
    )


def _vector(lead_value: float) -> list[float]:
    """A `EMBEDDING_DIM`-length vector distinguished only by its first
    component, so cosine distance orders results predictably.
    """
    return [lead_value] + [0.01] * (EMBEDDING_DIM - 1)


async def test_search_dense_orders_by_cosine_distance(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = FilingsRepository(session_factory)
    await repository.upsert_filing(_filing_meta("acc-dense-1"), content_sha256="h1")
    await repository.replace_chunks(
        "acc-dense-1",
        [
            _chunk("acc-dense-1", "1A", 0, "close match text"),
            _chunk("acc-dense-1", "1A", 1, "far match text"),
        ],
    )
    await repository.set_embedding("acc-dense-1#1A#0000", _vector(1.0), _MODEL_NAME)
    await repository.set_embedding("acc-dense-1#1A#0001", _vector(-1.0), _MODEL_NAME)

    results = await repository.search_dense(_vector(0.9), _MODEL_NAME, limit=10)

    assert [r.chunk.chunk_id for r in results] == ["acc-dense-1#1A#0000", "acc-dense-1#1A#0001"]
    assert results[0].score < results[1].score  # closer vector -> smaller cosine distance


async def test_search_dense_applies_freshness_filter(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = FilingsRepository(session_factory)
    await repository.upsert_filing(
        _filing_meta("acc-fresh-old", filed_at=date(2020, 1, 1)), content_sha256="h1"
    )
    await repository.upsert_filing(
        _filing_meta("acc-fresh-new", filed_at=date(2024, 1, 1)), content_sha256="h2"
    )
    for filing_id in ("acc-fresh-old", "acc-fresh-new"):
        await repository.replace_chunks(filing_id, [_chunk(filing_id, "1A", 0, "risk text")])
        await repository.set_embedding(f"{filing_id}#1A#0000", _vector(1.0), _MODEL_NAME)

    results = await repository.search_dense(
        _vector(1.0), _MODEL_NAME, filed_after=date(2023, 1, 1), limit=10
    )

    assert [r.filing.id for r in results] == ["acc-fresh-new"]


async def test_search_bm25_ranks_by_text_relevance(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = FilingsRepository(session_factory)
    await repository.upsert_filing(_filing_meta("acc-bm25-1"), content_sha256="h1")
    await repository.replace_chunks(
        "acc-bm25-1",
        [
            _chunk("acc-bm25-1", "1A", 0, "supply chain concentration risk supply chain"),
            _chunk("acc-bm25-1", "1A", 1, "unrelated discussion of office leases and facilities"),
        ],
    )

    results = await repository.search_bm25("supply chain concentration risk", limit=10)

    assert results
    assert results[0].chunk.chunk_id == "acc-bm25-1#1A#0000"


async def test_deleting_filing_cascades_chunks(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    repository = FilingsRepository(session_factory)
    await repository.upsert_filing(_filing_meta("acc-cascade-1"), content_sha256="h1")
    await repository.replace_chunks(
        "acc-cascade-1", [_chunk("acc-cascade-1", "1A", 0, "risk text")]
    )

    async with session_factory() as session, session.begin():
        filing_row = await session.get(FilingRow, "acc-cascade-1")
        assert filing_row is not None
        await session.delete(filing_row)

    async with session_factory() as session:
        stmt = select(FilingChunkRow).where(FilingChunkRow.filing_id == "acc-cascade-1")
        remaining = (await session.execute(stmt)).scalars().all()
    assert remaining == []
