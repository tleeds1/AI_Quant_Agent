"""data/repositories/filings_repository.py -- reads and writes for SEC
EDGAR filings and their chunks (architecture.md §4.7).

Deliberately does NOT inherit `RepositoryBase` and takes no `tenant_id`
parameter anywhere. SEC filings are public documents, identical for every
tenant -- there is no tenant-owning parent to join through (unlike
`Holding`/`Transaction`'s join to `Portfolio.tenant_id`), so I9's stated
scope ("portfolio data is tenant-scoped at the repository layer") never
covered this table. A `tenant_id` parameter this class validated but then
ignored would be an attractive nuisance implying an isolation guarantee
that doesn't exist -- see docs/PROGRESS.md's M5 section for the full
reasoning behind this deviation from every other repository's shape.

`search_dense` and `search_bm25` are Postgres-only (pgvector's `<=>`
operator and the migration's generated `tsvector` column both have no
sqlite equivalent) -- exercised by `tests/integration/...` against a real
database, never the sqlite unit-test fixture the CRUD methods use.

`NewChunk` is this module's own write-side input type for `replace_chunks`
-- deliberately NOT `rag.chunk.Chunk`: `.importlinter`'s `data-purity`
contract forbids `data/` from importing `rag/` at all. `rag/ingest.py`
(which may import both) converts its `Chunk` objects into `NewChunk` at
the call site.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import cast

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from quantagent.contracts.errors import DataError
from quantagent.contracts.evidence import SourceTier
from quantagent.data.models import Filing as FilingRow
from quantagent.data.models import FilingChunk as FilingChunkRow

_DEFAULT_SEARCH_LIMIT = 50


@dataclass(frozen=True, slots=True)
class FilingMeta:
    id: str
    cik: str
    ticker: str
    company_name: str
    form_type: str
    filed_at: date
    period_of_report: date | None
    primary_document_url: str
    source_tier: SourceTier


@dataclass(frozen=True, slots=True)
class NewChunk:
    """Write-side input for `replace_chunks` -- one item-scoped chunk of
    text, pre-chunking-algorithm's own concern (`rag.chunk.Chunk`) but
    reduced to exactly what this repository needs to persist a row.
    """

    chunk_id: str
    item: str
    section_path: str
    text: str


@dataclass(frozen=True, slots=True)
class FilingChunkRecord:
    chunk_id: str
    filing_id: str
    item: str
    section_path: str
    chunk_index: int
    chunk_text: str
    char_count: int
    embedding_model: str | None


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: FilingChunkRecord
    filing: FilingMeta
    score: float


def _to_filing_meta(row: FilingRow) -> FilingMeta:
    return FilingMeta(
        id=row.id,
        cik=row.cik,
        ticker=row.ticker,
        company_name=row.company_name,
        form_type=row.form_type,
        filed_at=row.filed_at,
        period_of_report=row.period_of_report,
        primary_document_url=row.primary_document_url,
        source_tier=cast(SourceTier, row.source_tier),
    )


def _to_chunk_record(row: FilingChunkRow) -> FilingChunkRecord:
    return FilingChunkRecord(
        chunk_id=row.chunk_id,
        filing_id=row.filing_id,
        item=row.item,
        section_path=row.section_path,
        chunk_index=row.chunk_index,
        chunk_text=row.chunk_text,
        char_count=row.char_count,
        embedding_model=row.embedding_model,
    )


class FilingsRepository:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def upsert_filing(self, meta: FilingMeta, content_sha256: str) -> str:
        async with self._session_factory() as session:
            existing = await session.get(FilingRow, meta.id)
            if existing is not None:
                existing.cik = meta.cik
                existing.ticker = meta.ticker
                existing.company_name = meta.company_name
                existing.form_type = meta.form_type
                existing.filed_at = meta.filed_at
                existing.period_of_report = meta.period_of_report
                existing.primary_document_url = meta.primary_document_url
                existing.source_tier = meta.source_tier
                existing.content_sha256 = content_sha256
            else:
                session.add(
                    FilingRow(
                        id=meta.id,
                        cik=meta.cik,
                        ticker=meta.ticker,
                        company_name=meta.company_name,
                        form_type=meta.form_type,
                        filed_at=meta.filed_at,
                        period_of_report=meta.period_of_report,
                        primary_document_url=meta.primary_document_url,
                        source_tier=meta.source_tier,
                        content_sha256=content_sha256,
                    )
                )
            await session.commit()
            return meta.id

    async def get_filing(self, filing_id: str) -> FilingMeta | None:
        async with self._session_factory() as session:
            row = await session.get(FilingRow, filing_id)
            return _to_filing_meta(row) if row is not None else None

    async def list_filings(
        self,
        ticker: str,
        form_types: Sequence[str] | None = None,
        since: date | None = None,
        limit: int = 20,
    ) -> list[FilingMeta]:
        async with self._session_factory() as session:
            stmt = select(FilingRow).where(FilingRow.ticker == ticker)
            if form_types:
                stmt = stmt.where(FilingRow.form_type.in_(form_types))
            if since is not None:
                stmt = stmt.where(FilingRow.filed_at >= since)
            stmt = stmt.order_by(FilingRow.filed_at.desc()).limit(limit)
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_filing_meta(row) for row in rows]

    async def replace_chunks(self, filing_id: str, chunks: Sequence[NewChunk]) -> int:
        """Fully replaces the chunk set for `filing_id` -- chunks have no
        natural upsert key across a re-chunk (a chunking-algorithm change
        can shift every boundary), mirroring
        `PortfolioRepository.upsert_holdings`'s idempotent-reingest pattern.
        """
        async with self._session_factory() as session:
            existing_stmt = select(FilingChunkRow).where(FilingChunkRow.filing_id == filing_id)
            existing_rows = (await session.execute(existing_stmt)).scalars().all()
            for row in existing_rows:
                await session.delete(row)
            await session.flush()

            for index, chunk in enumerate(chunks):
                session.add(
                    FilingChunkRow(
                        chunk_id=chunk.chunk_id,
                        filing_id=filing_id,
                        item=chunk.item,
                        section_path=chunk.section_path,
                        chunk_index=index,
                        chunk_text=chunk.text,
                        char_count=len(chunk.text),
                    )
                )
            await session.commit()
            return len(chunks)

    async def get_chunk(self, chunk_id: str) -> FilingChunkRecord | None:
        async with self._session_factory() as session:
            row = await session.get(FilingChunkRow, chunk_id)
            return _to_chunk_record(row) if row is not None else None

    async def get_chunks_missing_embeddings(
        self, model_name: str, limit: int = 100
    ) -> list[FilingChunkRecord]:
        async with self._session_factory() as session:
            stmt = (
                select(FilingChunkRow)
                .where(
                    (FilingChunkRow.embedding_model.is_(None))
                    | (FilingChunkRow.embedding_model != model_name)
                )
                .limit(limit)
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_to_chunk_record(row) for row in rows]

    async def set_embedding(
        self, chunk_id: str, embedding: Sequence[float], model_name: str
    ) -> None:
        async with self._session_factory() as session:
            row = await session.get(FilingChunkRow, chunk_id)
            if row is None:
                raise DataError(f"chunk {chunk_id!r} not found")
            row.embedding = list(embedding)
            row.embedding_model = model_name
            await session.commit()

    async def search_dense(
        self,
        query_embedding: Sequence[float],
        model_name: str,
        *,
        ticker: str | None = None,
        form_types: Sequence[str] | None = None,
        filed_after: date | None = None,
        filed_before: date | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> list[ScoredChunk]:
        """Cosine-distance ANN search via pgvector's `<=>` operator, matching
        the HNSW index's `vector_cosine_ops` opclass. `filed_after`/
        `filed_before` are pushed into the SQL `WHERE` clause -- freshness
        must be a pre-retrieval hard filter, not a post-fusion Python one
        (architecture.md §4.7), or a fresher chunk ranked just outside the
        unfiltered top-N would be silently lost.
        """
        distance = FilingChunkRow.embedding.cosine_distance(list(query_embedding)).label("distance")
        stmt = (
            select(FilingChunkRow, FilingRow, distance)
            .join(FilingRow, FilingChunkRow.filing_id == FilingRow.id)
            .where(
                FilingChunkRow.embedding.is_not(None), FilingChunkRow.embedding_model == model_name
            )
        )
        if ticker is not None:
            stmt = stmt.where(FilingRow.ticker == ticker)
        if form_types:
            stmt = stmt.where(FilingRow.form_type.in_(form_types))
        if filed_after is not None:
            stmt = stmt.where(FilingRow.filed_at >= filed_after)
        if filed_before is not None:
            stmt = stmt.where(FilingRow.filed_at <= filed_before)
        stmt = stmt.order_by(distance).limit(limit)

        async with self._session_factory() as session:
            rows = (await session.execute(stmt)).all()
            return [
                ScoredChunk(
                    chunk=_to_chunk_record(chunk_row),
                    filing=_to_filing_meta(filing_row),
                    score=float(distance_value),
                )
                for chunk_row, filing_row, distance_value in rows
            ]

    async def search_bm25(
        self,
        query_text: str,
        *,
        ticker: str | None = None,
        form_types: Sequence[str] | None = None,
        filed_after: date | None = None,
        filed_before: date | None = None,
        limit: int = _DEFAULT_SEARCH_LIMIT,
    ) -> list[ScoredChunk]:
        """Rank via Postgres `ts_rank` against the migration's generated
        `content_tsv` column -- raw SQL because that column has no ORM
        attribute to build an expression from (see
        `data/models.py::FilingChunk`'s docstring).
        """
        conditions = ["fc.content_tsv @@ plainto_tsquery('english', :query)"]
        params: dict[str, object] = {"query": query_text, "limit": limit}
        if ticker is not None:
            conditions.append("f.ticker = :ticker")
            params["ticker"] = ticker
        if form_types:
            conditions.append("f.form_type = ANY(:form_types)")
            params["form_types"] = list(form_types)
        if filed_after is not None:
            conditions.append("f.filed_at >= :filed_after")
            params["filed_after"] = filed_after
        if filed_before is not None:
            conditions.append("f.filed_at <= :filed_before")
            params["filed_before"] = filed_before

        stmt = text(f"""
            SELECT fc.chunk_id, fc.filing_id, fc.item, fc.section_path, fc.chunk_index,
                   fc.chunk_text, fc.char_count, fc.embedding_model,
                   f.id AS filing_id_full, f.cik, f.ticker, f.company_name, f.form_type,
                   f.filed_at, f.period_of_report, f.primary_document_url, f.source_tier,
                   ts_rank(fc.content_tsv, plainto_tsquery('english', :query)) AS score
            FROM filing_chunks fc
            JOIN filings f ON fc.filing_id = f.id
            WHERE {" AND ".join(conditions)}
            ORDER BY score DESC
            LIMIT :limit
            """)
        async with self._session_factory() as session:
            rows = (await session.execute(stmt, params)).mappings().all()
            return [_row_to_scored_chunk(row) for row in rows]


def _row_to_scored_chunk(row: object) -> ScoredChunk:
    mapping = cast(dict[str, object], row)
    chunk = FilingChunkRecord(
        chunk_id=cast(str, mapping["chunk_id"]),
        filing_id=cast(str, mapping["filing_id"]),
        item=cast(str, mapping["item"]),
        section_path=cast(str, mapping["section_path"]),
        chunk_index=cast(int, mapping["chunk_index"]),
        chunk_text=cast(str, mapping["chunk_text"]),
        char_count=cast(int, mapping["char_count"]),
        embedding_model=cast("str | None", mapping["embedding_model"]),
    )
    filing = FilingMeta(
        id=cast(str, mapping["filing_id_full"]),
        cik=cast(str, mapping["cik"]),
        ticker=cast(str, mapping["ticker"]),
        company_name=cast(str, mapping["company_name"]),
        form_type=cast(str, mapping["form_type"]),
        filed_at=cast(date, mapping["filed_at"]),
        period_of_report=cast("date | None", mapping["period_of_report"]),
        primary_document_url=cast(str, mapping["primary_document_url"]),
        source_tier=cast(SourceTier, mapping["source_tier"]),
    )
    return ScoredChunk(chunk=chunk, filing=filing, score=float(cast(float, mapping["score"])))
