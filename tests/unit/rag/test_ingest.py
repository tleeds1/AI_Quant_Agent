"""tests/unit/rag/test_ingest.py -- exercises the real `FilingsRepository`
against sqlite (same fixture pattern as
tests/unit/data/repositories/test_filings_repository.py), with a fake
`FilingsProvider`/`EmbeddingProvider` standing in for network/model I/O.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from datetime import UTC, date, datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantagent.data.models import Base
from quantagent.data.providers.edgar import FilingDocument, FilingRef
from quantagent.data.repositories.filings_repository import FilingsRepository
from quantagent.rag.ingest import backfill_embeddings, ingest_filings

_REF = FilingRef(
    cik="0001045810",
    accession_no="0001045810-24-000123",
    form_type="10-K",
    filed_at=date(2024, 2, 21),
    period_of_report=date(2024, 1, 28),
    primary_document="nvda-20240128.htm",
    company_name="NVIDIA CORP",
)

_FILING_HTML = (
    "<html><body><p>Item 1A. Risk Factors</p><p>Our supply chain is concentrated.</p></body></html>"
)


class FakeFilingsProvider:
    def __init__(self, refs: Sequence[FilingRef], html: str) -> None:
        self._refs = list(refs)
        self._html = html

    async def resolve_cik(self, ticker: str) -> str:
        return self._refs[0].cik

    async def list_filings(
        self, ticker: str, form_types: list[str], since: date | None = None, limit: int = 20
    ) -> list[FilingRef]:
        return self._refs[:limit]

    async def fetch_primary_document(self, ref: FilingRef) -> FilingDocument:
        return FilingDocument(ref=ref, html=self._html, fetched_at=datetime.now(UTC))


class FakeEmbeddingProvider:
    """Deterministic, hash-based embeddings -- no real model weights loaded,
    matching guideline.md §10.3's "no test touches a live endpoint."
    """

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def _vector(self, text: str) -> list[float]:
        return [float((hash(f"{text}{i}") % 1000) / 1000) for i in range(self._dim)]


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


async def test_ingest_filings_writes_filing_and_chunks(repository: FilingsRepository) -> None:
    provider = FakeFilingsProvider([_REF], _FILING_HTML)

    filing_ids = await ingest_filings("NVDA", ["10-K"], provider=provider, repository=repository)

    assert filing_ids == ["0001045810-24-000123"]
    filing = await repository.get_filing("0001045810-24-000123")
    assert filing is not None
    assert filing.ticker == "NVDA"
    assert filing.source_tier == "T1"

    chunk = await repository.get_chunk("0001045810-24-000123#1A#0000")
    assert chunk is not None
    assert "supply chain" in chunk.chunk_text


async def test_ingest_filings_is_idempotent_on_rerun(repository: FilingsRepository) -> None:
    provider = FakeFilingsProvider([_REF], _FILING_HTML)

    await ingest_filings("NVDA", ["10-K"], provider=provider, repository=repository)
    await ingest_filings("NVDA", ["10-K"], provider=provider, repository=repository)

    filings = await repository.list_filings("NVDA")
    assert len(filings) == 1


async def test_backfill_embeddings_embeds_every_pending_chunk(
    repository: FilingsRepository,
) -> None:
    provider = FakeFilingsProvider([_REF], _FILING_HTML)
    await ingest_filings("NVDA", ["10-K"], provider=provider, repository=repository)

    total = await backfill_embeddings(
        repository=repository, embeddings=FakeEmbeddingProvider(), model_name="fake-model"
    )

    assert total == 1
    assert await repository.get_chunks_missing_embeddings("fake-model") == []


async def test_backfill_embeddings_is_a_noop_when_nothing_pending(
    repository: FilingsRepository,
) -> None:
    total = await backfill_embeddings(
        repository=repository, embeddings=FakeEmbeddingProvider(), model_name="fake-model"
    )
    assert total == 0
