"""rag/ingest.py -- orchestrates EDGAR fetch -> chunk -> store, and a
separate embedding backfill pass (architecture.md §4.7).

The one file where `rag/`'s two allowed imports (`contracts`, `data`) are
both exercised together -- `.importlinter`'s `rag-scope` contract is the
guard that this stays true; `rag/chunk.py` itself stays pure and knows
nothing about `data/`.

Ingestion and embedding are deliberately separate passes: ingestion can run
without a loaded embedding model, and a future embedding-model swap can
re-embed every chunk without re-fetching or re-chunking a single filing.
"""

from __future__ import annotations

import hashlib
from datetime import date

import structlog

from quantagent.data.providers.edgar import FilingsProvider
from quantagent.data.providers.embeddings import EmbeddingProvider
from quantagent.data.repositories.filings_repository import FilingMeta, FilingsRepository, NewChunk
from quantagent.rag.chunk import FilingIdentity, chunk_filing

logger = structlog.get_logger(__name__)

EMBEDDING_BACKFILL_BATCH_SIZE = 64


async def ingest_filings(
    ticker: str,
    form_types: list[str],
    *,
    provider: FilingsProvider,
    repository: FilingsRepository,
    since: date | None = None,
    limit: int = 20,
) -> list[str]:
    """Fetches, chunks and stores `ticker`'s recent filings. Returns the
    filing_ids written, newest first (the order `list_filings` returns).
    """
    refs = await provider.list_filings(ticker, form_types, since=since, limit=limit)
    filing_ids: list[str] = []
    for ref in refs:
        document = await provider.fetch_primary_document(ref)
        content_sha256 = hashlib.sha256(document.html.encode("utf-8")).hexdigest()
        meta = FilingMeta(
            id=ref.accession_no,
            cik=ref.cik,
            ticker=ticker,
            company_name=ref.company_name,
            form_type=ref.form_type,
            filed_at=ref.filed_at,
            period_of_report=ref.period_of_report,
            primary_document_url=ref.source_url,
            source_tier="T1",
        )
        filing_id = await repository.upsert_filing(meta, content_sha256)

        identity = FilingIdentity(
            ticker=ticker,
            cik=ref.cik,
            accession_no=ref.accession_no,
            form_type=ref.form_type,
            filed_at=ref.filed_at,
            period_of_report=ref.period_of_report,
        )
        chunks = chunk_filing(document.html, identity)
        new_chunks = [
            NewChunk(
                chunk_id=chunk.metadata.chunk_id,
                item=chunk.metadata.item,
                section_path=chunk.metadata.section_path,
                text=chunk.text,
            )
            for chunk in chunks
        ]
        written = await repository.replace_chunks(filing_id, new_chunks)
        logger.info("filing_ingested", ticker=ticker, filing_id=filing_id, chunks_written=written)
        filing_ids.append(filing_id)
    return filing_ids


async def backfill_embeddings(
    *,
    repository: FilingsRepository,
    embeddings: EmbeddingProvider,
    model_name: str,
    batch_size: int = EMBEDDING_BACKFILL_BATCH_SIZE,
) -> int:
    """Embeds every chunk missing an up-to-date `embedding_model`. Loops
    until nothing is left rather than a single page -- a fixed one-page
    call would silently leave the tail of a large backlog unembedded.
    """
    total_embedded = 0
    while True:
        pending = await repository.get_chunks_missing_embeddings(model_name, limit=batch_size)
        if not pending:
            break
        vectors = await embeddings.embed_documents([chunk.chunk_text for chunk in pending])
        for chunk, vector in zip(pending, vectors, strict=True):
            await repository.set_embedding(chunk.chunk_id, vector, model_name)
        total_embedded += len(pending)
        logger.info("embeddings_backfilled", batch_size=len(pending), total=total_embedded)
    return total_embedded
