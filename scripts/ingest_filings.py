"""CLI: `make ingest-filings TICKER=NVDA` -- fetches, chunks and stores a
ticker's recent SEC EDGAR filings, then backfills chunk embeddings
(architecture.md §4.7). Mirrors scripts/seed_portfolio.py's
engine-per-run, dispose-at-end shape.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from quantagent.config import settings
from quantagent.data.cache import CacheClient
from quantagent.data.providers.edgar import EdgarFilingsProvider
from quantagent.data.providers.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from quantagent.data.repositories.filings_repository import FilingsRepository
from quantagent.rag.ingest import backfill_embeddings, ingest_filings

DEFAULT_FORM_TYPES = ["10-K", "10-Q", "8-K"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest SEC EDGAR filings for a ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--forms", default=",".join(DEFAULT_FORM_TYPES))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--skip-embeddings", action="store_true")
    return parser.parse_args()


async def main() -> None:
    args = _parse_args()
    form_types = [form.strip() for form in args.forms.split(",") if form.strip()]

    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    repository = FilingsRepository(session_factory)
    cache = CacheClient.from_settings()
    provider = EdgarFilingsProvider(cache=cache)

    filing_ids = await ingest_filings(
        args.ticker, form_types, provider=provider, repository=repository, limit=args.limit
    )
    print(f"Ingested {len(filing_ids)} filing(s) for {args.ticker}: {filing_ids}")

    if not args.skip_embeddings:
        embeddings = SentenceTransformerEmbeddingProvider()
        total = await backfill_embeddings(
            repository=repository, embeddings=embeddings, model_name=DEFAULT_EMBEDDING_MODEL
        )
        print(f"Backfilled embeddings for {total} chunk(s).")

    await cache.close()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
