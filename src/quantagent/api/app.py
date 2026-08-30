from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from anthropic import AsyncAnthropic
from fastapi import FastAPI
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from quantagent.api.deps import AppResources
from quantagent.api.routes.analyze import router as analyze_router
from quantagent.api.routes.health import router as health_router
from quantagent.config import settings
from quantagent.data.cache import CacheClient
from quantagent.data.providers.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from quantagent.data.providers.reranker import SentenceTransformersRerankerProvider
from quantagent.data.repositories.filings_repository import FilingsRepository
from quantagent.llm.prompts import PromptLoader
from quantagent.obs.logging import configure_logging
from quantagent.obs.tracing import configure_tracing
from quantagent.rag.retrieval import HybridRetriever


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    configure_tracing()
    engine = create_async_engine(settings.database_url)
    cache = CacheClient.from_settings()
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    # Constructing an engine/cache/Anthropic client/retriever here does no
    # I/O by itself (all are lazy, including the embedding/reranker models
    # HybridRetriever holds -- see their own classes' docstrings) -- safe
    # even with an empty dev ANTHROPIC_API_KEY or a Postgres/Redis that
    # isn't running yet; it only matters once a request actually needs one.
    app.state.resources = AppResources(
        engine=engine,
        session_factory=session_factory,
        cache=cache,
        anthropic_client=AsyncAnthropic(api_key=settings.anthropic_api_key),
        prompt_loader=PromptLoader(),
        retrieval=HybridRetriever(
            repository=FilingsRepository(session_factory),
            embeddings=SentenceTransformerEmbeddingProvider(),
            reranker=SentenceTransformersRerankerProvider(),
            embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        ),
    )
    try:
        yield
    finally:
        await cache.close()
        await engine.dispose()


def create_app() -> FastAPI:
    """Build the FastAPI application (architecture.md §4.1)."""
    app = FastAPI(title="AI Quant / Financial Agent", lifespan=lifespan)
    app.include_router(health_router)
    app.include_router(analyze_router)
    return app


app = create_app()
