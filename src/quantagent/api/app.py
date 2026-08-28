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
from quantagent.llm.prompts import PromptLoader
from quantagent.obs.logging import configure_logging
from quantagent.obs.tracing import configure_tracing


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    configure_logging()
    configure_tracing()
    engine = create_async_engine(settings.database_url)
    cache = CacheClient.from_settings()
    # Constructing an engine/cache/Anthropic client here does no I/O by
    # itself (all three are lazy) -- safe even with an empty dev
    # ANTHROPIC_API_KEY or a Postgres/Redis that isn't running yet; it only
    # matters once a request actually needs one of them.
    app.state.resources = AppResources(
        engine=engine,
        session_factory=async_sessionmaker(engine, expire_on_commit=False),
        cache=cache,
        anthropic_client=AsyncAnthropic(api_key=settings.anthropic_api_key),
        prompt_loader=PromptLoader(),
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
