"""MCP server exposing every registered tool to any MCP client
(docs/guideline.md §11, M2 DoD).

A thin, dynamic wrapper over `tools.registry`: `on_list_tools` maps every
`ToolSpec` to an MCP `Tool` (using its Pydantic-generated JSON schema,
never hand-written); `on_call_tool` validates+invokes via
`registry.invoke(...)` and returns the result as `TextContent`.

One `ToolContext` is built per server process against a fixed demo tenant
(matches `scripts/seed_portfolio.py`'s seeded data) -- a real multi-tenant
session story is API/auth-layer scope, not this milestone's.

    uv run python -m quantagent.tools.mcp_server
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

import mcp.types as types
from mcp.server import Server, ServerRequestContext
from mcp.server.stdio import stdio_server
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import quantagent.tools  # noqa: F401 -- import side effect: populates the registry
from quantagent.config import settings
from quantagent.contracts.errors import ToolValidationError
from quantagent.data.cache import CacheClient
from quantagent.data.providers.embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    SentenceTransformerEmbeddingProvider,
)
from quantagent.data.providers.factors import KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.providers.reranker import SentenceTransformersRerankerProvider
from quantagent.data.repositories.filings_repository import FilingsRepository
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.rag.retrieval import HybridRetriever
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry

DEFAULT_TENANT_ID = "tenant_demo"

CallToolHandler = Callable[
    [ServerRequestContext, types.CallToolRequestParams], Awaitable[types.CallToolResult]
]


def build_tool_context(*, tenant_id: str = DEFAULT_TENANT_ID) -> ToolContext:
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    cache = CacheClient.from_settings()
    retrieval = HybridRetriever(
        repository=FilingsRepository(session_factory),
        embeddings=SentenceTransformerEmbeddingProvider(),
        reranker=SentenceTransformersRerankerProvider(),
        embedding_model_name=DEFAULT_EMBEDDING_MODEL,
    )
    return ToolContext(
        tenant_id=tenant_id,
        portfolios=PortfolioRepository(session_factory),
        prices=YFinancePriceProvider(cache=cache),
        fundamentals=YFinanceFundamentalsProvider(cache=cache),
        factors=KenFrenchFactorDataProvider(cache=cache),
        cache=cache,
        retrieval=retrieval,
    )


async def handle_list_tools(
    ctx: ServerRequestContext, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    tools = [
        types.Tool(name=spec.name, description=spec.description, input_schema=spec.json_schema())
        for spec in registry.list_tools()
    ]
    return types.ListToolsResult(tools=tools)


def build_call_tool_handler(tool_ctx: ToolContext) -> CallToolHandler:
    async def handle_call_tool(
        ctx: ServerRequestContext, params: types.CallToolRequestParams
    ) -> types.CallToolResult:
        try:
            result = await registry.invoke(params.name, dict(params.arguments or {}), tool_ctx)
        except ToolValidationError as exc:
            return types.CallToolResult(
                content=[types.TextContent(type="text", text=str(exc))], is_error=True
            )
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=result.model_dump_json())]
        )

    return handle_call_tool


def build_server(tool_ctx: ToolContext) -> Server:
    return Server(
        name="quantagent",
        version="0.1.0",
        description="Governed financial analysis tools: portfolio, market, exposure, risk.",
        on_list_tools=handle_list_tools,
        on_call_tool=build_call_tool_handler(tool_ctx),
    )


async def main() -> None:
    tool_ctx = build_tool_context()
    server = build_server(tool_ctx)
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
