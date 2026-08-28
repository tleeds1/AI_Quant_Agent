"""api/deps.py -- process-lifetime resources for the API layer.

Mirrors `tools/mcp_server.py::build_tool_context`'s resource-assembly
pattern, but built once in `api/app.py`'s lifespan and shared across
requests via `app.state`, instead of rebuilt per process entrypoint.
"""

from __future__ import annotations

from dataclasses import dataclass

from anthropic import AsyncAnthropic
from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from quantagent.data.cache import CacheClient
from quantagent.data.providers.factors import KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.context import ToolContext


@dataclass(frozen=True, slots=True)
class AppResources:
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    cache: CacheClient
    anthropic_client: AsyncAnthropic
    prompt_loader: PromptLoader

    def tool_context(self, tenant_id: str) -> ToolContext:
        return ToolContext(
            tenant_id=tenant_id,
            portfolios=PortfolioRepository(self.session_factory),
            prices=YFinancePriceProvider(cache=self.cache),
            fundamentals=YFinanceFundamentalsProvider(cache=self.cache),
            factors=KenFrenchFactorDataProvider(cache=self.cache),
            cache=self.cache,
        )


def get_app_resources(request: Request) -> AppResources:
    resources = getattr(request.app.state, "resources", None)
    if not isinstance(resources, AppResources):
        raise RuntimeError("AppResources not initialised -- is create_app()'s lifespan running?")
    return resources
