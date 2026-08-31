import asyncio
import sys
from decimal import Decimal
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from quantagent.agent.budget import RequestBudget
from quantagent.agent.executor import execute_plan
from quantagent.agent.planner import Plan
from quantagent.config import settings
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
from quantagent.data.repositories.trace_repository import TraceRepository
from quantagent.rag.retrieval import HybridRetriever
from quantagent.tools.context import ToolContext


async def replay_trace(trace_id: str) -> None:
    print(f"Replaying trace {trace_id}...")

    # Set up engine and repositories
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    cache = CacheClient.from_settings()

    try:
        # Load trace from DB
        # To avoid requiring tenant_id in arguments, we can query trace directly.
        # But Invariant I9 requires tenant_id at repo layer.
        # Let's query using a direct select to find the trace and its tenant_id.
        async with session_factory() as session:
            from sqlalchemy import select
            from quantagent.data.models import Trace as TraceRow
            stmt = select(TraceRow).where(TraceRow.id == trace_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                print(f"Error: Trace {trace_id} not found in database.")
                sys.exit(1)
            tenant_id = row.tenant_id
            saved_plan = row.plan
            saved_ledger = row.ledger

        if not saved_plan:
            print("Error: Saved trace has no plan to execute.")
            sys.exit(1)

        # Re-create ToolContext
        retrieval = HybridRetriever(
            repository=FilingsRepository(session_factory),
            embeddings=SentenceTransformerEmbeddingProvider(),
            reranker=SentenceTransformersRerankerProvider(),
            embedding_model_name=DEFAULT_EMBEDDING_MODEL,
        )
        ctx = ToolContext(
            tenant_id=tenant_id,
            portfolios=PortfolioRepository(session_factory),
            prices=YFinancePriceProvider(cache=cache),
            fundamentals=YFinanceFundamentalsProvider(cache=cache),
            factors=KenFrenchFactorDataProvider(cache=cache),
            cache=cache,
            retrieval=retrieval,
        )

        plan = Plan.model_validate(saved_plan)
        budget = RequestBudget.from_settings()

        # Run execute_plan
        print("Executing plan...")
        execution = await execute_plan(plan, ctx, budget, trace_id)

        # Compare outputs
        saved_calls = {c["call_id"]: c for c in (saved_ledger or {}).get("calls", [])}
        new_calls = {c.call_id: c for c in execution.ledger.calls}

        mismatches = []
        for call_id, saved_call in saved_calls.items():
            if saved_call["status"] != "OK":
                continue
            
            if call_id not in new_calls:
                mismatches.append(f"Missing call: {call_id} ({saved_call['tool_name']}) did not run in replay.")
                continue
            
            new_call = new_calls[call_id]
            if new_call.status != "OK":
                mismatches.append(f"Call {call_id} failed in replay: status={new_call.status}, error={new_call.error}")
                continue

            # Compare result values
            saved_result = saved_call.get("result", {})
            new_result = new_call.result or {}

            # If it's a metric, compare the float values bit-for-bit
            saved_val = saved_result.get("value")
            new_val = new_result.get("value")

            if saved_val is not None or new_val is not None:
                if saved_val != new_val:
                    mismatches.append(
                        f"Metric value mismatch for {call_id} ({saved_call['tool_name']}): "
                        f"Saved={saved_val}, Replayed={new_val}"
                    )

        if mismatches:
            print("\nReplay FAILED with the following mismatches:")
            for m in mismatches:
                print(f"- {m}")
            sys.exit(1)
        else:
            print("\nReplay SUCCESS! All metrics reproduced bit-for-bit.")
            sys.exit(0)

    finally:
        await cache.close()
        await engine.dispose()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/replay_trace.py <trace_id>")
        sys.exit(1)
    asyncio.run(replay_trace(sys.argv[1]))
