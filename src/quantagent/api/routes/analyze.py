"""api/routes/analyze.py -- `POST /v1/analyze`, the SSE streaming endpoint
(architecture.md §4.1).

SSE framing is hand-rolled over `StreamingResponse`, not `sse-starlette`:
the wire format needed here (six single-line JSON `data:` frames, no
reconnect semantics) doesn't need what that package solves, and it isn't a
pinned dependency.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from quantagent.agent.events import FinalEvent, LoopEvent
from quantagent.agent.loop import build_unrecoverable_error_answer, run_agent_loop
from quantagent.api.deps import AppResources, get_app_resources

router = APIRouter(prefix="/v1", tags=["analyze"])
logger = structlog.get_logger(__name__)


class AnalyzeRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    portfolio_id: str | None = None


def _format_sse(event: LoopEvent) -> bytes:
    """`event: <name>\\ndata: <compact json>\\n\\n`. `model_dump_json()` is
    compact (no embedded newlines) by default, so no multi-line `data:`
    framing logic is needed.
    """
    return f"event: {event.event}\ndata: {event.model_dump_json()}\n\n".encode()


async def _stream(
    request: AnalyzeRequest, *, tenant_id: str, resources: AppResources, trace_id: str
) -> AsyncIterator[bytes]:
    structlog.contextvars.bind_contextvars(trace_id=trace_id, tenant_id=tenant_id)
    try:
        async for event in run_agent_loop(
            request.question,
            tenant_id=tenant_id,
            portfolio_id=request.portfolio_id,
            ctx=resources.tool_context(tenant_id),
            client=resources.anthropic_client,
            prompts=resources.prompt_loader,
            trace_id=trace_id,
        ):
            yield _format_sse(event)
    except Exception:
        # Defense in depth only: run_agent_loop already has its own
        # fail-closed net. This one covers failures on this side of the
        # call, e.g. resources.tool_context(...) raising before
        # run_agent_loop's generator even starts.
        logger.exception("analyze_stream_unhandled_exception", trace_id=trace_id)
        yield _format_sse(FinalEvent(answer=build_unrecoverable_error_answer(trace_id)))
    finally:
        structlog.contextvars.unbind_contextvars("trace_id", "tenant_id")


@router.post("/analyze")
async def analyze(
    request: AnalyzeRequest,
    x_tenant_id: str | None = Header(None, alias="X-Tenant-Id"),
    resources: AppResources = Depends(get_app_resources),
) -> StreamingResponse:
    """Streams `plan|tool_start|tool_done|draft|verdict|final` SSE events.

    `X-Tenant-Id` is a stopgap: architecture.md §4.1 assigns auth/tenancy to
    the API/Session layer, but that middleware still doesn't exist -- M5's
    guardrails (content-policy checks: scope/PII/injection/prohibited
    content/disclosures) are a separate concern from request authentication
    and don't touch this. This header is NOT cryptographically verified yet
    -- treat it as trusted-caller-only until real auth middleware is built.
    Optional-then-manually-checked (rather than `Header(...)`'s
    required-and-422) so a missing header surfaces as a deliberate `400`,
    matching this endpoint's own contract rather than FastAPI's generic
    validation-error shape.
    """
    if x_tenant_id is None or not x_tenant_id.strip():
        raise HTTPException(status_code=400, detail="X-Tenant-Id header is required")
    trace_id = f"tr_{uuid4().hex[:10]}"
    return StreamingResponse(
        _stream(request, tenant_id=x_tenant_id, resources=resources, trace_id=trace_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
