"""agent/events.py -- the six SSE event payloads (architecture.md §4.1).

Lives in `agent/`, not `api/`, because `loop.py` is the producer; `api/`
only serializes what it's handed (layering: api -> agent, never the
reverse). One tagged union so both sides get exhaustiveness checking from
mypy on the `event` discriminator.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field

from quantagent.contracts.answer import AgentAnswer


class PlanEvent(BaseModel):
    event: Literal["plan"] = "plan"
    steps: list[dict[str, object]]


class ToolStartEvent(BaseModel):
    event: Literal["tool_start"] = "tool_start"
    call_id: str
    tool: str


class ToolDoneEvent(BaseModel):
    event: Literal["tool_done"] = "tool_done"
    call_id: str
    latency_ms: int
    status: str


class DraftEvent(BaseModel):
    event: Literal["draft"] = "draft"
    answer: AgentAnswer


class VerdictEvent(BaseModel):
    event: Literal["verdict"] = "verdict"
    verdict: str
    warnings: int
    repair_attempts: int


class FinalEvent(BaseModel):
    event: Literal["final"] = "final"
    answer: AgentAnswer


LoopEvent = Annotated[
    PlanEvent | ToolStartEvent | ToolDoneEvent | DraftEvent | VerdictEvent | FinalEvent,
    Field(discriminator="event"),
]
