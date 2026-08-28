from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel

ToolCallStatus = Literal["OK", "ERROR", "TIMEOUT", "DEGRADED", "CACHED"]


class ToolCallRecord(BaseModel):
    """One executed step of the plan DAG, appended to the `Ledger` (architecture.md §5.4)."""

    call_id: str
    tool_name: str
    args: dict[str, Any]
    args_hash: str
    status: ToolCallStatus
    latency_ms: int
    cost_usd: float
    result: dict[str, Any] | None
    error: str | None


class Ledger(BaseModel):
    """Append-only record of every tool call for one request; the sole source of
    truth for synthesis and verification (architecture.md §4.2, §5.4).

    `numeric_index` flattens every numeric leaf of every successful tool
    result (e.g. `"tc_07.result.var_95" -> value`) and is the ground truth the
    numeric-grounding verifier matches free-text numbers against (§7.3).
    """

    trace_id: str
    calls: list[ToolCallRecord]
    numeric_index: dict[str, float]
