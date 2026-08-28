from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from quantagent.contracts.provenance import Provenance

MetricUnit = Literal["ratio", "pct", "usd", "bps", "count", "zscore", "days"]


class MetricValue(BaseModel):
    """The only legal carrier of a number into an `AgentAnswer` (architecture.md §5.1).

    A bare `float` anywhere in agent-facing output is a schema violation.
    """

    metric_id: str
    value: float
    unit: MetricUnit
    method: str
    window: str | None = None
    ci_95: tuple[float, float] | None = None
    provenance: Provenance
