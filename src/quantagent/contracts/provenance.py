from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Origin and reproducibility metadata carried by every computed value.

    `inputs_hash` is both the cache key and the replay key (architecture.md
    §5.1, §9.2): re-running a tool call with the same hash against the same
    pinned data snapshot must reproduce the value bit-for-bit.
    """

    tool_call_id: str
    tool_name: str
    as_of: date
    computed_at: datetime
    inputs_hash: str
    data_sources: list[str]
    estimator: str | None = None
    sample_size: int | None = None
    seed: int | None = None
    warnings: list[str] = Field(default_factory=list)
