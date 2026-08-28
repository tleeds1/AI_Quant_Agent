from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

EvidenceKind = Literal["metric", "filing", "news", "transaction", "market_data"]
SourceTier = Literal["T1", "T2", "T3", "T4"]
ClaimType = Literal["numeric", "factual", "causal", "forward_looking"]
Hedge = Literal["none", "may", "likely", "uncertain"]


class Evidence(BaseModel):
    """A single grounding reference for a `Claim` (architecture.md §5.2).

    `excerpt`, when set, must appear verbatim in the source at `char_span` —
    enforced by the verifier's citation-validity layer (architecture.md §7.4),
    not by this model itself.
    """

    evidence_id: str
    kind: EvidenceKind
    ref: str
    excerpt: str | None = Field(default=None, max_length=300)
    char_span: tuple[int, int] | None
    source_title: str
    source_url: str | None
    source_tier: SourceTier | None
    published_at: datetime | None
    retrieval_score: float | None


class Claim(BaseModel):
    """One evidence-linked assertion inside an `AgentAnswer.summary` (architecture.md §5.2).

    `claim_type` drives verification treatment: a `causal` claim requires
    stronger evidence than a `factual` one (architecture.md §3.2, §7.5 R-006).
    """

    claim_id: str
    text: str
    claim_type: ClaimType
    evidence_ids: list[str] = Field(min_length=1)
    hedge: Hedge = "none"
