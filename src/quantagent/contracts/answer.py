from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from quantagent.contracts.evidence import Claim, Evidence
from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.verification import ConstraintCheck, VerificationReport

Decision = Literal["BUY", "HOLD", "SELL", "REDUCE", "HEDGE", "NO_ACTION", "INSUFFICIENT_EVIDENCE"]
RiskLevel = Literal["LOW", "MEDIUM", "HIGH", "EXTREME"]


class AgentAnswer(BaseModel):
    """The final, released answer to a user query (architecture.md §5.3).

    `limitations` has `min_length=1`: a financial analysis with zero stated
    limitations is itself a red flag, so the schema refuses to represent one.
    """

    trace_id: str
    scope: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_basis: list[str]
    risk_level: RiskLevel
    horizon: str
    summary: str
    claims: list[Claim]
    evidence: list[Evidence]
    quant_metrics: dict[str, MetricValue]
    constraints_checked: list[ConstraintCheck]
    limitations: list[str] = Field(min_length=1)
    disclosures: list[str]
    verification: VerificationReport
