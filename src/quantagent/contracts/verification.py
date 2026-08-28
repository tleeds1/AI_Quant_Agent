from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ConstraintStatus = Literal["PASS", "BREACH", "NOT_APPLICABLE", "UNKNOWN"]
VerificationVerdict = Literal["PASS", "PASS_WITH_WARNINGS", "FAIL"]


class ConstraintCheck(BaseModel):
    """Result of one mandate/policy rule evaluated against a draft answer.

    See architecture.md §5.3 and the rules engine in §7.5.
    """

    rule_id: str
    description: str
    status: ConstraintStatus
    observed: float | None = None
    limit: float | None = None


class VerificationReport(BaseModel):
    """Aggregate verdict of the V1-V5 verifier pipeline (architecture.md §7.7).

    Not enumerated as its own model in architecture.md §5, but required there
    via `AgentAnswer.verification`; shape taken from §7.7's verdict values and
    the §16 worked example's `verification` payload. Per-check detail
    (`CheckResult` etc.) lives in `verify/` starting M4 — this is only the
    summary that crosses into `contracts/`.
    """

    verdict: VerificationVerdict
    checks: int
    warnings: int
    repair_attempts: int
