"""guardrails/types.py -- shared types every inbound/outbound check builds
against (architecture.md §8).

Mirrors two existing precedents: `verify/types.py::CheckResult` (a
serializable `BaseModel` per-check result) and
`verify/constraint_rules.py::RuleContext` (a frozen `dataclass` internal
invocation context, never serialized).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantagent.contracts.answer import AgentAnswer

GuardrailAction = Literal["ALLOW", "BLOCK"]

GuardrailCategory = Literal[
    "none",
    "out_of_scope",
    "prohibited_request",
    "pii_detected",
    "injection_detected",
    "rate_limit_exceeded",
    "jurisdiction_blocked",
    "prohibited_language",
    "advice_framing",
    "pii_egress",
    "leakage",
    "internal_error",
]


class GuardrailDecision(BaseModel):
    """One check's outcome. `redacted_text` is set only by the PII check on
    an ALLOW decision -- PII is redact-then-continue, never a refusal
    (architecture.md §8.1), the one case where a non-"none" category
    accompanies ALLOW. `reason`/`metadata` must never carry raw PII or the
    offending span itself (pattern/group ids only), so a decision is always
    safe to log verbatim.
    """

    check_id: str
    action: GuardrailAction
    category: GuardrailCategory
    reason: str = ""
    redacted_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InboundPayload:
    text: str
    portfolio_id: str | None = None


@dataclass(frozen=True, slots=True)
class OutboundPayload:
    answer: AgentAnswer


@dataclass(frozen=True, slots=True)
class GuardrailContext:
    """Per-request context every check reads from. Every gated field
    (`*_limit`, `*_cap`, `tenant_allowed_jurisdictions`) defaults to
    "cannot fire" until a real tenant-config/rate-limit store exists --
    same discipline as verify/constraint_rules.py's R-005/R-009 stubs: the
    check is real and independently testable, just not fed live data yet,
    and that gap is documented rather than hidden (docs/PROGRESS.md).
    """

    trace_id: str
    tenant_id: str
    jurisdiction: str | None = None
    tenant_allowed_jurisdictions: frozenset[str] | None = None
    requests_in_window: int = 0
    rate_limit_per_window: int = 1_000_000
    licensed_for_suitability_advice: bool = False


InboundCheck = Callable[[InboundPayload, GuardrailContext], GuardrailDecision]
OutboundCheck = Callable[[OutboundPayload, GuardrailContext], GuardrailDecision]
