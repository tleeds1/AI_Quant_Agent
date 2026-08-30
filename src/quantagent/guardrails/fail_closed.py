"""guardrails/fail_closed.py -- I4 ("guardrails and verifiers fail closed";
guideline.md §1, §9: "wrap every check so an internal exception yields
BLOCK, not a pass-through").
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from quantagent.guardrails.types import GuardrailContext, GuardrailDecision

_PayloadT = TypeVar("_PayloadT")


def run_check_safely(
    check: Callable[[_PayloadT, GuardrailContext], GuardrailDecision],
    payload: _PayloadT,
    context: GuardrailContext,
) -> GuardrailDecision:
    """The one place I4 is implemented. A check's own detection logic
    should never raise (mirrors verify/constraint_rules.py's predicates,
    which always return a verdict); this net exists purely for bugs -- a
    regex construction error, a KeyError -- not for a check's own intended
    block decision.
    """
    check_id = getattr(check, "__name__", "unknown_check")
    try:
        return check(payload, context)
    except Exception as exc:
        return GuardrailDecision(
            check_id=check_id,
            action="BLOCK",
            category="internal_error",
            reason=f"guardrail check raised {exc.__class__.__name__}; failing closed (I4).",
        )
