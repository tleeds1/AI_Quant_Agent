"""tests/unit/guardrails/test_fail_closed.py -- I4: an exception inside a
check must yield BLOCK, never propagate.
"""

from __future__ import annotations

from quantagent.guardrails.fail_closed import run_check_safely
from quantagent.guardrails.inbound import run_inbound_checks
from quantagent.guardrails.outbound import run_outbound_checks
from tests.unit.guardrails.builders import (
    build_context,
    build_inbound_payload,
    build_outbound_payload,
)


def _raising_check(payload: object, context: object) -> None:
    raise RuntimeError("boom")


def test_run_check_safely_converts_exception_to_block() -> None:
    decision = run_check_safely(_raising_check, build_inbound_payload(), build_context())
    assert decision.action == "BLOCK"
    assert decision.category == "internal_error"
    assert "RuntimeError" in decision.reason


def test_run_inbound_checks_fails_closed_on_raising_check() -> None:
    decision, _payload = run_inbound_checks(
        build_inbound_payload(), build_context(), checks=(_raising_check,)
    )
    assert decision.action == "BLOCK"
    assert decision.category == "internal_error"


def test_run_outbound_checks_fails_closed_on_raising_check() -> None:
    decision = run_outbound_checks(
        build_outbound_payload(), build_context(), checks=(_raising_check,)
    )
    assert decision.action == "BLOCK"
    assert decision.category == "internal_error"
