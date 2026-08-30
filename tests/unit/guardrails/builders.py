"""tests/unit/guardrails/builders.py -- shared test builders for guardrails/ tests."""

from __future__ import annotations

from quantagent.guardrails.types import GuardrailContext, InboundPayload, OutboundPayload
from tests.unit.verify.builders import build_answer


def build_context(**overrides: object) -> GuardrailContext:
    defaults: dict[str, object] = dict(trace_id="tr_1", tenant_id="tenant_1")
    defaults.update(overrides)
    return GuardrailContext(**defaults)  # type: ignore[arg-type]


def build_inbound_payload(
    text: str = "What is my portfolio's VaR?", **overrides: object
) -> InboundPayload:
    defaults: dict[str, object] = dict(text=text, portfolio_id=None)
    defaults.update(overrides)
    return InboundPayload(**defaults)  # type: ignore[arg-type]


def build_outbound_payload(**answer_overrides: object) -> OutboundPayload:
    return OutboundPayload(answer=build_answer(**answer_overrides))
