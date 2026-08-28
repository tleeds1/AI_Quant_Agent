from __future__ import annotations

from quantagent.agent.synthesizer import (
    PROMPT_STAGE,
    TOOL_DEGRADED_CONFIDENCE_CAP,
    SynthesisInput,
    _apply_confidence_calibration,
    _DraftAnswer,
    _ledger_rows_for_prompt,
    _round_for_display,
    synthesize_answer,
)
from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.llm.prompts import PromptLoader
from tests.unit.llm.fixtures import build_mock_anthropic, tool_use_response

_OUTPUT_TOOL = "emit_structured_output"


def test_draft_schema_matches_answer_minus_verification() -> None:
    assert set(_DraftAnswer.model_fields) == set(AgentAnswer.model_fields) - {"verification"}


def _valid_draft_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "trace_id": "tr_1",
        "scope": "PORTFOLIO",
        "decision": "HOLD",
        "confidence": 0.8,
        "confidence_basis": ["ledger fully populated"],
        "risk_level": "MEDIUM",
        "horizon": "n/a",
        "summary": "Holdings are concentrated in tech.",
        "claims": [
            {
                "claim_id": "c1",
                "text": "Portfolio VaR is 2.5%.",
                "claim_type": "numeric",
                "evidence_ids": ["ev1"],
                "hedge": "none",
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev1",
                "kind": "metric",
                "ref": "portfolio_var",
                "excerpt": None,
                "char_span": None,
                "source_title": "calculate_portfolio_var",
                "source_url": None,
                "source_tier": None,
                "published_at": None,
                "retrieval_score": None,
            }
        ],
        "quant_metrics": {},
        "constraints_checked": [],
        "limitations": ["Based on 504-day lookback."],
        "disclosures": ["Analysis only, not investment advice."],
    }
    payload.update(overrides)
    return payload


def _ledger() -> Ledger:
    return Ledger(
        trace_id="tr_1",
        calls=[
            ToolCallRecord(
                call_id="tc_1",
                tool_name="calculate_portfolio_var",
                args={"portfolio_id": "pf_1"},
                args_hash="h1",
                status="OK",
                latency_ms=100,
                cost_usd=0.0,
                result={"metric_id": "portfolio_var", "value": 0.025000123, "unit": "pct"},
                error=None,
            )
        ],
        numeric_index={"tc_1.result": 0.025000123},
    )


async def test_synthesize_answer_happy_path() -> None:
    client, session = build_mock_anthropic(
        [tool_use_response(_OUTPUT_TOOL, _valid_draft_payload())]
    )
    inp = SynthesisInput(question="what's my VaR?", trace_id="tr_1", ledger=_ledger())

    answer, meta = await synthesize_answer(inp, client=client, prompts=PromptLoader())

    assert answer.trace_id == "tr_1"
    assert answer.verification.verdict == "PASS"
    assert (
        answer.verification.checks == 0
    )  # placeholder, overwritten by the loop's real VERIFY call
    assert meta.prompt_version == f"{PROMPT_STAGE}/answer.v1"
    assert session.call_count == 1


async def test_synthesize_answer_retries_on_validation_failure() -> None:
    bad_payload = _valid_draft_payload(confidence="not-a-float")
    client, session = build_mock_anthropic(
        [
            tool_use_response(_OUTPUT_TOOL, bad_payload),
            tool_use_response(_OUTPUT_TOOL, _valid_draft_payload()),
        ]
    )
    inp = SynthesisInput(question="what's my VaR?", trace_id="tr_1", ledger=_ledger())

    answer, meta = await synthesize_answer(inp, client=client, prompts=PromptLoader())

    assert answer.trace_id == "tr_1"
    assert meta.retried is True
    assert session.call_count == 2


def test_confidence_calibration_passthrough_when_not_degraded() -> None:
    confidence, basis, limitations = _apply_confidence_calibration(
        0.9, ["x"], ["y"], degraded=False
    )
    assert confidence == 0.9
    assert basis == ["x"]
    assert limitations == ["y"]


def test_confidence_calibration_caps_when_degraded() -> None:
    confidence, basis, limitations = _apply_confidence_calibration(0.9, ["x"], ["y"], degraded=True)
    assert confidence == TOOL_DEGRADED_CONFIDENCE_CAP
    assert len(basis) == 2
    assert len(limitations) == 2


def test_confidence_calibration_does_not_raise_an_already_low_confidence() -> None:
    confidence, basis, limitations = _apply_confidence_calibration(0.4, ["x"], ["y"], degraded=True)
    assert confidence == 0.4
    assert basis == ["x"]
    assert limitations == ["y"]


def test_round_for_display_rounds_nested_floats() -> None:
    rounded = _round_for_display({"a": 0.123456789, "b": [1.0000001, {"c": 2.99999999}]})
    assert rounded == {"a": 0.1235, "b": [1.0, {"c": 3.0}]}


def test_ledger_rows_for_prompt_rounds_and_handles_none_result() -> None:
    ledger = _ledger()
    rows = _ledger_rows_for_prompt(ledger)
    assert rows[0]["result"]["value"] == 0.025

    ledger_with_error = Ledger(
        trace_id="tr_1",
        calls=[
            ToolCallRecord(
                call_id="tc_2",
                tool_name="get_prices",
                args={},
                args_hash="h2",
                status="ERROR",
                latency_ms=0,
                cost_usd=0.0,
                result=None,
                error="timeout",
            )
        ],
        numeric_index={},
    )
    rows2 = _ledger_rows_for_prompt(ledger_with_error)
    assert rows2[0]["result"] is None
    assert rows2[0]["error"] == "timeout"
