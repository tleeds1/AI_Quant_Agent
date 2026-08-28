from __future__ import annotations

from unittest.mock import AsyncMock

import quantagent.agent.loop as loop_module
from quantagent.agent.events import FinalEvent, PlanEvent, VerdictEvent
from quantagent.agent.executor import ExecutionResult
from quantagent.agent.intent import IntentResult
from quantagent.agent.loop import MandateContext, _classify_degradation, run_agent_loop
from quantagent.agent.planner import Plan, PlanStep
from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import Claim, Evidence
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.contracts.verification import VerificationReport
from quantagent.llm.client import LLMCallMetadata
from quantagent.verify.types import CheckResult

_META = LLMCallMetadata(
    model="m",
    prompt_version="v",
    input_tokens=1,
    output_tokens=1,
    latency_ms=1,
    cost_usd=0.0,
    retried=False,
)


def _plan(n_steps: int = 1) -> Plan:
    steps = [
        PlanStep(id=f"s{i}", tool="get_holdings", args={}, depends_on=[]) for i in range(n_steps)
    ]
    return Plan(steps=steps, success_criteria="test")


def _ledger(statuses: list[str] | None = None) -> Ledger:
    statuses = statuses or ["OK"]
    calls = [
        ToolCallRecord(
            call_id=f"tc_{i}",
            tool_name="get_holdings",
            args={},
            args_hash="h",
            status=status,  # type: ignore[arg-type]
            latency_ms=10,
            cost_usd=0.0,
            result={"ok": True} if status in ("OK", "CACHED") else None,
            error=None if status in ("OK", "CACHED") else "boom",
        )
        for i, status in enumerate(statuses)
    ]
    return Ledger(trace_id="tr_1", calls=calls, numeric_index={})


def _answer(*, confidence: float = 0.7) -> AgentAnswer:
    return AgentAnswer(
        trace_id="tr_1",
        scope="PORTFOLIO",
        decision="HOLD",
        confidence=confidence,
        confidence_basis=[],
        risk_level="MEDIUM",
        horizon="n/a",
        summary="s",
        claims=[Claim(claim_id="c1", text="x", claim_type="factual", evidence_ids=["ev1"])],
        evidence=[
            Evidence(
                evidence_id="ev1",
                kind="metric",
                ref="m1",
                excerpt=None,
                char_span=None,
                source_title="get_holdings",
                source_url=None,
                source_tier=None,
                published_at=None,
                retrieval_score=None,
            )
        ],
        quant_metrics={},
        constraints_checked=[],
        limitations=["none"],
        disclosures=[],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


def _verification(verdict: str = "PASS", *, repair_attempts: int = 0) -> VerificationReport:
    return VerificationReport(
        verdict=verdict,  # type: ignore[arg-type]
        checks=1,
        warnings=0,
        repair_attempts=repair_attempts,
    )


def _mock_run_verification(*results_and_verdicts: tuple[str, list[CheckResult]]) -> AsyncMock:
    """Builds an AsyncMock for `loop_module.run_verification` with one
    `(answer, VerificationReport(verdict=...), check_results)` return per
    call, in order -- mirrors the real function's `(answer, report,
    results)` tuple shape without invoking the real V1-V5 pipeline (that
    pipeline has its own dedicated tests in tests/unit/verify/).
    """
    side_effects = [
        (_answer(), _verification(verdict), results) for verdict, results in results_and_verdicts
    ]
    return AsyncMock(side_effect=side_effects)


async def _drain(agen: object) -> list[object]:
    return [e async for e in agen]  # type: ignore[union-attr]


def _fresh_ctx() -> object:
    return object()  # run_agent_loop's ctx is only ever forwarded, never introspected here


def _no_mandate() -> AsyncMock:
    return AsyncMock(return_value=MandateContext(summary=None, constraints=None))


async def test_out_of_scope_short_circuits_before_plan_or_execute(monkeypatch) -> None:
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="OUT_OF_SCOPE",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    create_plan_mock = AsyncMock()
    execute_plan_mock = AsyncMock()
    monkeypatch.setattr(loop_module, "create_plan", create_plan_mock)
    monkeypatch.setattr(loop_module, "execute_plan", execute_plan_mock)
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "what's the weather?",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], FinalEvent)
    assert events[0].answer.scope == "OUT_OF_SCOPE"
    create_plan_mock.assert_not_called()
    execute_plan_mock.assert_not_called()


async def test_simple_lookup_with_direct_tool_skips_planner(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="SIMPLE_LOOKUP",
                confidence=0.9,
                rationale="x",
                direct_tool=plan,
                llm_call=_META,
            )
        ),
    )
    create_plan_mock = AsyncMock()
    monkeypatch.setattr(loop_module, "create_plan", create_plan_mock)
    monkeypatch.setattr(
        loop_module,
        "execute_plan",
        AsyncMock(
            return_value=ExecutionResult(ledger=_ledger(), budget_exhausted=False, degraded=False)
        ),
    )
    monkeypatch.setattr(
        loop_module, "synthesize_answer", AsyncMock(return_value=(_answer(), _META))
    )
    monkeypatch.setattr(loop_module, "run_verification", _mock_run_verification(("PASS", [])))
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "what are my holdings?",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    create_plan_mock.assert_not_called()
    assert isinstance(events[0], PlanEvent)
    assert isinstance(events[-1], FinalEvent)


async def test_simple_lookup_without_direct_tool_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="SIMPLE_LOOKUP",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "q",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], FinalEvent)
    assert events[0].answer.decision == "INSUFFICIENT_EVIDENCE"


async def test_clean_portfolio_analysis_path_emits_full_event_sequence(monkeypatch) -> None:
    plan = _plan(n_steps=2)
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="PORTFOLIO_ANALYSIS",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    monkeypatch.setattr(loop_module, "create_plan", AsyncMock(return_value=(plan, [_META])))
    ledger = _ledger(["OK", "OK"])
    monkeypatch.setattr(
        loop_module,
        "execute_plan",
        AsyncMock(
            return_value=ExecutionResult(ledger=ledger, budget_exhausted=False, degraded=False)
        ),
    )
    synth_mock = AsyncMock(return_value=(_answer(), _META))
    monkeypatch.setattr(loop_module, "synthesize_answer", synth_mock)
    verify_mock = _mock_run_verification(("PASS", []))
    monkeypatch.setattr(loop_module, "run_verification", verify_mock)
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "how risky is my portfolio?",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    kinds = [type(e).__name__ for e in events]
    assert kinds == [
        "PlanEvent",
        "ToolStartEvent",
        "ToolDoneEvent",
        "ToolStartEvent",
        "ToolDoneEvent",
        "DraftEvent",
        "VerdictEvent",
        "FinalEvent",
    ]
    verdict_event = events[-2]
    assert isinstance(verdict_event, VerdictEvent)
    assert verdict_event.verdict == "PASS"
    assert verdict_event.repair_attempts == 0
    assert synth_mock.await_count == 1
    assert verify_mock.await_count == 1


async def test_degraded_execution_seeds_limitation_and_caps_confidence(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="PORTFOLIO_ANALYSIS",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    monkeypatch.setattr(loop_module, "create_plan", AsyncMock(return_value=(plan, [_META])))
    ledger = _ledger(["DEGRADED"])
    monkeypatch.setattr(
        loop_module,
        "execute_plan",
        AsyncMock(
            return_value=ExecutionResult(ledger=ledger, budget_exhausted=False, degraded=True)
        ),
    )
    synth_mock = AsyncMock(return_value=(_answer(confidence=0.9), _META))
    monkeypatch.setattr(loop_module, "synthesize_answer", synth_mock)
    monkeypatch.setattr(loop_module, "run_verification", _mock_run_verification(("PASS", [])))
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    await _drain(
        run_agent_loop(
            "q",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    synth_input = synth_mock.await_args.args[0]
    assert synth_input.degraded is True
    assert synth_input.seeded_limitation is not None


async def test_budget_exhausted_execution_seeds_distinct_limitation(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="PORTFOLIO_ANALYSIS",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    monkeypatch.setattr(loop_module, "create_plan", AsyncMock(return_value=(plan, [_META])))
    ledger = _ledger(["OK"])
    monkeypatch.setattr(
        loop_module,
        "execute_plan",
        AsyncMock(
            return_value=ExecutionResult(ledger=ledger, budget_exhausted=True, degraded=False)
        ),
    )
    synth_mock = AsyncMock(return_value=(_answer(confidence=0.9), _META))
    monkeypatch.setattr(loop_module, "synthesize_answer", synth_mock)
    monkeypatch.setattr(loop_module, "run_verification", _mock_run_verification(("PASS", [])))
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    await _drain(
        run_agent_loop(
            "q",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    synth_input = synth_mock.await_args.args[0]
    assert synth_input.degraded is True
    assert "budget" in synth_input.seeded_limitation.lower()


def test_classify_degradation_prefers_budget_exhausted_message() -> None:
    result = ExecutionResult(ledger=_ledger(), budget_exhausted=True, degraded=True)
    degraded, limitation = _classify_degradation(result)
    assert degraded is True
    assert "budget" in (limitation or "").lower()


def test_classify_degradation_clean_execution() -> None:
    result = ExecutionResult(ledger=_ledger(), budget_exhausted=False, degraded=False)
    degraded, limitation = _classify_degradation(result)
    assert degraded is False
    assert limitation is None


async def test_verify_fail_then_pass_exercises_repair(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="PORTFOLIO_ANALYSIS",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    monkeypatch.setattr(loop_module, "create_plan", AsyncMock(return_value=(plan, [_META])))
    monkeypatch.setattr(
        loop_module,
        "execute_plan",
        AsyncMock(
            return_value=ExecutionResult(ledger=_ledger(), budget_exhausted=False, degraded=False)
        ),
    )
    synth_mock = AsyncMock(return_value=(_answer(), _META))
    monkeypatch.setattr(loop_module, "synthesize_answer", synth_mock)
    failing_result = CheckResult(
        layer="V1", check_id="v1.evidence_resolution", verdict="FAIL", message="dangling"
    )
    verify_mock = _mock_run_verification(("FAIL", [failing_result]), ("PASS", []))
    monkeypatch.setattr(loop_module, "run_verification", verify_mock)
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "q",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    assert synth_mock.await_count == 2
    assert verify_mock.await_count == 2
    second_call_input = synth_mock.await_args_list[1].args[0]
    assert second_call_input.repair_feedback is not None
    assert "dangling" in second_call_input.repair_feedback
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.answer.verification.verdict == "PASS"
    assert final.answer.verification.repair_attempts == 1


async def test_verify_fail_twice_triggers_safe_fallback(monkeypatch) -> None:
    plan = _plan()
    monkeypatch.setattr(
        loop_module,
        "classify_intent",
        AsyncMock(
            return_value=IntentResult(
                label="PORTFOLIO_ANALYSIS",
                confidence=0.9,
                rationale="x",
                direct_tool=None,
                llm_call=_META,
            )
        ),
    )
    monkeypatch.setattr(loop_module, "create_plan", AsyncMock(return_value=(plan, [_META])))
    monkeypatch.setattr(
        loop_module,
        "execute_plan",
        AsyncMock(
            return_value=ExecutionResult(ledger=_ledger(), budget_exhausted=False, degraded=False)
        ),
    )
    synth_mock = AsyncMock(return_value=(_answer(), _META))
    monkeypatch.setattr(loop_module, "synthesize_answer", synth_mock)
    failing_result = CheckResult(
        layer="V1", check_id="v1.evidence_resolution", verdict="FAIL", message="dangling"
    )
    verify_mock = _mock_run_verification(("FAIL", [failing_result]), ("FAIL", [failing_result]))
    monkeypatch.setattr(loop_module, "run_verification", verify_mock)
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "q",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    assert synth_mock.await_count == 2  # capped at 1 repair, no third call
    assert verify_mock.await_count == 2
    final = events[-1]
    assert isinstance(final, FinalEvent)
    assert final.answer.decision == "INSUFFICIENT_EVIDENCE"
    assert final.answer.risk_level == "EXTREME"
    assert final.answer.quant_metrics == {}
    assert final.answer.verification.verdict == "FAIL"
    assert final.answer.verification.repair_attempts == 1


async def test_unhandled_exception_still_yields_one_final_event(monkeypatch) -> None:
    monkeypatch.setattr(loop_module, "classify_intent", AsyncMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(loop_module, "_load_mandate_context", _no_mandate())

    events = await _drain(
        run_agent_loop(
            "q",
            tenant_id="t1",
            portfolio_id=None,
            ctx=_fresh_ctx(),
            client=object(),
            prompts=object(),
            trace_id="tr_1",
        )
    )

    assert len(events) == 1
    assert isinstance(events[0], FinalEvent)
    assert events[0].answer.decision == "INSUFFICIENT_EVIDENCE"


async def test_load_mandate_context_none_when_no_portfolio_id(monkeypatch) -> None:
    invoke_mock = AsyncMock()
    monkeypatch.setattr(loop_module.registry, "invoke", invoke_mock)

    result = await loop_module._load_mandate_context(_fresh_ctx(), None)  # type: ignore[arg-type]

    assert result.summary is None
    assert result.constraints is None
    invoke_mock.assert_not_called()
