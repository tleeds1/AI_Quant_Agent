"""agent/loop.py -- the orchestrator state machine (architecture.md §4.2).

INTAKE -> PLAN -> EXECUTE -> SYNTHESIZE -> VERIFY -> RELEASE, with
REFUSE / DIRECT_TOOL / DEGRADE / REPAIR(x1) / SAFE_FALLBACK branches.

Hand-rolled, not a framework (ADR-002): the transitions below *are* the
governance model, so each stage is a plain, independently testable async
helper and `run_agent_loop` is the one place transition order is decided.
Every code path terminates in exactly one `FinalEvent` carrying a
schema-valid `AgentAnswer` (P3, fail closed).
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import structlog
from anthropic import AsyncAnthropic

from quantagent.agent.budget import RequestBudget
from quantagent.agent.events import (
    DraftEvent,
    FinalEvent,
    LoopEvent,
    PlanEvent,
    ToolDoneEvent,
    ToolStartEvent,
    VerdictEvent,
)
from quantagent.agent.executor import ExecutionResult, execute_plan
from quantagent.agent.intent import IntentResult, classify_intent
from quantagent.agent.planner import Plan, create_plan
from quantagent.agent.synthesizer import SynthesisInput, synthesize_answer
from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import Evidence
from quantagent.contracts.ledger import Ledger
from quantagent.contracts.tools import PortfolioOutput
from quantagent.contracts.verification import VerificationReport
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry
from quantagent.verify.types import CheckResult
from quantagent.verify.verdict import run_verification

logger = structlog.get_logger(__name__)

MAX_REPAIR_ATTEMPTS = 1  # architecture.md §7.7 -- do not raise this


@dataclass(frozen=True, slots=True)
class MandateContext:
    """Both derived forms of one `get_portfolio` call: architecture.md
    §6.2's [MANDATE] prompt block needs the formatted `summary`; V4's rules
    engine (§7.5) needs the raw `constraints` dict to evaluate against (e.g.
    a VaR/concentration limit). Fetched once, not twice, from the same call.
    """

    summary: str | None
    constraints: dict[str, Any] | None


async def run_agent_loop(
    question: str,
    *,
    tenant_id: str,
    portfolio_id: str | None,
    ctx: ToolContext,
    client: AsyncAnthropic,
    prompts: PromptLoader,
    trace_id: str,
) -> AsyncIterator[LoopEvent]:
    """Public entrypoint. Wraps `_run_agent_loop_inner` in a last-resort
    fail-closed net: any exception the inner state machine's own DEGRADE/
    REPAIR/SAFE_FALLBACK handling didn't anticipate still terminates the
    stream with one honest `FinalEvent`, never a bare exception surfaced to
    the SSE layer (P3).
    """
    try:
        async for event in _run_agent_loop_inner(
            question,
            tenant_id=tenant_id,
            portfolio_id=portfolio_id,
            ctx=ctx,
            client=client,
            prompts=prompts,
            trace_id=trace_id,
        ):
            yield event
    except Exception:
        logger.exception("agent_loop_unhandled_exception", trace_id=trace_id)
        yield FinalEvent(answer=build_unrecoverable_error_answer(trace_id))


async def _run_agent_loop_inner(
    question: str,
    *,
    tenant_id: str,
    portfolio_id: str | None,
    ctx: ToolContext,
    client: AsyncAnthropic,
    prompts: PromptLoader,
    trace_id: str,
) -> AsyncIterator[LoopEvent]:
    budget = RequestBudget.from_settings()
    mandate = await _load_mandate_context(ctx, portfolio_id)

    # ---- INTAKE ------------------------------------------------------
    intent = await classify_intent(
        question, client=client, prompts=prompts, mandate_summary=mandate.summary
    )
    logger.info(
        "intake_complete", trace_id=trace_id, label=intent.label, confidence=intent.confidence
    )

    if intent.label == "OUT_OF_SCOPE":
        yield FinalEvent(answer=_build_refusal_answer(trace_id=trace_id, intent=intent))
        return

    # ---- PLAN / DIRECT_TOOL --------------------------------------------
    if intent.label == "SIMPLE_LOOKUP":
        if intent.direct_tool is None:
            logger.error("simple_lookup_missing_direct_tool", trace_id=trace_id)
            yield FinalEvent(
                answer=build_unrecoverable_error_answer(
                    trace_id, reason="intent classifier returned SIMPLE_LOOKUP with no direct_tool"
                )
            )
            return
        plan: Plan = intent.direct_tool
    else:  # PORTFOLIO_ANALYSIS
        plan, _plan_llm_calls = await create_plan(
            question, client=client, prompts=prompts, mandate_summary=mandate.summary
        )

    yield PlanEvent(steps=[step.model_dump(mode="json") for step in plan.steps])

    # ---- EXECUTE -----------------------------------------------------
    execution = await execute_plan(plan, ctx, budget, trace_id)
    for call in execution.ledger.calls:
        yield ToolStartEvent(call_id=call.call_id, tool=call.tool_name)
        yield ToolDoneEvent(call_id=call.call_id, latency_ms=call.latency_ms, status=call.status)

    degraded, seeded_limitation = _classify_degradation(execution)

    # ---- SYNTHESIZE + VERIFY + REPAIR (x1) --------------------------------
    answer, verification = await _synthesize_verify_repair(
        question=question,
        trace_id=trace_id,
        ledger=execution.ledger,
        mandate_summary=mandate.summary,
        mandate_constraints=mandate.constraints,
        degraded=degraded,
        seeded_limitation=seeded_limitation,
        client=client,
        prompts=prompts,
    )
    yield DraftEvent(answer=answer)
    yield VerdictEvent(
        verdict=verification.verdict,
        warnings=verification.warnings,
        repair_attempts=verification.repair_attempts,
    )

    # ---- RELEASE / SAFE_FALLBACK ------------------------------------------
    if verification.verdict == "FAIL":
        answer = _build_safe_fallback_answer(
            trace_id=trace_id,
            ledger=execution.ledger,
            reason="verification failed after the single permitted repair attempt",
            verification=verification,
        )
    yield FinalEvent(answer=answer)


def _classify_degradation(execution: ExecutionResult) -> tuple[bool, str | None]:
    """DEGRADE and budget-exhaustion are distinct triggers (architecture.md
    §4.2) that fold into the same downstream behaviour: proceed to
    SYNTHESIZE on a partial ledger with a seeded limitation, never abort
    (I8: no silent degradation).
    """
    if execution.budget_exhausted:
        return True, (
            "Request budget was exhausted before all planned tool calls completed; "
            "this answer is based on partial data."
        )
    if execution.degraded:
        return True, (
            "One or more tool calls failed, timed out, or returned degraded data; "
            "this answer is based on partial data."
        )
    return False, None


async def _synthesize_verify_repair(
    *,
    question: str,
    trace_id: str,
    ledger: Ledger,
    mandate_summary: str | None,
    mandate_constraints: dict[str, Any] | None,
    degraded: bool,
    seeded_limitation: str | None,
    client: AsyncAnthropic,
    prompts: PromptLoader,
) -> tuple[AgentAnswer, VerificationReport]:
    synth_input = SynthesisInput(
        question=question,
        trace_id=trace_id,
        ledger=ledger,
        mandate_summary=mandate_summary,
        degraded=degraded,
        seeded_limitation=seeded_limitation,
    )
    answer, _meta = await synthesize_answer(synth_input, client=client, prompts=prompts)
    answer, verification, results = await run_verification(
        answer, ledger, client=client, prompts=prompts, mandate_constraints=mandate_constraints
    )
    if verification.verdict != "FAIL":
        return answer, verification

    repair_input = synth_input.model_copy(
        update={"repair_feedback": _describe_verification_failure(results)}
    )
    answer, _meta = await synthesize_answer(repair_input, client=client, prompts=prompts)
    answer, verification, _results = await run_verification(
        answer, ledger, client=client, prompts=prompts, mandate_constraints=mandate_constraints
    )
    verification = verification.model_copy(update={"repair_attempts": MAX_REPAIR_ATTEMPTS})
    answer = answer.model_copy(update={"verification": verification})
    return answer, verification


def _describe_verification_failure(results: list[CheckResult]) -> str:
    """Builds the structured critique text architecture.md §7.7 requires
    ("offending spans, unmatched numbers, rule breaches") fed back to the
    synthesiser's [REPAIR] block, organised by layer -- not a flat dump --
    so the model can see at a glance whether it needs to fix a number, a
    rule breach, or a claim's own overclaiming.
    """
    failing = [r for r in results if r.verdict == "FAIL"]
    if not failing:
        # Defensive: this is only called when verdict=="FAIL", which
        # implies at least one FAIL CheckResult by construction of
        # run_verification's aggregation -- a WARN-only repair would be a
        # caller bug (repair must not trigger on PASS_WITH_WARNINGS).
        return "Verification failed for an unspecified reason; review the answer for defects."

    by_layer: dict[str, list[CheckResult]] = {}
    for result in failing:
        by_layer.setdefault(result.layer, []).append(result)

    sections: list[str] = []
    if "V1" in by_layer:
        lines = "\n".join(f"- {r.message}" for r in by_layer["V1"])
        sections.append(f"Schema/contract defects (V1):\n{lines}")
    if "V2" in by_layer:
        lines = "\n".join(
            f"- offending number {r.offending_text!r} at {r.span}: {r.message}"
            + (
                f" (nearest ledger value: {r.nearest_ledger_value})"
                if r.nearest_ledger_value is not None
                else ""
            )
            for r in by_layer["V2"]
        )
        sections.append(f"Unmatched numbers not found in the ledger (V2):\n{lines}")
    if "V3" in by_layer:
        lines = "\n".join(f"- evidence {r.evidence_id}: {r.message}" for r in by_layer["V3"])
        sections.append(f"Citation defects (V3):\n{lines}")
    if "V4" in by_layer:
        lines = "\n".join(f"- rule {r.rule_id}: {r.message}" for r in by_layer["V4"])
        sections.append(f"Constraint rule breaches (V4):\n{lines}")
    if "V5" in by_layer:
        lines = "\n".join(f"- claim {r.claim_id}: {r.message}" for r in by_layer["V5"])
        sections.append(f"Unsupported/contradicted claims (V5):\n{lines}")

    return (
        "Verification failed. Correct every issue below -- do not argue the point, "
        "and do not narrate what you changed inside `summary`.\n\n" + "\n\n".join(sections)
    )


def _build_refusal_answer(*, trace_id: str, intent: IntentResult) -> AgentAnswer:
    """OUT_OF_SCOPE still produces a schema-valid `AgentAnswer` (P3); reused
    on the `final` SSE event rather than a new event type.
    """
    return AgentAnswer(
        trace_id=trace_id,
        scope="OUT_OF_SCOPE",
        decision="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        confidence_basis=["out_of_scope_request -> no analysis performed"],
        risk_level="LOW",
        horizon="n/a",
        summary=(
            f"This question falls outside financial portfolio analysis ({intent.rationale}). "
            "I can analyze your portfolio's risk, exposure, and concentration -- ask me "
            "about those instead."
        ),
        claims=[],
        evidence=[],
        quant_metrics={},
        constraints_checked=[],
        limitations=["Request was out of scope for this agent; no tools were called."],
        disclosures=["Analysis only, not investment advice."],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


def _build_safe_fallback_answer(
    *, trace_id: str, ledger: Ledger, reason: str, verification: VerificationReport | None = None
) -> AgentAnswer:
    """Deterministic, LLM-free fallback (architecture.md §7.7): no
    narrative synthesis, `decision=INSUFFICIENT_EVIDENCE`. `risk_level` is a
    forced conservative default (EXTREME) since `RiskLevel` has no "unknown"
    literal -- asserted defensively, not assessed, and disclosed as such.
    `quant_metrics` stays empty: `ToolCallRecord.result` is an untyped
    `dict[str, Any]` at the `Ledger` level, and reconstructing a typed
    `MetricValue` from it here would assume a shape the ledger contract
    doesn't guarantee.

    "Every fallback is alerted" (§7.7): a structured log event is the
    minimal real implementation -- a persisted review-sample queue ("sampled
    for review") has no backing store anywhere in this codebase yet,
    deferred and flagged, not silently dropped.
    """
    logger.warning(
        "safe_fallback_triggered",
        trace_id=trace_id,
        reason=reason,
        repair_attempts=(verification.repair_attempts if verification else MAX_REPAIR_ATTEMPTS),
    )
    return AgentAnswer(
        trace_id=trace_id,
        scope="PORTFOLIO",
        decision="INSUFFICIENT_EVIDENCE",
        confidence=0.0,
        confidence_basis=[
            "safe_fallback -> synthesis/verification could not be completed reliably"
        ],
        risk_level="EXTREME",
        horizon="n/a",
        summary=(
            "Analysis could not be completed reliably and is withheld. "
            f"Reason: {reason}. Tool results collected before this point are listed "
            "below as raw references only, with no interpretation."
        ),
        claims=[],
        evidence=_evidence_from_ledger(ledger),
        quant_metrics={},
        constraints_checked=[],
        limitations=[
            f"Safe fallback triggered: {reason}.",
            "risk_level is asserted defensively (EXTREME) and does not reflect a "
            "completed risk assessment.",
        ],
        disclosures=[
            "Analysis only, not investment advice.",
            "This response reflects a fallback path; treat with caution.",
        ],
        verification=verification
        or VerificationReport(
            verdict="FAIL", checks=0, warnings=0, repair_attempts=MAX_REPAIR_ATTEMPTS
        ),
    )


def _evidence_from_ledger(ledger: Ledger) -> list[Evidence]:
    """One `Evidence` stub per successfully executed call, referencing the
    ledger's own `call_id` -- does not reconstruct a `MetricValue`.
    """
    return [
        Evidence(
            evidence_id=f"ev_{call.call_id}",
            kind="metric",
            ref=call.call_id,
            excerpt=None,
            char_span=None,
            source_title=call.tool_name,
            source_url=None,
            source_tier=None,
            published_at=None,
            retrieval_score=None,
        )
        for call in ledger.calls
        if call.status in ("OK", "CACHED")
    ]


def build_unrecoverable_error_answer(trace_id: str, reason: str | None = None) -> AgentAnswer:
    """Exposed (not underscore-private) so `api/routes/analyze.py` can reuse
    it for failures that happen before `run_agent_loop` even starts.
    """
    return _build_safe_fallback_answer(
        trace_id=trace_id,
        ledger=Ledger(trace_id=trace_id, calls=[], numeric_index={}),
        reason=reason or "an internal error occurred before analysis could complete",
    )


async def _load_mandate_context(ctx: ToolContext, portfolio_id: str | None) -> MandateContext:
    """Fetches `get_portfolio` once per request. `summary` feeds the
    [MANDATE] prompt block; `constraints` is the raw dict V4's rules engine
    (§7.5) evaluates against -- e.g. a VaR/concentration limit. Both `None`
    for portfolio-less questions.
    """
    if portfolio_id is None:
        return MandateContext(summary=None, constraints=None)
    portfolio = await registry.invoke("get_portfolio", {"portfolio_id": portfolio_id}, ctx)
    assert isinstance(portfolio, PortfolioOutput)
    summary = (
        f"Portfolio {portfolio.name} ({portfolio.portfolio_id}), base currency "
        f"{portfolio.base_currency}, benchmark {portfolio.benchmark_ticker}. "
        f"Mandate constraints: {portfolio.mandate_constraints}."
    )
    return MandateContext(summary=summary, constraints=portfolio.mandate_constraints)
