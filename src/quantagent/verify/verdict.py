"""verify/verdict.py -- top-level VERIFY orchestrator (architecture.md §7,
§7.7). Runs V1-V5 in spec order with the short-circuit semantics documented
below, reduces the combined `CheckResult`s into one `VerificationReport`,
and returns the answer with `constraints_checked` overwritten by V4's real
rule output -- never the synthesiser's own guess.

LAYER SHORT-CIRCUITING (the §7 diagram leaves this implicit; stated here
rather than guessed silently): the diagram marks V1, V2, V3 "fail = hard
stop" but V4 "breach = downgrade/block" and V5 "fail = repair" -- neither
labelled "hard stop". Honoured literally: a V1/V2/V3 FAIL skips every layer
after it (V2/V3 assume a schema-valid, evidence-resolved answer to run
safely against; skipping V5 avoids a real LLM call against an answer already
guaranteed to fail and go to repair). V4 and V5 both always run once V1-V3
pass, regardless of V4's own outcome -- because repair is capped at exactly
one attempt, so that single repair pass should see the fullest possible
critique across every applicable layer at once, not just the first thing
that broke.

CONCURRENCY: V1-V4 are pure, synchronous, sub-millisecond functions with no
shared mutable state and no I/O -- unlike M3's executor (real network-bound
tool calls), `asyncio.gather` here would add coordination overhead for no
measurable win, and the V1/V2/V3 short-circuiting above requires sequencing
anyway. Called sequentially, in spec order. Only V5 is async (a real LLM
call).
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.ledger import Ledger
from quantagent.contracts.verification import (
    ConstraintCheck,
    VerificationReport,
    VerificationVerdict,
)
from quantagent.llm.prompts import PromptLoader
from quantagent.verify.citation import DocumentIndex, run_v3_checks
from quantagent.verify.constraint_rules import run_v4_checks
from quantagent.verify.numeric_grounding import run_v2_numeric_grounding
from quantagent.verify.structural import run_v1_checks
from quantagent.verify.types import CheckResult
from quantagent.verify.v5_critic import run_v5_critique


async def run_verification(
    answer: AgentAnswer,
    ledger: Ledger,
    *,
    client: AsyncAnthropic,
    prompts: PromptLoader,
    document_index: DocumentIndex | None = None,
    mandate_constraints: dict[str, Any] | None = None,
    critic_model: str | None = None,
) -> tuple[AgentAnswer, VerificationReport, list[CheckResult]]:
    """Returns `(answer_with_constraints_checked_and_limitations_updated,
    report, all_check_results)`. `all_check_results` is the raw, layer-
    tagged material `agent/loop.py::_describe_verification_failure` formats
    into the repair-prompt's structured critique -- this function only
    judges, it never formats prompt text (single responsibility).
    """
    results: list[CheckResult] = []

    v1_results = run_v1_checks(answer)
    results.extend(v1_results)
    if _has_fail(v1_results):
        return _finalize(answer, results, constraint_checks=[])

    v2_results = run_v2_numeric_grounding(answer, ledger)
    results.extend(v2_results)
    if _has_fail(v2_results):
        return _finalize(answer, results, constraint_checks=[])

    v3_results = run_v3_checks(answer, document_index=document_index)
    results.extend(v3_results)
    if _has_fail(v3_results):
        return _finalize(answer, results, constraint_checks=[])

    v4_results, constraint_checks = run_v4_checks(
        answer, ledger, mandate_constraints=mandate_constraints
    )
    results.extend(v4_results)

    v5_results = await run_v5_critique(answer, client=client, prompts=prompts, model=critic_model)
    results.extend(v5_results)

    return _finalize(answer, results, constraint_checks=constraint_checks)


def _has_fail(results: list[CheckResult]) -> bool:
    return any(r.verdict == "FAIL" for r in results)


def _aggregate(results: list[CheckResult]) -> VerificationVerdict:
    """§7.7: any FAIL -> FAIL; else any WARN -> PASS_WITH_WARNINGS; else PASS."""
    if any(r.verdict == "FAIL" for r in results):
        return "FAIL"
    if any(r.verdict == "WARN" for r in results):
        return "PASS_WITH_WARNINGS"
    return "PASS"


def _merge_warning_limitations(existing: list[str], results: list[CheckResult]) -> list[str]:
    """§7.7: "warnings are merged into `limitations` and surfaced." Only
    WARN entries reach here -- when the aggregate verdict is FAIL, the
    caller (agent/loop.py) replaces the whole answer via repair/safe-
    fallback instead of folding FAIL text into `limitations`.
    """
    merged = list(existing)
    for result in results:
        if result.verdict != "WARN":
            continue
        text = f"[{result.layer}/{result.check_id}] {result.message}"
        if text not in merged:
            merged.append(text)
    return merged


def _finalize(
    answer: AgentAnswer,
    results: list[CheckResult],
    *,
    constraint_checks: list[ConstraintCheck],
) -> tuple[AgentAnswer, VerificationReport, list[CheckResult]]:
    verdict = _aggregate(results)
    report = VerificationReport(
        verdict=verdict,
        checks=len(results),
        warnings=sum(1 for r in results if r.verdict == "WARN"),
        repair_attempts=0,
    )
    updated_answer = answer.model_copy(
        update={
            "constraints_checked": constraint_checks,
            "verification": report,
            "limitations": _merge_warning_limitations(answer.limitations, results),
        }
    )
    return updated_answer, report, results
