"""tests/unit/evals/test_golden_set.py -- the M4 DoD's evidentiary backbone:
proves hallucinated_number_rate==0, every flawed-answer fixture is caught by
the correct layer with the correct check, precision/recall are computed and
reported, the critic is mechanically advisory-vs-blocking as specified, and
the safe-fallback path is both reachable and alerted.
"""

from __future__ import annotations

from evals.fixtures import (
    FIXTURE_CLEAN_PASS,
    FIXTURE_CONTRADICTORY_CLAIMS,
    FIXTURE_DANGLING_EVIDENCE,
    FIXTURE_HALLUCINATED_NUMBER,
    FIXTURE_R001_EXTREME_BUY,
    FIXTURE_R008_GUARANTEED_LANGUAGE,
    FIXTURE_UNRESOLVED_METRIC_REF,
    FIXTURE_UNSUPPORTED_CLAIM,
    GOLDEN_FIXTURES,
)

import quantagent.agent.loop as loop_module
from quantagent.agent.loop import _build_safe_fallback_answer
from quantagent.contracts.ledger import Ledger
from quantagent.llm.prompts import PromptLoader
from quantagent.verify.numeric_grounding import hallucinated_number_rate
from tests.unit.evals.harness import precision_recall, run_fixture

_PROMPTS = PromptLoader()


async def test_clean_answer_passes_end_to_end() -> None:
    run = await run_fixture(FIXTURE_CLEAN_PASS, prompts=_PROMPTS)
    assert run.report.verdict == "PASS"
    assert not any(r.verdict == "FAIL" for r in run.results)


async def test_hallucinated_number_caught_by_v2_with_zero_rate_on_clean_set() -> None:
    flawed = await run_fixture(FIXTURE_HALLUCINATED_NUMBER, prompts=_PROMPTS)
    assert flawed.report.verdict == "FAIL"
    assert any(
        r.layer == "V2" and r.verdict == "FAIL" and r.check_id == "v2.numeric_grounding"
        for r in flawed.results
    )

    # architecture.md §7.3's headline gate: rate == 0 on the *clean* golden
    # set specifically -- a flawed fixture is expected to score >0, so this
    # is asserted against the clean-pass fixture's own V2 results, not the
    # whole batch (which would dilute a real regression with an
    # intentionally-flawed fixture's expected nonzero contribution).
    clean = await run_fixture(FIXTURE_CLEAN_PASS, prompts=_PROMPTS)
    v2_results = [r for r in clean.results if r.layer == "V2"]
    assert hallucinated_number_rate(v2_results) == 0.0


async def test_dangling_evidence_caught_by_v1() -> None:
    run = await run_fixture(FIXTURE_DANGLING_EVIDENCE, prompts=_PROMPTS)
    assert run.report.verdict == "FAIL"
    assert any(
        r.layer == "V1" and r.verdict == "FAIL" and r.check_id == "v1.evidence_resolution"
        for r in run.results
    )


async def test_unresolved_metric_ref_caught_by_v1() -> None:
    run = await run_fixture(FIXTURE_UNRESOLVED_METRIC_REF, prompts=_PROMPTS)
    assert run.report.verdict == "FAIL"
    assert any(
        r.layer == "V1" and r.verdict == "FAIL" and r.check_id == "v1.metric_ref_resolution"
        for r in run.results
    )


async def test_r001_extreme_risk_buy_caught_by_v4() -> None:
    run = await run_fixture(FIXTURE_R001_EXTREME_BUY, prompts=_PROMPTS)
    assert run.report.verdict == "FAIL"
    assert any(
        r.layer == "V4" and r.verdict == "FAIL" and r.rule_id == "R-001" for r in run.results
    )
    # V4 breach also overwrote constraints_checked with the real rule result
    # (not left as the synthesiser's own unverified guess) -- architecture.md
    # §5.3/§7.5; see verify/verdict.py::_finalize.
    assert any(
        c.rule_id == "R-001" and c.status == "BREACH" for c in run.answer.constraints_checked
    )


async def test_r008_guaranteed_language_caught_by_v4() -> None:
    run = await run_fixture(FIXTURE_R008_GUARANTEED_LANGUAGE, prompts=_PROMPTS)
    assert run.report.verdict == "FAIL"
    assert any(
        r.layer == "V4" and r.verdict == "FAIL" and r.rule_id == "R-008" for r in run.results
    )


async def test_unsupported_claim_caught_by_v5() -> None:
    run = await run_fixture(FIXTURE_UNSUPPORTED_CLAIM, prompts=_PROMPTS)
    assert run.report.verdict == "FAIL"
    assert any(
        r.layer == "V5" and r.check_id == "v5.entailment" and r.verdict == "FAIL"
        for r in run.results
    )


async def test_contradictory_claims_caught_by_v5_contradiction() -> None:
    run = await run_fixture(FIXTURE_CONTRADICTORY_CLAIMS, prompts=_PROMPTS)
    assert run.report.verdict == "FAIL"
    assert any(
        r.layer == "V5" and r.check_id == "v5.contradiction" and r.verdict == "FAIL"
        for r in run.results
    )


async def test_partially_supported_is_advisory_unsupported_is_blocking() -> None:
    """Mechanical proxy for §7.6's <8% false-positive gate (no live model
    available in this environment -- see the M4 plan's disclosed limitation):
    proves the aggregation NEVER treats PARTIALLY_SUPPORTED as blocking and
    ALWAYS treats UNSUPPORTED/CONTRADICTED as blocking, by construction,
    regardless of any measured rate.
    """
    from evals.fixtures import GoldenFixture, _base_answer, _clean_ledger, _evidence

    from quantagent.contracts.evidence import Claim
    from tests.unit.llm.fixtures import tool_use_response

    partially_supported_fixture = GoldenFixture(
        name="partially_supported_is_advisory",
        ledger=_clean_ledger(),
        answer=_base_answer(
            claims=[
                Claim(
                    claim_id="c1",
                    text="Portfolio VaR is stable.",
                    claim_type="factual",
                    evidence_ids=["ev1"],
                )
            ],
            evidence=[_evidence()],
        ),
        expected_verdict="PASS_WITH_WARNINGS",
        expected_layer="V5",
        expected_check_id="v5.entailment",
        critic_responses=[
            tool_use_response(
                "emit_structured_output",
                {
                    "claim_verdicts": [
                        {
                            "claim_id": "c1",
                            "verdict": "PARTIALLY_SUPPORTED",
                            "reason": "Evidence supports the value but not the stated precision.",
                            "severity": "medium",
                        }
                    ],
                    "contradictions": [],
                },
            )
        ],
    )
    run = await run_fixture(partially_supported_fixture, prompts=_PROMPTS)
    assert run.report.verdict == "PASS_WITH_WARNINGS"  # advisory, never FAIL
    assert any(r.layer == "V5" and r.verdict == "WARN" for r in run.results)
    assert not any(r.layer == "V5" and r.verdict == "FAIL" for r in run.results)
    # merged into limitations per §7.7
    assert any("v5.entailment" in text for text in run.answer.limitations)


async def test_precision_and_recall_are_1_0_on_the_curated_golden_set() -> None:
    runs = [await run_fixture(f, prompts=_PROMPTS) for f in GOLDEN_FIXTURES]
    precision, recall = precision_recall(runs)
    # Achievable and expected here specifically because every fixture is
    # hand-curated with an unambiguous ground truth (no borderline cases) --
    # this is not a claim about a live model's real-world precision/recall,
    # only that the verifier's deterministic reduction logic is correct
    # against a fully-controlled fixture set. A real empirical measurement
    # against a live model is out of reach in this environment (no
    # ANTHROPIC_API_KEY) -- disclosed, not silently assumed equivalent.
    assert precision == 1.0
    assert recall == 1.0


async def test_safe_fallback_is_alerted(monkeypatch) -> None:
    # `structlog.testing.capture_logs()` intercepts by mutating the CURRENT
    # `_Configuration.default_processors` list in place -- but
    # `obs/logging.py::configure_logging()` (called once per FastAPI app
    # lifespan, i.e. by every e2e/api test in this suite) reassigns a FRESH
    # list on each call. A logger cached before the most recent such call
    # (structlog's `cache_logger_on_first_use=True`) holds a reference to a
    # now-stale list, so `capture_logs()` silently captures nothing --
    # reproducible only when run after other tests, not in isolation.
    # Mocking `loop_module.logger` directly sidesteps global structlog state
    # entirely, making this test order-independent.
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class _MockLogger:
        def warning(self, *args: object, **kwargs: object) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr(loop_module, "logger", _MockLogger())

    answer = _build_safe_fallback_answer(
        trace_id="tr_1",
        ledger=Ledger(trace_id="tr_1", calls=[], numeric_index={}),
        reason="verification failed after the single permitted repair attempt",
    )

    assert answer.decision == "INSUFFICIENT_EVIDENCE"
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args[0] == "safe_fallback_triggered"
    assert kwargs.get("trace_id") == "tr_1"
