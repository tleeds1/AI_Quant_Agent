from __future__ import annotations

import quantagent.verify.verdict as verdict_module
from quantagent.llm.prompts import PromptLoader
from quantagent.verify.verdict import run_verification
from tests.unit.llm.fixtures import build_mock_llm_client, tool_use_response
from tests.unit.verify.builders import build_answer, build_claim, build_evidence, build_ledger

_OUTPUT_TOOL = "emit_structured_output"


def _supported_critic_response(claim_id: str = "c1") -> dict[str, object]:
    return tool_use_response(
        _OUTPUT_TOOL,
        {
            "claim_verdicts": [
                {"claim_id": claim_id, "verdict": "SUPPORTED", "reason": "x", "severity": "low"}
            ],
            "contradictions": [],
        },
    )


async def test_v1_fail_short_circuits_v2_through_v5(monkeypatch) -> None:
    calls = {"v2": 0, "v3": 0, "v4": 0, "v5": 0}
    monkeypatch.setattr(
        verdict_module,
        "run_v2_numeric_grounding",
        lambda *a, **k: calls.__setitem__("v2", calls["v2"] + 1) or [],
    )
    monkeypatch.setattr(
        verdict_module,
        "run_v3_checks",
        lambda *a, **k: calls.__setitem__("v3", calls["v3"] + 1) or [],
    )
    monkeypatch.setattr(
        verdict_module,
        "run_v4_checks",
        lambda *a, **k: (calls.__setitem__("v4", calls["v4"] + 1), ([], []))[1],
    )

    async def _fake_v5(*a, **k):
        calls["v5"] += 1
        return [], None

    monkeypatch.setattr(verdict_module, "run_v5_critique", _fake_v5)

    # A dangling evidence id -> V1 FAIL.
    claim = build_claim("c1", ["ev_missing"])
    answer = build_answer(claims=[claim], evidence=[])
    client, _session = build_mock_llm_client([])

    _answer, report, _results, _verify_calls = await run_verification(
        answer, build_ledger(), client=client, prompts=PromptLoader()
    )

    assert report.verdict == "FAIL"
    assert calls == {"v2": 0, "v3": 0, "v4": 0, "v5": 0}


async def test_v4_warn_still_lets_v5_run() -> None:
    # R-003 (WARN severity) breaches via a DEGRADED ledger call + high
    # confidence; V1-V3 all pass, so V4+V5 both run.
    from tests.unit.verify.builders import build_tool_call_record

    claim = build_claim("c1", ["ev1"])
    evidence = build_evidence("ev1", kind="metric", ref="m1")
    from quantagent.contracts.metrics import MetricValue
    from tests.unit.agent.builders import build_provenance

    answer = build_answer(
        claims=[claim],
        evidence=[evidence],
        confidence=0.9,
        quant_metrics={
            "m1": MetricValue(
                metric_id="m1", value=1.0, unit="ratio", method="t", provenance=build_provenance()
            )
        },
    )
    ledger = build_ledger(calls=[build_tool_call_record(status="DEGRADED")])
    client, session = build_mock_llm_client([_supported_critic_response()])

    _answer, report, results, _verify_calls = await run_verification(
        answer, ledger, client=client, prompts=PromptLoader()
    )

    assert session.call_count == 1  # V5 genuinely ran
    assert any(r.check_id == "R-003" and r.verdict == "WARN" for r in results)
    assert report.verdict in ("PASS_WITH_WARNINGS", "FAIL")


async def test_v4_fail_still_lets_v5_run() -> None:
    # R-001 (BREACH severity) -> V4 FAIL, but V5 still runs per the "always
    # run once V1-V3 pass" rule.
    from quantagent.contracts.metrics import MetricValue
    from tests.unit.agent.builders import build_provenance

    claim = build_claim("c1", ["ev1"])
    evidence = build_evidence("ev1", kind="metric", ref="m1")
    answer = build_answer(
        claims=[claim],
        evidence=[evidence],
        risk_level="EXTREME",
        decision="BUY",
        quant_metrics={
            "m1": MetricValue(
                metric_id="m1", value=1.0, unit="ratio", method="t", provenance=build_provenance()
            )
        },
    )
    client, session = build_mock_llm_client([_supported_critic_response()])

    _answer, report, results, _verify_calls = await run_verification(
        answer, build_ledger(), client=client, prompts=PromptLoader()
    )

    assert session.call_count == 1
    assert report.verdict == "FAIL"
    assert any(r.check_id == "R-001" and r.verdict == "FAIL" for r in results)


async def test_clean_answer_passes_and_constraints_checked_is_overwritten() -> None:
    claim = build_claim("c1", ["ev1"])
    evidence = build_evidence("ev1", kind="metric", ref="m1")
    from quantagent.contracts.metrics import MetricValue
    from tests.unit.agent.builders import build_provenance

    answer = build_answer(
        claims=[claim],
        evidence=[evidence],
        quant_metrics={
            "m1": MetricValue(
                metric_id="m1", value=1.0, unit="ratio", method="t", provenance=build_provenance()
            )
        },
        constraints_checked=[],  # synthesizer's own (empty) guess
    )
    client, _session = build_mock_llm_client([_supported_critic_response()])

    updated_answer, report, _results, _verify_calls = await run_verification(
        answer, build_ledger(), client=client, prompts=PromptLoader()
    )

    assert report.verdict == "PASS"
    assert len(updated_answer.constraints_checked) == 10  # V4's real 10-rule output


async def test_zero_claims_answer_skips_v5_but_still_passes() -> None:
    # decision="NO_ACTION" specifically avoids R-007's "decision != NO_ACTION
    # requires >=1 fresh cited metric" rule from WARNing on this metric-free
    # fixture -- that WARN would be real, correct R-007 behavior, just not
    # what this test is checking (V5 being skipped for a zero-claim answer).
    answer = build_answer(claims=[], evidence=[], decision="NO_ACTION")
    client, session = build_mock_llm_client([])

    _answer, report, _results, _verify_calls = await run_verification(
        answer, build_ledger(), client=client, prompts=PromptLoader()
    )

    assert session.call_count == 0
    assert report.verdict == "PASS"
