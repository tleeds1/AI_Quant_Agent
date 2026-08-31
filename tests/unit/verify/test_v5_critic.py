from __future__ import annotations

from quantagent.llm.prompts import PromptLoader
from quantagent.verify.v5_critic import run_v5_critique
from tests.unit.llm.fixtures import build_mock_llm_client, tool_use_response
from tests.unit.verify.builders import build_answer, build_claim, build_evidence

_OUTPUT_TOOL = "emit_structured_output"


async def test_empty_claims_skips_without_any_llm_call() -> None:
    client, session = build_mock_llm_client([])
    answer = build_answer(claims=[], evidence=[])

    results, _meta = await run_v5_critique(answer, client=client, prompts=PromptLoader())

    assert results == []
    assert session.call_count == 0


async def test_prompt_excludes_summary_decision_and_confidence() -> None:
    prompts = PromptLoader()
    claim = build_claim("c1", ["ev1"], text="Portfolio VaR is 2.5%.")
    evidence = build_evidence("ev1", kind="metric")
    answer = build_answer(
        claims=[claim],
        evidence=[evidence],
        summary="THIS_SECRET_SUMMARY_MUST_NOT_LEAK",
        decision="BUY",
        confidence=0.987654,
    )
    client, session = build_mock_llm_client(
        [
            tool_use_response(
                _OUTPUT_TOOL,
                {
                    "claim_verdicts": [
                        {
                            "claim_id": "c1",
                            "verdict": "SUPPORTED",
                            "reason": "matches",
                            "severity": "low",
                        }
                    ],
                    "contradictions": [],
                },
            )
        ]
    )

    _, _ = await run_v5_critique(answer, client=client, prompts=prompts)

    body = session.request_body(0)
    system_text = body["messages"][0]["content"]
    assert body["messages"][0]["role"] == "system"
    assert "THIS_SECRET_SUMMARY_MUST_NOT_LEAK" not in system_text
    assert "0.987654" not in system_text
    assert '"decision"' not in system_text.lower()


async def test_contradiction_produces_two_check_results_sharing_message_content() -> None:
    claim1 = build_claim("c1", ["ev1"], text="VaR is 2.5%.")
    claim2 = build_claim("c2", ["ev1"], text="VaR is 4.0%.")
    evidence = build_evidence("ev1", kind="metric")
    answer = build_answer(claims=[claim1, claim2], evidence=[evidence])
    client, _session = build_mock_llm_client(
        [
            tool_use_response(
                _OUTPUT_TOOL,
                {
                    "claim_verdicts": [
                        {
                            "claim_id": "c1",
                            "verdict": "SUPPORTED",
                            "reason": "x",
                            "severity": "low",
                        },
                        {
                            "claim_id": "c2",
                            "verdict": "SUPPORTED",
                            "reason": "x",
                            "severity": "low",
                        },
                    ],
                    "contradictions": [
                        {
                            "claim_ids": ["c1", "c2"],
                            "explanation": "conflicting VaR figures",
                            "severity": "high",
                        }
                    ],
                },
            )
        ]
    )

    results, _meta = await run_v5_critique(answer, client=client, prompts=PromptLoader())

    contradiction_results = [r for r in results if r.check_id == "v5.contradiction"]
    assert len(contradiction_results) == 1
    assert contradiction_results[0].verdict == "FAIL"
    assert "c1" in contradiction_results[0].message and "c2" in contradiction_results[0].message


async def test_single_llm_call_regardless_of_claim_count() -> None:
    claims = [build_claim(f"c{i}", ["ev1"], text=f"claim {i}") for i in range(3)]
    evidence = build_evidence("ev1", kind="metric")
    answer = build_answer(claims=claims, evidence=[evidence])
    client, session = build_mock_llm_client(
        [
            tool_use_response(
                _OUTPUT_TOOL,
                {
                    "claim_verdicts": [
                        {
                            "claim_id": f"c{i}",
                            "verdict": "SUPPORTED",
                            "reason": "x",
                            "severity": "low",
                        }
                        for i in range(3)
                    ],
                    "contradictions": [],
                },
            )
        ]
    )

    _, _ = await run_v5_critique(answer, client=client, prompts=PromptLoader())

    assert session.call_count == 1


async def test_partially_supported_maps_to_warn_unsupported_maps_to_fail() -> None:
    claim1 = build_claim("c1", ["ev1"], text="x")
    claim2 = build_claim("c2", ["ev1"], text="y")
    evidence = build_evidence("ev1", kind="metric")
    answer = build_answer(claims=[claim1, claim2], evidence=[evidence])
    client, _session = build_mock_llm_client(
        [
            tool_use_response(
                _OUTPUT_TOOL,
                {
                    "claim_verdicts": [
                        {
                            "claim_id": "c1",
                            "verdict": "PARTIALLY_SUPPORTED",
                            "reason": "x",
                            "severity": "medium",
                        },
                        {
                            "claim_id": "c2",
                            "verdict": "UNSUPPORTED",
                            "reason": "x",
                            "severity": "high",
                        },
                    ],
                    "contradictions": [],
                },
            )
        ]
    )

    results, _meta = await run_v5_critique(answer, client=client, prompts=PromptLoader())

    by_claim = {r.claim_id: r.verdict for r in results}
    assert by_claim["c1"] == "WARN"
    assert by_claim["c2"] == "FAIL"
