"""tests/unit/evals/harness.py -- runs GOLDEN_FIXTURES through the real
verify/verdict.py::run_verification pipeline and computes the DoD's
precision/recall numbers. Not itself a test module (no test_ functions) --
imported by test_golden_set.py, same split as tests/unit/llm/fixtures.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.verification import VerificationReport
from quantagent.llm.prompts import PromptLoader
from quantagent.verify.types import CheckResult
from quantagent.verify.verdict import run_verification
from tests.unit.evals.fixtures import GoldenFixture
from tests.unit.llm.fixtures import build_mock_anthropic


@dataclass(frozen=True, slots=True)
class FixtureRun:
    fixture: GoldenFixture
    answer: AgentAnswer
    report: VerificationReport
    results: list[CheckResult]


async def run_fixture(fixture: GoldenFixture, *, prompts: PromptLoader) -> FixtureRun:
    client, _session = build_mock_anthropic(fixture.critic_responses or [])
    answer, report, results = await run_verification(
        fixture.answer, fixture.ledger, client=client, prompts=prompts
    )
    return FixtureRun(fixture=fixture, answer=answer, report=report, results=results)


def precision_recall(runs: list[FixtureRun]) -> tuple[float, float]:
    """precision = of all FAIL verdicts produced, how many were on a
    genuinely-flawed fixture (expected_verdict == "FAIL"), not a clean one
    incorrectly failed; recall = of all genuinely-flawed fixtures, how many
    were actually caught. Reported (printed) as well as returned so the
    numbers are visible in `pytest -s` output per the DoD's "measured and
    reported" wording.
    """
    predicted_fail = [r for r in runs if r.report.verdict == "FAIL"]
    true_flawed = [r for r in runs if r.fixture.expected_verdict == "FAIL"]
    true_positives = [r for r in predicted_fail if r.fixture.expected_verdict == "FAIL"]

    precision = len(true_positives) / len(predicted_fail) if predicted_fail else 1.0
    recall = len(true_positives) / len(true_flawed) if true_flawed else 1.0
    print(
        f"[golden-set] precision={precision:.3f} recall={recall:.3f} "
        f"(fixtures={len(runs)}, flawed={len(true_flawed)}, flagged={len(predicted_fail)})"
    )
    return precision, recall
