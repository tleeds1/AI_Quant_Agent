"""tests/unit/evals/citation_harness.py -- runs CITATION_FIXTURES through
the real `verify.citation.run_v3_checks` and computes architecture.md
§10.4's citation-precision gate (`>= 0.98`). Not itself a test module --
imported by test_citation_precision_filings.py, same split as
tests/unit/evals/harness.py/evals/fixtures.py.

Deterministic, synchronous, no LLM call: V3 never calls a model, so this
(unlike the full golden-set harness) needs no mocked Anthropic client and
computes a real number with no live `ANTHROPIC_API_KEY` required.
"""

from __future__ import annotations

from evals.citation_fixtures import CitationFixture

from quantagent.verify.citation import run_v3_checks
from quantagent.verify.types import CheckResult


def run_citation_fixture(fixture: CitationFixture) -> list[CheckResult]:
    return run_v3_checks(fixture.answer, document_index=fixture.document_index)


def citation_precision_recall(fixtures: list[CitationFixture]) -> tuple[float, float]:
    """precision = of all fixtures where at least one V3 check FAILed, how
    many were genuinely flawed (expected_verdict == "FAIL"); recall = of
    all genuinely-flawed fixtures, how many produced >=1 V3 FAIL. Reported
    via `print` as well as returned, matching `harness.py::precision_recall`'s
    convention (visible in `pytest -s` output).
    """
    predicted_fail: list[CitationFixture] = []
    true_flawed: list[CitationFixture] = []
    true_positives: list[CitationFixture] = []

    for fixture in fixtures:
        results = run_citation_fixture(fixture)
        has_fail = any(r.verdict == "FAIL" for r in results)
        if has_fail:
            predicted_fail.append(fixture)
        if fixture.expected_verdict == "FAIL":
            true_flawed.append(fixture)
            if has_fail:
                true_positives.append(fixture)

    precision = len(true_positives) / len(predicted_fail) if predicted_fail else 1.0
    recall = len(true_positives) / len(true_flawed) if true_flawed else 1.0
    print(
        f"[citation-precision] precision={precision:.3f} recall={recall:.3f} "
        f"(fixtures={len(fixtures)}, flawed={len(true_flawed)}, flagged={len(predicted_fail)})"
    )
    return precision, recall
