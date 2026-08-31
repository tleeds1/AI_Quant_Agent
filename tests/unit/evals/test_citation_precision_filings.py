"""tests/unit/evals/test_citation_precision_filings.py -- V3 citation
validity (`verify.citation.run_v3_checks`) against `kind="filing"`
Evidence, using the shared `evals/citation_fixtures.py` (one fixture per
§7.4 sub-check failure mode, mirroring M4's golden-set convention).

M4's golden set (`evals/fixtures.py`) only ever exercises `kind="metric"`
Evidence -- nothing produced `kind="filing"` Evidence before M5's RAG
tools existed, so citation precision was previously unmeasurable against
anything but a vacuous, zero-citation case. This closes that gap and
computes architecture.md §10.4's real citation-precision gate (`>= 0.98`).
"""

from __future__ import annotations

from evals.citation_fixtures import (
    CITATION_FIXTURES,
    FIXTURE_CLEAN_FILING_CITATION,
    FIXTURE_EXCERPT_MISMATCH,
    FIXTURE_FABRICATED_CITATION,
    FIXTURE_T4_TIER_ON_CAUSAL_CLAIM,
    FIXTURE_WRONG_SOURCE_URL,
)

from tests.unit.evals.citation_harness import citation_precision_recall, run_citation_fixture


def test_clean_filing_citation_passes_every_subcheck() -> None:
    results = run_citation_fixture(FIXTURE_CLEAN_FILING_CITATION)
    assert results
    assert not any(r.verdict == "FAIL" for r in results)


def test_fabricated_citation_fails_document_exists() -> None:
    results = run_citation_fixture(FIXTURE_FABRICATED_CITATION)
    assert any(r.check_id.endswith(":document_exists") and r.verdict == "FAIL" for r in results)


def test_excerpt_not_matching_stored_text_fails_excerpt_match() -> None:
    results = run_citation_fixture(FIXTURE_EXCERPT_MISMATCH)
    assert any(r.check_id.endswith(":excerpt_match") and r.verdict == "FAIL" for r in results)


def test_wrong_source_url_fails_source_url_check() -> None:
    results = run_citation_fixture(FIXTURE_WRONG_SOURCE_URL)
    assert any(r.check_id.endswith(":source_url") and r.verdict == "FAIL" for r in results)


def test_t4_tier_citation_fails_a_causal_claims_tier_floor() -> None:
    results = run_citation_fixture(FIXTURE_T4_TIER_ON_CAUSAL_CLAIM)
    assert any(r.check_id.endswith(":source_tier") and r.verdict == "FAIL" for r in results)


def test_citation_precision_meets_the_architecture_gate() -> None:
    """architecture.md §10.4: citation precision >= 0.98, measured for
    real against `CITATION_FIXTURES` -- no live model needed (V3 is
    deterministic). This is the number `evals/run_scorecard.py` reports.
    """
    precision, recall = citation_precision_recall(CITATION_FIXTURES)
    assert precision >= 0.98
    assert recall >= 0.98
