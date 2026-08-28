from __future__ import annotations

from datetime import date, datetime

from quantagent.verify.citation import run_v3_checks
from tests.unit.verify.builders import (
    FakeDocumentIndex,
    RaisingDocumentIndex,
    build_answer,
    build_claim,
    build_evidence,
)


def _check(results, evidence_id: str, subcheck: str):
    matches = [r for r in results if r.check_id == f"{evidence_id}:{subcheck}"]
    assert (
        matches
    ), f"no {subcheck!r} CheckResult for {evidence_id!r} in {[r.check_id for r in results]}"
    return matches[0]


def test_valid_filing_citation_all_pass() -> None:
    index = FakeDocumentIndex()
    index.seed(
        "doc1",
        chunk_text="Revenue grew 4.3% year over year, driven by cloud segment growth.",
        source_tier="T2",
        published_at=datetime(2026, 6, 1),
        source_url="https://example.com/doc1",
    )
    evidence = build_evidence(
        "ev1",
        kind="filing",
        ref="doc1",
        excerpt="Revenue grew 4.3% year over year",
        char_span=(0, 33),
        source_url="https://example.com/doc1",
    )
    answer = build_answer(
        claims=[build_claim("c1", ["ev1"], claim_type="factual")], evidence=[evidence]
    )

    results = run_v3_checks(
        answer, document_index=index, requested_window=(date(2026, 1, 1), date(2026, 12, 31))
    )

    assert all(r.verdict == "PASS" for r in results), results


def test_fabricated_document_id_fails_and_stops_at_document_exists() -> None:
    index = FakeDocumentIndex()
    evidence = build_evidence("ev1", kind="filing", ref="does_not_exist")
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=index)

    assert len(results) == 1
    assert results[0].check_id == "ev1:document_exists"
    assert results[0].verdict == "FAIL"


def test_excerpt_fuzzy_mismatch_fails() -> None:
    index = FakeDocumentIndex()
    index.seed(
        "doc1",
        chunk_text="Revenue grew 4.3% year over year.",
        source_tier="T1",
        published_at=datetime(2026, 6, 1),
    )
    evidence = build_evidence(
        "ev1",
        kind="filing",
        ref="doc1",
        excerpt="Profits declined sharply in Q2",
        char_span=(0, 33),
    )
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=index)

    check = _check(results, "ev1", "excerpt_match")
    assert check.verdict == "FAIL"
    assert "fuzzy ratio" in check.message


def test_char_span_out_of_bounds_fails() -> None:
    index = FakeDocumentIndex()
    index.seed("doc1", chunk_text="short text", source_tier="T1", published_at=datetime(2026, 6, 1))
    evidence = build_evidence(
        "ev1", kind="filing", ref="doc1", excerpt="short text", char_span=(0, 999)
    )
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=index)

    assert _check(results, "ev1", "excerpt_match").verdict == "FAIL"


def test_stale_published_at_outside_requested_window_fails() -> None:
    index = FakeDocumentIndex()
    index.seed("doc1", chunk_text="text", source_tier="T1", published_at=datetime(2020, 1, 1))
    evidence = build_evidence("ev1", kind="filing", ref="doc1")
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(
        answer, document_index=index, requested_window=(date(2026, 1, 1), date(2026, 12, 31))
    )

    assert _check(results, "ev1", "published_window").verdict == "FAIL"


def test_no_requested_window_skips_published_check_entirely() -> None:
    index = FakeDocumentIndex()
    index.seed("doc1", chunk_text="text", source_tier="T1", published_at=datetime(2020, 1, 1))
    evidence = build_evidence("ev1", kind="filing", ref="doc1")
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=index)

    assert not any(r.check_id == "ev1:published_window" for r in results)


def test_insufficient_source_tier_for_causal_claim_fails() -> None:
    index = FakeDocumentIndex()
    index.seed("doc1", chunk_text="text", source_tier="T4", published_at=datetime(2026, 6, 1))
    evidence = build_evidence("ev1", kind="filing", ref="doc1")
    answer = build_answer(
        claims=[build_claim("c1", ["ev1"], claim_type="causal")], evidence=[evidence]
    )

    results = run_v3_checks(answer, document_index=index)

    assert _check(results, "ev1", "source_tier").verdict == "FAIL"


def test_source_url_mismatch_fails() -> None:
    index = FakeDocumentIndex()
    index.seed(
        "doc1",
        chunk_text="text",
        source_tier="T1",
        published_at=datetime(2026, 6, 1),
        source_url="https://example.com/real",
    )
    evidence = build_evidence(
        "ev1", kind="filing", ref="doc1", source_url="https://example.com/fake"
    )
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=index)

    assert _check(results, "ev1", "source_url").verdict == "FAIL"


def test_document_index_none_fails_closed() -> None:
    evidence = build_evidence("ev1", kind="filing", ref="doc1")
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=None)

    assert len(results) == 1
    assert results[0].verdict == "FAIL"
    assert "no citation index" in results[0].message


def test_non_retrieval_kind_evidence_never_touches_the_index() -> None:
    evidence = build_evidence("ev1", kind="metric", ref="portfolio_var")
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=RaisingDocumentIndex())

    assert results == []


def test_evidence_with_no_excerpt_url_claimed_passes_those_subchecks() -> None:
    index = FakeDocumentIndex()
    index.seed("doc1", chunk_text="text", source_tier="T1", published_at=datetime(2026, 6, 1))
    evidence = build_evidence(
        "ev1", kind="filing", ref="doc1", excerpt=None, char_span=None, source_url=None
    )
    answer = build_answer(evidence=[evidence])

    results = run_v3_checks(answer, document_index=index)

    assert _check(results, "ev1", "excerpt_match").verdict == "PASS"
    assert _check(results, "ev1", "source_url").verdict == "PASS"
