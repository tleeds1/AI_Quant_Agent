"""verify/citation.py -- V3: citation validity (architecture.md §7.4).

Scope: only `kind in {"filing", "news"}` Evidence is retrieval-sourced (a
document/chunk to validate against an index). `kind in {"metric",
"transaction", "market_data"}` Evidence references a ledger tool call, not a
retrieved document -- it has no document_id/excerpt/char_span to validate in
the sense §7.4 means. Those kinds are already covered by other layers: V1's
structural check (`Evidence.ref` of kind "metric" resolves to a
`quant_metrics` key) and V2's numeric-grounding layer (the cited number
actually matches the ledger). Passing those kinds through V3 untouched --
`document_index` is never even called for them -- is deliberate, not a
stub-that-always-passes: there is nothing for V3 to check that isn't already
someone else's job.

Nothing in this codebase produces "filing"/"news" evidence today (that's
M5/RAG scope); the checks below are still real, generically-correct code
against a `DocumentIndex` Protocol M5's real implementation will satisfy,
not a stub that always passes. `.importlinter`'s verify-cross-cutting
contract forbids `verify/` from importing `rag/`, so this Protocol is
defined here (dependency inversion) -- M5 supplies a concrete
implementation from outside; `verify/` never imports a concrete `rag/` type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from difflib import SequenceMatcher
from typing import Protocol

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import ClaimType, Evidence, SourceTier
from quantagent.verify.types import CheckResult

_RETRIEVAL_KINDS = frozenset({"filing", "news"})
_FUZZY_MATCH_THRESHOLD = 0.95

# T1 = strongest/primary source, T4 = weakest (contracts/evidence.py's
# SourceTier ordering is nominal, not ordinal -- we impose the order here).
_TIER_ORDER: dict[SourceTier, int] = {"T1": 1, "T2": 2, "T3": 3, "T4": 4}

# Minimum tier a piece of retrieval evidence must carry to support a given
# claim type. Independent of, and complementary to, V4/R-006's "a causal
# claim needs >=1 T1/T2 evidence somewhere among possibly several" downgrade
# rule: V3 here fails the individual citation outright if ITS OWN tier can't
# support what it's cited for (defense in depth, not duplication -- V3
# judges the citation, R-006 judges the claim as a whole).
_MIN_TIER_FOR_CLAIM_TYPE: dict[ClaimType, int] = {
    "causal": _TIER_ORDER["T2"],
    "forward_looking": _TIER_ORDER["T2"],
    "factual": _TIER_ORDER["T3"],
    "numeric": _TIER_ORDER["T3"],
}


@dataclass(frozen=True)
class DocumentMetadata:
    """Minimal per-document facts V3 needs -- not a full RAG chunk/embedding
    record. M5's `rag/` adapts its real index to this shape.
    """

    document_id: str
    source_tier: SourceTier
    published_at: datetime


class DocumentIndex(Protocol):
    """The retrieval-index interface V3 needs. `verify/` owns this Protocol
    (dependency inversion): M5 supplies a concrete implementation from
    outside; `verify/` never imports it.
    """

    def get_metadata(self, document_id: str) -> DocumentMetadata | None:
        """None if `document_id` is not in the index (fabricated citation)."""
        ...

    def get_chunk_text(self, document_id: str) -> str | None:
        """Full stored text for the chunk `document_id` refers to, against
        which `Evidence.excerpt`/`char_span` are validated. None mirrors
        `get_metadata`'s "not indexed" signal.
        """
        ...

    def resolves_source_url(self, document_id: str, source_url: str) -> bool:
        """True only if `source_url` is a recognised URL for THIS
        `document_id` specifically (not merely a URL that resolves to *some*
        document) -- this is what catches a citation pointing at the right
        title but the wrong/swapped URL.
        """
        ...


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _fuzzy_ratio(a: str, b: str) -> float:
    """stdlib `difflib.SequenceMatcher`, not a new dependency -- no fuzzy-
    matching library is pinned today, and citation excerpts are capped at
    300 chars (`Evidence.excerpt.max_length`), so this is a non-issue in
    practice.
    """
    return SequenceMatcher(None, a, b).ratio()


def _claim_types_for_evidence(answer: AgentAnswer, evidence_id: str) -> set[ClaimType]:
    return {claim.claim_type for claim in answer.claims if evidence_id in claim.evidence_ids}


def run_v3_checks(
    answer: AgentAnswer,
    *,
    document_index: DocumentIndex | None = None,
    requested_window: tuple[date, date] | None = None,
) -> list[CheckResult]:
    """architecture.md §7.4. Emits `CheckResult`s only for `kind in
    {"filing","news"}` Evidence; every other kind is skipped without ever
    calling `document_index`.

    Fail-closed policy (P3): if `kind in {"filing","news"}` Evidence exists
    and `document_index is None`, every such Evidence FAILs its
    `document_exists` sub-check with an explicit "no citation index
    configured" message -- not silently skipped, not optimistically passed.
    Unreachable in production today (nothing produces filing/news evidence
    before M5), but real and tested.

    `requested_window`: no caller supplies this today -- when `None`, the
    `published_window` sub-check is skipped entirely (not counted, not
    failed) for every evidence item -- deliberately different from the
    `document_index is None` case, where the resource is specified and
    simply absent.
    """
    results: list[CheckResult] = []
    for evidence in answer.evidence:
        if evidence.kind not in _RETRIEVAL_KINDS:
            continue
        results.extend(
            _check_one_evidence(
                evidence,
                claim_types=_claim_types_for_evidence(answer, evidence.evidence_id),
                document_index=document_index,
                requested_window=requested_window,
            )
        )
    return results


def _check_one_evidence(
    evidence: Evidence,
    *,
    claim_types: set[ClaimType],
    document_index: DocumentIndex | None,
    requested_window: tuple[date, date] | None,
) -> list[CheckResult]:
    if document_index is None:
        return [
            CheckResult(
                layer="V3",
                check_id=f"{evidence.evidence_id}:document_exists",
                verdict="FAIL",
                message=(
                    f"evidence {evidence.evidence_id} (kind={evidence.kind}) cites "
                    f"document {evidence.ref!r} but no citation index is configured; "
                    "failing closed rather than trusting an unverifiable filing/news "
                    "citation (architecture.md P3)."
                ),
                evidence_id=evidence.evidence_id,
            )
        ]

    metadata = document_index.get_metadata(evidence.ref)
    if metadata is None:
        return [
            CheckResult(
                layer="V3",
                check_id=f"{evidence.evidence_id}:document_exists",
                verdict="FAIL",
                message=(
                    f"evidence {evidence.evidence_id} cites document {evidence.ref!r}, "
                    "which does not exist in the citation index (fabricated citation)."
                ),
                evidence_id=evidence.evidence_id,
            )
        ]

    results = [
        CheckResult(
            layer="V3",
            check_id=f"{evidence.evidence_id}:document_exists",
            verdict="PASS",
            message=f"document {evidence.ref!r} exists in the citation index.",
            evidence_id=evidence.evidence_id,
        ),
        _check_excerpt_and_span(evidence, document_index),
        _check_source_url(evidence, document_index),
        _check_source_tier(evidence, metadata, claim_types),
    ]
    published_check = _check_published_window(evidence, metadata, requested_window)
    if published_check is not None:
        results.append(published_check)
    return results


def _check_excerpt_and_span(evidence: Evidence, document_index: DocumentIndex) -> CheckResult:
    check_id = f"{evidence.evidence_id}:excerpt_match"
    if evidence.excerpt is None or evidence.char_span is None:
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="PASS",
            message="no excerpt/char_span claimed; nothing to validate.",
            evidence_id=evidence.evidence_id,
        )
    chunk_text = document_index.get_chunk_text(evidence.ref)
    if chunk_text is None:
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="FAIL",
            message=f"document {evidence.ref!r} has metadata but no retrievable chunk text.",
            evidence_id=evidence.evidence_id,
        )
    start, end = evidence.char_span
    if not (0 <= start < end <= len(chunk_text)):
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="FAIL",
            message=(
                f"char_span {evidence.char_span} is out of bounds for document "
                f"{evidence.ref!r} (chunk length {len(chunk_text)})."
            ),
            evidence_id=evidence.evidence_id,
        )
    spanned = _normalise_whitespace(chunk_text[start:end])
    claimed = _normalise_whitespace(evidence.excerpt)
    ratio = _fuzzy_ratio(spanned, claimed)
    if ratio < _FUZZY_MATCH_THRESHOLD:
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="FAIL",
            message=(
                f"excerpt does not match stored text at char_span {evidence.char_span} "
                f"(fuzzy ratio {ratio:.3f} < {_FUZZY_MATCH_THRESHOLD}); citation may "
                "misrepresent the source."
            ),
            evidence_id=evidence.evidence_id,
        )
    return CheckResult(
        layer="V3",
        check_id=check_id,
        verdict="PASS",
        message=f"excerpt matches stored text (fuzzy ratio {ratio:.3f}).",
        evidence_id=evidence.evidence_id,
    )


def _check_source_url(evidence: Evidence, document_index: DocumentIndex) -> CheckResult:
    check_id = f"{evidence.evidence_id}:source_url"
    if evidence.source_url is None:
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="PASS",
            message="no source_url claimed; nothing to validate.",
            evidence_id=evidence.evidence_id,
        )
    if not document_index.resolves_source_url(evidence.ref, evidence.source_url):
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="FAIL",
            message=(
                f"source_url {evidence.source_url!r} does not resolve to document "
                f"{evidence.ref!r}."
            ),
            evidence_id=evidence.evidence_id,
        )
    return CheckResult(
        layer="V3",
        check_id=check_id,
        verdict="PASS",
        message="source_url resolves to the cited document.",
        evidence_id=evidence.evidence_id,
    )


def _check_published_window(
    evidence: Evidence, metadata: DocumentMetadata, requested_window: tuple[date, date] | None
) -> CheckResult | None:
    if requested_window is None:
        return None
    window_start, window_end = requested_window
    published_date = metadata.published_at.date()
    check_id = f"{evidence.evidence_id}:published_window"
    if not (window_start <= published_date <= window_end):
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="FAIL",
            message=(
                f"document {evidence.ref!r} published {published_date.isoformat()}, "
                f"outside the requested window [{window_start}, {window_end}]."
            ),
            evidence_id=evidence.evidence_id,
        )
    return CheckResult(
        layer="V3",
        check_id=check_id,
        verdict="PASS",
        message="published_at falls inside the requested window.",
        evidence_id=evidence.evidence_id,
    )


def _check_source_tier(
    evidence: Evidence, metadata: DocumentMetadata, claim_types: set[ClaimType]
) -> CheckResult:
    check_id = f"{evidence.evidence_id}:source_tier"
    if not claim_types:
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="PASS",
            message="evidence is not linked to any claim; no tier floor applies.",
            evidence_id=evidence.evidence_id,
        )
    required = max(_MIN_TIER_FOR_CLAIM_TYPE[ct] for ct in claim_types)
    actual = _TIER_ORDER[metadata.source_tier]
    if actual > required:  # higher number == weaker tier
        strictest = max(claim_types, key=lambda ct: _MIN_TIER_FOR_CLAIM_TYPE[ct])
        return CheckResult(
            layer="V3",
            check_id=check_id,
            verdict="FAIL",
            message=(
                f"document {evidence.ref!r} is tier {metadata.source_tier}, below the "
                f"minimum required to support a {strictest!r} claim."
            ),
            evidence_id=evidence.evidence_id,
        )
    return CheckResult(
        layer="V3",
        check_id=check_id,
        verdict="PASS",
        message=f"document tier {metadata.source_tier} satisfies the claim-type floor.",
        evidence_id=evidence.evidence_id,
    )
