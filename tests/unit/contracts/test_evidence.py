from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from quantagent.contracts.evidence import Claim, Evidence


def build_evidence(**overrides: object) -> Evidence:
    defaults: dict[str, object] = {
        "evidence_id": "ev_01",
        "kind": "filing",
        "ref": "doc_42#chunk_7",
        "excerpt": "AI-related capital expenditure concentration risk.",
        "char_span": (120, 168),
        "source_title": "NVDA 10-K Item 1A",
        "source_url": "https://www.sec.gov/example",
        "source_tier": "T1",
        "published_at": datetime(2026, 2, 15, 0, 0, 0),
        "retrieval_score": 0.91,
    }
    defaults.update(overrides)
    return Evidence(**defaults)


def build_claim(**overrides: object) -> Claim:
    defaults: dict[str, object] = {
        "claim_id": "cl_01",
        "text": "AI-linked names contribute 61% of 1-day 95% VaR.",
        "claim_type": "numeric",
        "evidence_ids": ["ev_01"],
        "hedge": "none",
    }
    defaults.update(overrides)
    return Claim(**defaults)


def test_evidence_round_trips_through_json() -> None:
    original = build_evidence()

    restored = Evidence.model_validate_json(original.model_dump_json())

    assert restored == original


def test_claim_round_trips_through_json() -> None:
    original = build_claim()

    restored = Claim.model_validate_json(original.model_dump_json())

    assert restored == original


def test_claim_rejects_empty_evidence_ids() -> None:
    with pytest.raises(ValidationError):
        build_claim(evidence_ids=[])
