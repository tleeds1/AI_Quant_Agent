"""tests/unit/data/providers/test_reranker.py

Never loads real `sentence-transformers` model weights (guideline.md
§10.3) -- `_load_model` is monkeypatched with a deterministic fake.
"""

from __future__ import annotations

import pytest

from quantagent.data.providers.reranker import (
    RerankerProvider,
    SentenceTransformersRerankerProvider,
)


class _FakeCrossEncoder:
    def __init__(self) -> None:
        self.scored_pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        self.scored_pairs.extend(pairs)
        return [float(len(candidate)) for _query, candidate in pairs]


def test_satisfies_the_reranker_provider_protocol() -> None:
    provider: RerankerProvider = SentenceTransformersRerankerProvider()
    assert isinstance(provider, RerankerProvider)


async def test_score_returns_one_score_per_candidate_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SentenceTransformersRerankerProvider()
    fake_model = _FakeCrossEncoder()
    monkeypatch.setattr(provider, "_load_model", lambda: fake_model)

    scores = await provider.score("query", ["a", "bb", "ccc"])

    assert scores == [1.0, 2.0, 3.0]
    assert fake_model.scored_pairs == [("query", "a"), ("query", "bb"), ("query", "ccc")]


async def test_score_empty_candidates_short_circuits(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SentenceTransformersRerankerProvider()
    fake_model = _FakeCrossEncoder()
    monkeypatch.setattr(provider, "_load_model", lambda: fake_model)

    scores = await provider.score("query", [])

    assert scores == []
    assert fake_model.scored_pairs == []


def test_load_model_caches_the_constructed_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    construction_count = 0

    class _FakeCrossEncoderClass:
        def __init__(self, model_name: str, device: str) -> None:
            nonlocal construction_count
            construction_count += 1

    monkeypatch.setattr("sentence_transformers.CrossEncoder", _FakeCrossEncoderClass)
    provider = SentenceTransformersRerankerProvider()

    first = provider._load_model()
    second = provider._load_model()

    assert construction_count == 1
    assert first is second
