"""tests/unit/data/providers/test_embeddings.py

Never loads real `sentence-transformers` model weights (guideline.md
§10.3: "no test touches a live network endpoint") -- `_load_model` is
monkeypatched with a deterministic fake so `embed_documents`/`embed_query`'s
own logic (batching pass-through, the query-instruction prefix asymmetry)
is exercised for real.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from quantagent.data.providers.embeddings import (
    EMBEDDING_DIM,
    EmbeddingProvider,
    SentenceTransformerEmbeddingProvider,
)


class _FakeModel:
    """Records every text it was asked to encode; returns a 1-valued numpy
    array per text, matching real `SentenceTransformer.encode`'s
    `convert_to_numpy=True` output shape (`_encode` calls `.tolist()` on
    each row).
    """

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(self, texts: list[str], **_kwargs: Any) -> np.ndarray:
        self.encoded_texts.extend(texts)
        return np.ones((len(texts), EMBEDDING_DIM))


def test_satisfies_the_embedding_provider_protocol() -> None:
    provider: EmbeddingProvider = SentenceTransformerEmbeddingProvider()
    assert isinstance(provider, EmbeddingProvider)


def test_dimension_matches_the_declared_constant() -> None:
    provider = SentenceTransformerEmbeddingProvider()
    assert provider.dimension == EMBEDDING_DIM


async def test_embed_documents_returns_one_vector_per_text(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SentenceTransformerEmbeddingProvider()
    fake_model = _FakeModel()
    monkeypatch.setattr(provider, "_load_model", lambda: fake_model)

    vectors = await provider.embed_documents(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(v) == EMBEDDING_DIM for v in vectors)
    assert fake_model.encoded_texts == ["alpha", "beta"]  # no query instruction prefix


async def test_embed_documents_empty_input_short_circuits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = SentenceTransformerEmbeddingProvider()
    fake_model = _FakeModel()
    monkeypatch.setattr(provider, "_load_model", lambda: fake_model)

    vectors = await provider.embed_documents([])

    assert vectors == []
    assert fake_model.encoded_texts == []  # never even calls the model


async def test_embed_query_prepends_the_bge_instruction(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = SentenceTransformerEmbeddingProvider()
    fake_model = _FakeModel()
    monkeypatch.setattr(provider, "_load_model", lambda: fake_model)

    vector = await provider.embed_query("supply chain risk")

    assert len(vector) == EMBEDDING_DIM
    assert fake_model.encoded_texts == [
        "Represent this sentence for searching relevant passages: supply chain risk"
    ]


def test_load_model_caches_the_constructed_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercises the real (unmocked) `_load_model`, patching only the
    `SentenceTransformer` constructor it calls -- proves the lazy-singleton
    caching itself works, not just that a monkeypatched replacement does.
    """
    construction_count = 0

    class _FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            nonlocal construction_count
            construction_count += 1

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", _FakeSentenceTransformer)
    provider = SentenceTransformerEmbeddingProvider()

    first = provider._load_model()
    second = provider._load_model()

    assert construction_count == 1
    assert first is second
