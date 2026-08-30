"""data/providers/reranker.py -- cross-encoder reranking for RAG hybrid
retrieval (architecture.md §4.7's "cross-encoder rerank of the top 50 ->
top 8").

A second, distinct `sentence-transformers` model (a cross-encoder, not the
bi-encoder in `embeddings.py`) -- lives in `data/providers/` for the same
layering reason as `embeddings.py`: `rag/` may import only `contracts` +
`data` (`.importlinter`'s `rag-scope` contract), never `llm/`.
"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, runtime_checkable

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@runtime_checkable
class RerankerProvider(Protocol):
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        """One relevance score per candidate, in the same order as
        `candidates` -- the caller sorts/truncates, this only scores.
        """
        ...


class SentenceTransformersRerankerProvider:
    """Local CPU inference via a `sentence-transformers` `CrossEncoder`.
    Lazily-loaded, per-instance singleton -- same rationale as
    `embeddings.py::SentenceTransformerEmbeddingProvider`.
    """

    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL) -> None:
        self._model_name = model_name
        self._model: Any | None = None

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        if not candidates:
            return []
        pairs = [(query, candidate) for candidate in candidates]
        return await asyncio.to_thread(self._score, pairs)

    def _score(self, pairs: list[tuple[str, str]]) -> list[float]:
        model = self._load_model()
        return [float(score) for score in model.predict(pairs)]

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self._model_name, device="cpu")
        return self._model
