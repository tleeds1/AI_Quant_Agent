"""data/providers/embeddings.py -- dense embedding generation for RAG hybrid
retrieval (architecture.md §4.7). Local, CPU-only `sentence-transformers`
inference -- the embedding decision resolved in docs/PROGRESS.md's M5
section. Lives in `data/providers/` (an I/O-shaped adapter, like
`YFinancePriceProvider`), never in `llm/`: `rag/` may import only
`contracts` + `data` (`.importlinter`'s `rag-scope` contract), so embedding
generation must be a data-layer provider, not the Anthropic wrapper in
`llm/client.py`.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
# Must match the HNSW index's opclass in the Alembic migration
# (`vector_cosine_ops`): embeddings are stored L2-normalized specifically so
# a cosine-space index returns correctly-ordered results.
EMBEDDING_METRIC = "cosine"
_BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class EmbeddingProvider(Protocol):
    @property
    def dimension(self) -> int: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


class SentenceTransformerEmbeddingProvider:
    """Local CPU inference via a `sentence-transformers` bi-encoder.

    The model is a lazily-loaded, per-instance singleton: loading takes a
    couple of seconds, so it must not repeat per call. Inference is
    CPU-bound; wrapped in `asyncio.to_thread` rather than made naively
    `async def` (guideline.md §4.5 -- making a CPU-bound function `async`
    solves nothing and hides the blocking).

    Bi-encoder asymmetry: bge models are trained with a query-side
    instruction prefix that document-side text does not get. Mismatching
    this silently halves retrieval quality, so it is baked into
    `embed_query` here rather than left for callers to remember.
    """

    def __init__(self, model_name: str = DEFAULT_EMBEDDING_MODEL, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: Any | None = None

    @property
    def dimension(self) -> int:
        return EMBEDDING_DIM

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await asyncio.to_thread(self._encode, list(texts))

    async def embed_query(self, text: str) -> list[float]:
        [embedding] = await asyncio.to_thread(self._encode, [f"{_BGE_QUERY_INSTRUCTION}{text}"])
        return embedding

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load_model()
        vectors = model.encode(
            texts, batch_size=self._batch_size, normalize_embeddings=True, convert_to_numpy=True
        )
        return [row.tolist() for row in vectors]

    def _load_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model
