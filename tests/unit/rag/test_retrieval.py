"""tests/unit/rag/test_retrieval.py -- exercises the real RRF fusion/
truncation/item-filter/min-score logic in `HybridRetriever`, against fake
repository/embedding/reranker collaborators (no sqlite/Postgres, no real
model weights -- guideline.md §10.3).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from quantagent.data.repositories.filings_repository import (
    FilingChunkRecord,
    FilingMeta,
    ScoredChunk,
)
from quantagent.rag.retrieval import (
    EXCERPT_MAX_CHARS,
    HybridRetriever,
    RetrievalFilters,
    _to_retrieved_chunk,
)

_FILING = FilingMeta(
    id="acc-1",
    cik="0001045810",
    ticker="NVDA",
    company_name="NVIDIA CORP",
    form_type="10-K",
    filed_at=date(2024, 2, 21),
    period_of_report=date(2024, 1, 28),
    primary_document_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
    source_tier="T1",
)


def _scored(chunk_id: str, item: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=FilingChunkRecord(
            chunk_id=chunk_id,
            filing_id="acc-1",
            item=item,
            section_path=f"10-K#Item {item}",
            chunk_index=0,
            chunk_text=text,
            char_count=len(text),
            embedding_model="fake-model",
        ),
        filing=_FILING,
        score=0.0,
    )


class FakeRepository:
    def __init__(self, bm25_hits: list[ScoredChunk], dense_hits: list[ScoredChunk]) -> None:
        self._bm25_hits = bm25_hits
        self._dense_hits = dense_hits
        self.bm25_calls: list[dict[str, object]] = []
        self.dense_calls: list[dict[str, object]] = []

    async def search_bm25(self, query_text: str, **kwargs: object) -> list[ScoredChunk]:
        self.bm25_calls.append({"query_text": query_text, **kwargs})
        return self._bm25_hits

    async def search_dense(
        self, query_embedding: Sequence[float], model_name: str, **kwargs: object
    ) -> list[ScoredChunk]:
        self.dense_calls.append({"model_name": model_name, **kwargs})
        return self._dense_hits


class FakeEmbeddings:
    async def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    @property
    def dimension(self) -> int:
        return 2


class FakeReranker:
    """Assigns each candidate a score equal to its position in a fixed
    preference order, so tests can assert on final ranking deterministically.
    """

    def __init__(self, preference_order: list[str]) -> None:
        self._preference_order = preference_order

    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [
            float(len(self._preference_order) - self._preference_order.index(c)) for c in candidates
        ]


async def test_search_dedupes_a_chunk_present_in_both_arms() -> None:
    shared = _scored("c1", "1A", "shared text")
    repository = FakeRepository(bm25_hits=[shared], dense_hits=[shared])
    retriever = HybridRetriever(
        repository, FakeEmbeddings(), FakeReranker(["shared text"]), "fake-model"
    )

    results = await retriever.search("query", RetrievalFilters())

    assert [r.chunk_id for r in results] == ["c1"]


async def test_search_fuses_and_reranks_across_arms() -> None:
    bm25_hits = [_scored("c1", "1A", "bm25 text")]
    dense_hits = [_scored("c2", "1A", "dense text")]
    repository = FakeRepository(bm25_hits, dense_hits)
    retriever = HybridRetriever(
        repository, FakeEmbeddings(), FakeReranker(["dense text", "bm25 text"]), "fake-model"
    )

    results = await retriever.search("query", RetrievalFilters())

    assert [r.chunk_id for r in results] == ["c2", "c1"]  # reranker prefers dense text


async def test_search_truncates_to_top_k() -> None:
    hits = [_scored(f"c{i}", "1A", f"text {i}") for i in range(5)]
    repository = FakeRepository(bm25_hits=hits, dense_hits=[])
    retriever = HybridRetriever(
        repository, FakeEmbeddings(), FakeReranker([f"text {i}" for i in range(5)]), "fake-model"
    )

    results = await retriever.search("query", RetrievalFilters(), top_k=2)

    assert len(results) == 2


async def test_search_applies_item_filter_post_fusion() -> None:
    hits = [_scored("c1", "1A", "risk text"), _scored("c2", "7", "mda text")]
    repository = FakeRepository(bm25_hits=hits, dense_hits=[])
    retriever = HybridRetriever(
        repository, FakeEmbeddings(), FakeReranker(["risk text", "mda text"]), "fake-model"
    )

    results = await retriever.search("query", RetrievalFilters(item="1A"))

    assert [r.chunk_id for r in results] == ["c1"]


async def test_search_drops_results_below_min_rerank_score() -> None:
    hits = [_scored("c1", "1A", "low relevance text")]
    repository = FakeRepository(bm25_hits=hits, dense_hits=[])

    class _AllBelowFloorReranker:
        async def score(self, query: str, candidates: list[str]) -> list[float]:
            return [-999.0 for _ in candidates]

    retriever = HybridRetriever(
        repository, FakeEmbeddings(), _AllBelowFloorReranker(), "fake-model"
    )

    results = await retriever.search("query", RetrievalFilters())

    assert results == []


async def test_search_returns_empty_when_no_candidates_at_all() -> None:
    repository = FakeRepository(bm25_hits=[], dense_hits=[])
    retriever = HybridRetriever(repository, FakeEmbeddings(), FakeReranker([]), "fake-model")

    results = await retriever.search("query", RetrievalFilters())

    assert results == []


async def test_search_passes_filters_identically_to_both_arms() -> None:
    repository = FakeRepository(bm25_hits=[], dense_hits=[])
    retriever = HybridRetriever(repository, FakeEmbeddings(), FakeReranker([]), "fake-model")
    filters = RetrievalFilters(
        ticker="NVDA",
        form_types=["10-K"],
        published_after=date(2023, 1, 1),
        published_before=date(2025, 1, 1),
    )

    await retriever.search("query", filters)

    for call in (repository.bm25_calls[0], repository.dense_calls[0]):
        assert call["ticker"] == "NVDA"
        assert call["form_types"] == ["10-K"]
        assert call["filed_after"] == date(2023, 1, 1)
        assert call["filed_before"] == date(2025, 1, 1)


def test_retrieved_chunk_excerpt_and_char_span_and_tier_come_from_the_filing() -> None:
    long_text = "x" * (EXCERPT_MAX_CHARS + 50)
    scored = _scored("c1", "1A", long_text)

    retrieved = _to_retrieved_chunk(scored, rerank_score=1.0)

    assert retrieved.excerpt == long_text[:EXCERPT_MAX_CHARS]
    assert retrieved.char_span == (0, EXCERPT_MAX_CHARS)
    assert retrieved.source_tier == "T1"
    assert retrieved.source_url == _FILING.primary_document_url
