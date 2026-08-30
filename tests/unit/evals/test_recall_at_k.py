"""tests/unit/evals/test_recall_at_k.py -- recall@8 through the REAL
`HybridRetriever` (real RRF fusion, real item-filter, real cross-encoder
rerank call, real top-k truncation, real min-score floor) against
`rag_fixtures.py`'s hand-labelled corpus.

No numeric recall@8 threshold is mandated anywhere in architecture.md or
guideline.md (only citation precision has a hard `>= 0.98` CI-gate number,
§10.4) -- the threshold below is a placeholder pending real calibration
against a labelled set with the real `sentence-transformers` models
(docs/PROGRESS.md's M5 section), not an architecture-mandated figure. This
test validates the retrieval ALGORITHM (fusion/filter/rerank/truncation)
against known-correct rankings, not real semantic retrieval quality --
see rag_fixtures.py's module docstring.
"""

from __future__ import annotations

from quantagent.rag.retrieval import HybridRetriever, RetrievalFilters
from tests.unit.evals.rag_fixtures import (
    RECALL_CASES,
    FakeChunkSearchRepository,
    FakeEmbeddings,
    FakeReranker,
    recall_at_k,
)

_PLACEHOLDER_RECALL_THRESHOLD = 0.7


async def test_recall_at_8_meets_the_placeholder_threshold() -> None:
    retriever = HybridRetriever(
        FakeChunkSearchRepository(), FakeEmbeddings(), FakeReranker(), "fake-model"
    )

    per_case_recall: list[float] = []
    for case in RECALL_CASES:
        results = await retriever.search(case.query, RetrievalFilters(ticker=case.ticker), top_k=8)
        retrieved_ids = [r.chunk_id for r in results]
        per_case_recall.append(recall_at_k(retrieved_ids, case.expected_relevant_chunk_ids))

    aggregate_recall = sum(per_case_recall) / len(per_case_recall)
    print(
        f"[rag-recall@8] aggregate={aggregate_recall:.3f} "
        f"per_case={[round(r, 3) for r in per_case_recall]} (n={len(RECALL_CASES)} cases)"
    )
    assert aggregate_recall >= _PLACEHOLDER_RECALL_THRESHOLD


async def test_top_k_truncation_ranks_relevant_chunks_ahead_of_distractors() -> None:
    """Each per-ticker slice of the fixture corpus has exactly 2 relevant
    chunks and 2 off-topic distractors -- requesting `top_k=2` (rather than
    `top_k=8`, which is larger than the whole per-ticker slice and would
    trivially include everything) proves reranking actually ranks the
    relevant chunks above the distractors, not just that they're present
    somewhere in an unfiltered pool.
    """
    retriever = HybridRetriever(
        FakeChunkSearchRepository(), FakeEmbeddings(), FakeReranker(), "fake-model"
    )

    for case in RECALL_CASES:
        results = await retriever.search(case.query, RetrievalFilters(ticker=case.ticker), top_k=2)
        retrieved_ids = {r.chunk_id for r in results}
        assert (
            retrieved_ids == case.expected_relevant_chunk_ids
        ), f"{case.query!r} retrieved {retrieved_ids}, expected {case.expected_relevant_chunk_ids}"
