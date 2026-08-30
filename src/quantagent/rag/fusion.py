"""rag/fusion.py -- Reciprocal Rank Fusion (architecture.md §4.7: "BM25 ...
dense, fused with RRF"). Pure math, no I/O.
"""

from __future__ import annotations

RRF_K = 60  # standard RRF constant -- named, never inlined (guideline.md §4.3)


def reciprocal_rank_fusion(ranked_id_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """`score(d) = sum_r 1/(k + rank_r(d))`, `rank_r` 1-based within each
    ranked list; a document absent from a given list contributes 0 for
    that list. Higher score = more relevant.
    """
    scores: dict[str, float] = {}
    for ranked_ids in ranked_id_lists:
        for rank, doc_id in enumerate(ranked_ids, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores
