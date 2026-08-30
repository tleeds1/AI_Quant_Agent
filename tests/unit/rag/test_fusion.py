"""tests/unit/rag/test_fusion.py"""

from __future__ import annotations

from quantagent.rag.fusion import RRF_K, reciprocal_rank_fusion


def test_document_ranked_first_in_both_lists_scores_highest() -> None:
    scores = reciprocal_rank_fusion([["a", "b", "c"], ["a", "c", "b"]])
    assert max(scores, key=lambda doc_id: scores[doc_id]) == "a"


def test_document_absent_from_one_list_still_scores() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["c"]])
    assert scores["a"] == 1.0 / (RRF_K + 1)
    assert scores["b"] == 1.0 / (RRF_K + 2)
    assert scores["c"] == 1.0 / (RRF_K + 1)


def test_document_in_both_lists_sums_contributions() -> None:
    scores = reciprocal_rank_fusion([["a"], ["a"]])
    assert scores["a"] == 2.0 / (RRF_K + 1)


def test_empty_lists_produce_no_scores() -> None:
    assert reciprocal_rank_fusion([[], []]) == {}


def test_custom_k_changes_the_score() -> None:
    default_scores = reciprocal_rank_fusion([["a"]])
    custom_scores = reciprocal_rank_fusion([["a"]], k=1)
    assert custom_scores["a"] > default_scores["a"]  # smaller k -> larger contribution
