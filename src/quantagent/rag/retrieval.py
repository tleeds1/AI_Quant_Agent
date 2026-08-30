"""rag/retrieval.py -- hybrid BM25+dense retrieval, RRF fusion, cross-
encoder rerank (architecture.md §4.7). The orchestration layer `tools/
research.py` calls into (`agent/` never calls `rag/` directly per
guideline.md §3).

`rag/` may import only `contracts` + `data` (`.importlinter`'s `rag-scope`
contract) -- `ChunkSearchRepository` is a narrow Protocol matching exactly
`FilingsRepository.search_bm25`/`search_dense`'s real signatures, so a unit
test can substitute a fake without touching sqlite/Postgres, mirroring
`data/providers/factors.py::FactorDataProvider`'s dependency-inversion
pattern.

`RetrievalFilters.ticker` is singular, not a list: `FilingsRepository`'s
search methods take one `ticker: str | None` each, and every RAG tool this
milestone builds (`retrieve_company_filings`, `retrieve_filing_section`)
operates on one ticker at a time -- no caller needs multi-ticker filtering
yet (YAGNI).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

from quantagent.contracts.evidence import SourceTier
from quantagent.data.providers.embeddings import EmbeddingProvider
from quantagent.data.providers.reranker import RerankerProvider
from quantagent.data.repositories.filings_repository import ScoredChunk
from quantagent.rag.fusion import reciprocal_rank_fusion

BM25_DENSE_CANDIDATE_LIMIT = 50  # top-50 per arm before fusion (architecture.md §4.7)
DEFAULT_TOP_K = 8
EXCERPT_MAX_CHARS = 300  # matches contracts.tools.RetrievedFilingChunk.excerpt's max_length

# ms-marco-MiniLM-L-6-v2 (DEFAULT_RERANKER_MODEL) outputs raw, uncalibrated
# logits, not a 0-1 probability -- this floor is a conservative placeholder
# (deliberately low, so it filters only genuinely poor matches) pending real
# calibration against a labelled set; not an architecture-mandated figure.
# See docs/PROGRESS.md's M5 recall@8 note for the same caveat.
MIN_RERANK_SCORE = -10.0


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    ticker: str | None = None
    form_types: list[str] | None = None
    published_after: date | None = None
    published_before: date | None = None
    # Applied post-fusion (see `search`'s docstring for why this one filter,
    # unlike the others, does not need to be pushed into the DB query):
    # `FilingsRepository`'s search methods have no per-item filter, and
    # `retrieve_filing_section`'s ticker+form scope already keeps each
    # arm's top-50 pool small enough that every item is very likely present.
    item: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    chunk_id: str
    ticker: str
    cik: str
    form_type: str
    filed_at: date
    item: str
    section_path: str
    text: str  # full stored chunk text -- used only for injection screening
    excerpt: str
    char_span: tuple[int, int]
    source_url: str
    source_tier: SourceTier
    retrieval_score: float


@runtime_checkable
class ChunkSearchRepository(Protocol):
    async def search_bm25(
        self,
        query_text: str,
        *,
        ticker: str | None = None,
        form_types: Sequence[str] | None = None,
        filed_after: date | None = None,
        filed_before: date | None = None,
        limit: int = BM25_DENSE_CANDIDATE_LIMIT,
    ) -> list[ScoredChunk]: ...

    async def search_dense(
        self,
        query_embedding: Sequence[float],
        model_name: str,
        *,
        ticker: str | None = None,
        form_types: Sequence[str] | None = None,
        filed_after: date | None = None,
        filed_before: date | None = None,
        limit: int = BM25_DENSE_CANDIDATE_LIMIT,
    ) -> list[ScoredChunk]: ...


class HybridRetriever:
    def __init__(
        self,
        repository: ChunkSearchRepository,
        embeddings: EmbeddingProvider,
        reranker: RerankerProvider,
        embedding_model_name: str,
    ) -> None:
        self._repository = repository
        self._embeddings = embeddings
        self._reranker = reranker
        self._embedding_model_name = embedding_model_name

    async def search(
        self, query_text: str, filters: RetrievalFilters, *, top_k: int = DEFAULT_TOP_K
    ) -> list[RetrievedChunk]:
        """1. BM25 and dense search run concurrently (independent I/O).
        2. Results are fused by RRF, deduplicating a chunk that appears in
           both arms.
        3. `filters.item`, if set, is applied post-fusion (see
           `RetrievalFilters.item`'s docstring for why this one filter is
           the exception).
        4. The fused pool is cross-encoder reranked and truncated to
           `top_k`, dropping anything below `MIN_RERANK_SCORE`.
        Every OTHER filter is applied identically to both arms as a hard
        pre-retrieval filter, never a post-fusion Python one: filtering
        only one arm would silently let stale documents back in through
        the other, and filtering after fusion could leave fewer than
        `top_k` results even though fresher candidates existed further
        down an unfiltered pool.
        """
        query_embedding = await self._embeddings.embed_query(query_text)
        bm25_hits, dense_hits = await asyncio.gather(
            self._repository.search_bm25(
                query_text,
                ticker=filters.ticker,
                form_types=filters.form_types,
                filed_after=filters.published_after,
                filed_before=filters.published_before,
                limit=BM25_DENSE_CANDIDATE_LIMIT,
            ),
            self._repository.search_dense(
                query_embedding,
                self._embedding_model_name,
                ticker=filters.ticker,
                form_types=filters.form_types,
                filed_after=filters.published_after,
                filed_before=filters.published_before,
                limit=BM25_DENSE_CANDIDATE_LIMIT,
            ),
        )
        fused = _fuse(bm25_hits, dense_hits)
        if filters.item is not None:
            fused = [hit for hit in fused if hit.chunk.item.upper() == filters.item.upper()]
        if not fused:
            return []

        candidate_texts = [hit.chunk.chunk_text for hit in fused]
        rerank_scores = await self._reranker.score(query_text, candidate_texts)
        ranked = sorted(
            zip(fused, rerank_scores, strict=True), key=lambda pair: pair[1], reverse=True
        )

        results: list[RetrievedChunk] = []
        for hit, rerank_score in ranked:
            if rerank_score < MIN_RERANK_SCORE:
                continue
            results.append(_to_retrieved_chunk(hit, rerank_score))
            if len(results) >= top_k:
                break
        return results


def _fuse(bm25_hits: list[ScoredChunk], dense_hits: list[ScoredChunk]) -> list[ScoredChunk]:
    bm25_ids = [hit.chunk.chunk_id for hit in bm25_hits]
    dense_ids = [hit.chunk.chunk_id for hit in dense_hits]
    rrf_scores = reciprocal_rank_fusion([bm25_ids, dense_ids])

    by_id: dict[str, ScoredChunk] = {}
    for hit in (*bm25_hits, *dense_hits):
        by_id.setdefault(hit.chunk.chunk_id, hit)

    ordered_ids = sorted(rrf_scores, key=lambda chunk_id: rrf_scores[chunk_id], reverse=True)
    return [by_id[chunk_id] for chunk_id in ordered_ids]


def _to_retrieved_chunk(scored: ScoredChunk, rerank_score: float) -> RetrievedChunk:
    excerpt = scored.chunk.chunk_text[:EXCERPT_MAX_CHARS]
    return RetrievedChunk(
        chunk_id=scored.chunk.chunk_id,
        ticker=scored.filing.ticker,
        cik=scored.filing.cik,
        form_type=scored.filing.form_type,
        filed_at=scored.filing.filed_at,
        item=scored.chunk.item,
        section_path=scored.chunk.section_path,
        text=scored.chunk.chunk_text,
        excerpt=excerpt,
        char_span=(0, len(excerpt)),
        source_url=scored.filing.primary_document_url,
        source_tier=scored.filing.source_tier,
        retrieval_score=rerank_score,
    )
