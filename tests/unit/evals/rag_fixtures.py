"""tests/unit/evals/rag_fixtures.py -- a small, hand-labelled retrieval
corpus for `test_recall_at_k.py`. Not itself a test module (no `test_`
functions) -- same split as `tests/unit/evals/fixtures.py`.

`FakeChunkSearchRepository` implements `rag.retrieval.ChunkSearchRepository`
structurally (duck-typed, no inheritance needed). `search_bm25` and
`FakeReranker` both use a deterministic term-overlap heuristic against the
real query text (both receive it in production too). `search_dense` cannot
use the raw query text -- the real repository method only ever receives an
embedding vector -- so each corpus chunk is tagged with a fixed 1-D "topic"
scalar and `FakeEmbeddings.embed_query` maps a query to the matching topic
scalar; `search_dense` then ranks by distance in that 1-D space, a
realistic stand-in for cosine-similarity ranking.

This validates the REAL `HybridRetriever` orchestration (RRF fusion, item
filter, cross-encoder rerank, top-k truncation, min-score floor) against
known-correct rankings -- it does not validate real BM25/embedding
quality, which lives in Postgres/pgvector and real model weights, out of a
unit fixture's reach.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from quantagent.data.repositories.filings_repository import (
    FilingChunkRecord,
    FilingMeta,
    ScoredChunk,
)

SUPPLY_CHAIN_TOPIC = 1.0
ANTITRUST_TOPIC = 2.0
CYBERSECURITY_TOPIC = 3.0
OTHER_TOPIC = 9.0  # far from every real topic above -- distractor chunks

_FILINGS: dict[str, FilingMeta] = {
    "NVDA": FilingMeta(
        id="nvda-10k",
        cik="0001045810",
        ticker="NVDA",
        company_name="NVIDIA CORP",
        form_type="10-K",
        filed_at=date(2024, 2, 21),
        period_of_report=date(2024, 1, 28),
        primary_document_url="https://www.sec.gov/Archives/edgar/data/1045810/x/nvda.htm",
        source_tier="T1",
    ),
    "AAPL": FilingMeta(
        id="aapl-10k",
        cik="0000320193",
        ticker="AAPL",
        company_name="APPLE INC",
        form_type="10-K",
        filed_at=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        primary_document_url="https://www.sec.gov/Archives/edgar/data/320193/x/aapl.htm",
        source_tier="T1",
    ),
    "MSFT": FilingMeta(
        id="msft-10k",
        cik="0000789019",
        ticker="MSFT",
        company_name="MICROSOFT CORP",
        form_type="10-K",
        filed_at=date(2024, 7, 30),
        period_of_report=date(2024, 6, 30),
        primary_document_url="https://www.sec.gov/Archives/edgar/data/789019/x/msft.htm",
        source_tier="T1",
    ),
}

# (chunk_id, ticker, item, text, topic) -- one clearly-relevant "risk" chunk
# and one clearly-relevant "MD&A" chunk per ticker on its named topic, plus
# one off-topic distractor per ticker, so recall@8 is a real, non-trivial
# signal rather than "return everything and pass trivially."
_CORPUS: list[tuple[str, str, str, str, float]] = [
    (
        "nvda-10k#1A#0000",
        "NVDA",
        "1A",
        "Our supply chain is highly concentrated among a small number of third-party foundries.",
        SUPPLY_CHAIN_TOPIC,
    ),
    (
        "nvda-10k#7#0000",
        "NVDA",
        "7",
        "Revenue grew on data center demand; foundry supply chain capacity remains a constraint.",
        SUPPLY_CHAIN_TOPIC,
    ),
    (
        "nvda-10k#1#0000",
        "NVDA",
        "1",
        "We design graphics processing units and are headquartered in Santa Clara, California.",
        OTHER_TOPIC,
    ),
    (
        "aapl-10k#1A#0000",
        "AAPL",
        "1A",
        "We are subject to antitrust and competition-law regulatory scrutiny in several markets.",
        ANTITRUST_TOPIC,
    ),
    (
        "aapl-10k#7#0000",
        "AAPL",
        "7",
        "Services revenue increased; regulatory antitrust proceedings in the EU may affect terms.",
        ANTITRUST_TOPIC,
    ),
    (
        "aapl-10k#1#0000",
        "AAPL",
        "1",
        "We design, manufacture, and market smartphones, computers, and wearable devices.",
        OTHER_TOPIC,
    ),
    (
        "msft-10k#1A#0000",
        "MSFT",
        "1A",
        "A cybersecurity incident or data breach could damage our reputation and cause loss.",
        CYBERSECURITY_TOPIC,
    ),
    (
        "msft-10k#7#0000",
        "MSFT",
        "7",
        "Cloud revenue increased; we keep investing in cybersecurity to reduce breach exposure.",
        CYBERSECURITY_TOPIC,
    ),
    (
        "msft-10k#1#0000",
        "MSFT",
        "1",
        "We develop, license, and support a wide range of software products and cloud services.",
        OTHER_TOPIC,
    ),
    (
        "nvda-10k#1A#0001",
        "NVDA",
        "1A",
        "Our stock price has been and may continue to be volatile, unrelated to one risk factor.",
        OTHER_TOPIC,
    ),
    (
        "aapl-10k#1A#0001",
        "AAPL",
        "1A",
        "Currency exchange rate fluctuations could adversely affect our reported foreign sales.",
        OTHER_TOPIC,
    ),
    (
        "msft-10k#1A#0001",
        "MSFT",
        "1A",
        "Changes in tax law in any jurisdiction where we operate could raise our tax rate.",
        OTHER_TOPIC,
    ),
]

_TOPIC_KEYWORDS: dict[float, tuple[str, ...]] = {
    SUPPLY_CHAIN_TOPIC: ("supply", "chain", "foundries", "foundry"),
    ANTITRUST_TOPIC: ("antitrust", "regulatory", "regulators", "competition-law"),
    CYBERSECURITY_TOPIC: ("cybersecurity", "cyber", "data", "breach"),
}


@dataclass(frozen=True, slots=True)
class RecallCase:
    query: str
    ticker: str | None
    expected_relevant_chunk_ids: frozenset[str]


RECALL_CASES: list[RecallCase] = [
    RecallCase(
        query="What does NVDA's 10-K say about supply chain risk?",
        ticker="NVDA",
        expected_relevant_chunk_ids=frozenset({"nvda-10k#1A#0000", "nvda-10k#7#0000"}),
    ),
    RecallCase(
        query="What antitrust and regulatory risks does AAPL disclose?",
        ticker="AAPL",
        expected_relevant_chunk_ids=frozenset({"aapl-10k#1A#0000", "aapl-10k#7#0000"}),
    ),
    RecallCase(
        query="What does MSFT say about cybersecurity and data breach risk?",
        ticker="MSFT",
        expected_relevant_chunk_ids=frozenset({"msft-10k#1A#0000", "msft-10k#7#0000"}),
    ),
]


def _term_overlap_score(query: str, text: str) -> float:
    query_terms = set(query.lower().split())
    text_terms = set(text.lower().split())
    return float(len(query_terms & text_terms))


def _query_topic(query_text: str) -> float:
    normalized = query_text.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(keyword in normalized for keyword in keywords):
            return topic
    return OTHER_TOPIC


def _all_chunks(ticker: str | None) -> list[ScoredChunk]:
    return [
        ScoredChunk(
            chunk=FilingChunkRecord(
                chunk_id=chunk_id,
                filing_id=_FILINGS[chunk_ticker].id,
                item=item,
                section_path=f"10-K#Item {item}",
                chunk_index=0,
                chunk_text=text,
                char_count=len(text),
                embedding_model="fake-model",
            ),
            filing=_FILINGS[chunk_ticker],
            score=0.0,
        )
        for chunk_id, chunk_ticker, item, text, _topic in _CORPUS
        if ticker is None or chunk_ticker == ticker
    ]


_TOPIC_BY_CHUNK_ID: dict[str, float] = {chunk_id: topic for chunk_id, *_rest, topic in _CORPUS}


class FakeChunkSearchRepository:
    """Structurally satisfies `rag.retrieval.ChunkSearchRepository`."""

    async def search_bm25(
        self,
        query_text: str,
        *,
        ticker: str | None = None,
        form_types: Sequence[str] | None = None,
        filed_after: object = None,
        filed_before: object = None,
        limit: int = 50,
    ) -> list[ScoredChunk]:
        candidates = _all_chunks(ticker)
        candidates.sort(
            key=lambda c: _term_overlap_score(query_text, c.chunk.chunk_text), reverse=True
        )
        return candidates[:limit]

    async def search_dense(
        self,
        query_embedding: Sequence[float],
        model_name: str,
        *,
        ticker: str | None = None,
        form_types: Sequence[str] | None = None,
        filed_after: object = None,
        filed_before: object = None,
        limit: int = 50,
    ) -> list[ScoredChunk]:
        query_topic = query_embedding[0]
        candidates = _all_chunks(ticker)
        candidates.sort(key=lambda c: abs(_TOPIC_BY_CHUNK_ID[c.chunk.chunk_id] - query_topic))
        return candidates[:limit]


class FakeEmbeddings:
    async def embed_query(self, text: str) -> list[float]:
        return [_query_topic(text)]

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [[OTHER_TOPIC] for _ in texts]

    @property
    def dimension(self) -> int:
        return 1


class FakeReranker:
    async def score(self, query: str, candidates: list[str]) -> list[float]:
        return [_term_overlap_score(query, candidate) for candidate in candidates]


def recall_at_k(
    retrieved_chunk_ids: list[str], expected_relevant_chunk_ids: frozenset[str]
) -> float:
    if not expected_relevant_chunk_ids:
        return 1.0
    hits = len(set(retrieved_chunk_ids) & expected_relevant_chunk_ids)
    return hits / len(expected_relevant_chunk_ids)
