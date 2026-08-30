"""tools/research.py -- RAG research tools (architecture.md §4.3's Research
catalogue, §4.7, §11.3).

`retrieve_company_filings`/`retrieve_filing_section` are the real build
this milestone (architecture.md §16's worked example is what actually
exercises them). `search_recent_news`/`get_earnings_transcript_snippets`
have no configured data source and are permanently-degraded stubs --
registered so the planner's tool catalogue and any DAG referencing them by
name resolve against a real, known tool rather than failing plan
validation, mirroring how `verify/constraint_rules.py` registers R-005/
R-009 as real, always-`NOT_APPLICABLE` rules rather than omitting them.

This is the one place retrieved chunks are screened for prompt injection
(architecture.md §11.3): `tools/` may import `guardrails/`, unlike `rag/`
(`.importlinter`'s `rag-scope` contract forbids that import), so the
screening call lives here, right after `HybridRetriever.search()` returns
and before a chunk is wrapped into the tool's typed output / written to
the ledger.
"""

from __future__ import annotations

import structlog

from quantagent.contracts.tools import (
    GET_EARNINGS_TRANSCRIPT_SNIPPETS,
    RETRIEVE_COMPANY_FILINGS,
    RETRIEVE_FILING_SECTION,
    SEARCH_RECENT_NEWS,
    GetEarningsTranscriptSnippetsInput,
    GetEarningsTranscriptSnippetsOutput,
    RetrieveCompanyFilingsInput,
    RetrieveCompanyFilingsOutput,
    RetrievedFilingChunk,
    RetrieveFilingSectionInput,
    RetrieveFilingSectionOutput,
    SearchRecentNewsInput,
    SearchRecentNewsOutput,
)
from quantagent.guardrails.injection import classify_injection
from quantagent.rag.retrieval import RetrievalFilters, RetrievedChunk
from quantagent.tools.context import ToolContext
from quantagent.tools.registry import registry

logger = structlog.get_logger(__name__)

_RETRIEVAL_TOP_K = 8


def _screen_and_build_chunks(
    hits: list[RetrievedChunk], *, tool_name: str
) -> tuple[list[RetrievedFilingChunk], list[str]]:
    """Screens each hit's full chunk text (not just its excerpt -- an
    injected instruction could sit outside the excerpt window but still
    poison a future excerpt-selection heuristic). A flagged hit is dropped,
    logged, and turned into a provenance warning (I8: one poisoned chunk
    degrades the result set, it never fails the whole tool call --
    architecture.md §11.3's "layer 2 will never be perfect" defense-in-
    depth framing).
    """
    chunks: list[RetrievedFilingChunk] = []
    warnings: list[str] = []
    for hit in hits:
        verdict = classify_injection(hit.text)
        if verdict.is_injection:
            logger.warning(
                "retrieved_chunk_quarantined",
                tool_name=tool_name,
                chunk_id=hit.chunk_id,
                matched_group_ids=verdict.matched_group_ids,
            )
            warnings.append(
                f"chunk {hit.chunk_id} quarantined by the injection classifier "
                f"({verdict.matched_group_ids}) and excluded from this result."
            )
            continue
        chunks.append(
            RetrievedFilingChunk(
                chunk_id=hit.chunk_id,
                ticker=hit.ticker,
                cik=hit.cik,
                form_type=hit.form_type,
                filed_at=hit.filed_at,
                item=hit.item,
                section_path=hit.section_path,
                excerpt=hit.excerpt,
                char_span=hit.char_span,
                source_url=hit.source_url,
                source_tier=hit.source_tier,
                retrieval_score=hit.retrieval_score,
            )
        )
    return chunks, warnings


@registry.tool(
    name=RETRIEVE_COMPANY_FILINGS,
    description=(
        "Retrieves the most relevant passages from a company's recent SEC filings "
        "(10-K/10-Q/8-K by default) for a given research question. Use for company-specific "
        "research (risk factors, MD&A, business description). Do NOT use for portfolio-level "
        "metrics (use the risk/exposure tools) or for a single named section of one specific "
        "filing (use retrieve_filing_section)."
    ),
    p95_latency_ms=800,
    est_cost_usd=0.0,
    cache_ttl_s=3600,
    side_effects="READ_ONLY",
)
async def retrieve_company_filings(
    inp: RetrieveCompanyFilingsInput, ctx: ToolContext
) -> RetrieveCompanyFilingsOutput:
    if ctx.retrieval is None:
        return RetrieveCompanyFilingsOutput(
            ticker=inp.ticker,
            chunks=[],
            provenance=ctx.build_provenance(
                data_sources=["edgar"],
                warnings=["RAG retrieval is not configured in this environment"],
            ),
        )
    filters = RetrievalFilters(
        ticker=inp.ticker, form_types=inp.form_types, published_after=inp.since
    )
    hits = await ctx.retrieval.search(inp.query, filters, top_k=_RETRIEVAL_TOP_K)
    chunks, warnings = _screen_and_build_chunks(hits, tool_name=RETRIEVE_COMPANY_FILINGS)
    return RetrieveCompanyFilingsOutput(
        ticker=inp.ticker,
        chunks=chunks,
        provenance=ctx.build_provenance(data_sources=["edgar"], warnings=warnings),
    )


@registry.tool(
    name=RETRIEVE_FILING_SECTION,
    description=(
        "Retrieves the most relevant passages from one named section (e.g. Item 1A Risk "
        "Factors, Item 7 MD&A) of a company's specific filing form. Use when the question "
        "names a specific section/form. Do NOT use for a broad, cross-filing research "
        "question (use retrieve_company_filings)."
    ),
    p95_latency_ms=800,
    est_cost_usd=0.0,
    cache_ttl_s=3600,
    side_effects="READ_ONLY",
)
async def retrieve_filing_section(
    inp: RetrieveFilingSectionInput, ctx: ToolContext
) -> RetrieveFilingSectionOutput:
    if ctx.retrieval is None:
        return RetrieveFilingSectionOutput(
            ticker=inp.ticker,
            form=inp.form,
            section=inp.section,
            chunks=[],
            provenance=ctx.build_provenance(
                data_sources=["edgar"],
                warnings=["RAG retrieval is not configured in this environment"],
            ),
        )
    item_token = inp.section.removeprefix("item_").upper()  # "item_1a" -> "1A"
    filters = RetrievalFilters(ticker=inp.ticker, form_types=[inp.form], item=item_token)
    hits = await ctx.retrieval.search(inp.query, filters, top_k=_RETRIEVAL_TOP_K)
    chunks, warnings = _screen_and_build_chunks(hits, tool_name=RETRIEVE_FILING_SECTION)
    return RetrieveFilingSectionOutput(
        ticker=inp.ticker,
        form=inp.form,
        section=inp.section,
        chunks=chunks,
        provenance=ctx.build_provenance(data_sources=["edgar"], warnings=warnings),
    )


@registry.tool(
    name=SEARCH_RECENT_NEWS,
    description=(
        "NOT CONFIGURED in this release -- no live news data source is wired up. Always "
        "returns an empty result with a documented limitation. Do not rely on this tool for "
        "current news; prefer retrieve_company_filings for filing-based research."
    ),
    p95_latency_ms=50,
    est_cost_usd=0.0,
    cache_ttl_s=1800,
    side_effects="READ_ONLY",
)
async def search_recent_news(
    inp: SearchRecentNewsInput, ctx: ToolContext
) -> SearchRecentNewsOutput:
    return SearchRecentNewsOutput(
        tickers=inp.tickers,
        provenance=ctx.build_provenance(
            data_sources=["none_configured"],
            warnings=["no live news data source is configured in this milestone"],
        ),
    )


@registry.tool(
    name=GET_EARNINGS_TRANSCRIPT_SNIPPETS,
    description=(
        "NOT CONFIGURED in this release -- no earnings-transcript data source is wired up. "
        "Always returns an empty result with a documented limitation."
    ),
    p95_latency_ms=50,
    est_cost_usd=0.0,
    cache_ttl_s=1800,
    side_effects="READ_ONLY",
)
async def get_earnings_transcript_snippets(
    inp: GetEarningsTranscriptSnippetsInput, ctx: ToolContext
) -> GetEarningsTranscriptSnippetsOutput:
    return GetEarningsTranscriptSnippetsOutput(
        ticker=inp.ticker,
        provenance=ctx.build_provenance(
            data_sources=["none_configured"],
            warnings=["no earnings-transcript data source is configured in this milestone"],
        ),
    )
