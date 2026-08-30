# Progress and Resume Point

Living status doc — not a design doc (that's `architecture.md`) and not the build-order doc
(that's `guideline.md`). This tracks where implementation actually is and, for the
milestone in progress, exactly what to do next so work can resume on a different machine without
re-deriving context.

## Milestone status

| Milestone | Status | Notes |
|---|---|---|
| M0 — Skeleton and contracts | **Done** | |
| M1 — Data layer + quant core | **Done** | |
| M2 — Tool layer + MCP surface | **Done** | 18 tools registered, MCP server smoke-tested |
| M3 — Agent loop | **Done** | `make check`: 404 tests, 98-99% coverage |
| M4 — Hybrid verifier (V1-V5) | **Done** | `make check`: 489 tests, 98% coverage. Live-verified end-to-end against real Postgres/yfinance data. Disclosed gap: critic false-positive-rate <8% only mechanically provable (no live `ANTHROPIC_API_KEY`). |
| M5 — Guardrails and RAG | **Done** | 629 unit/e2e/property tests, 96% coverage. Disclosed gap: Postgres-only `FilingsRepository.search_dense`/`search_bm25` integration tests are written but unverified live -- no Docker in this environment (see below). |
| M6 — Observability, evals, hardening | Not started | |
| M7 — Portfolio engine and polish | Not started | |

`make check` (lint + typecheck + arch + test) passes as of end of M5.

## M5 — Guardrails and RAG: completed (2026-08-29)

All 3 stages shipped in one continuous build (Guardrails → RAG ingestion → RAG retrieval+wiring+
evals). `make check`-equivalent (black, ruff, mypy, import-linter,
`pytest tests/unit tests/e2e tests/property`) green at every stage; final: **629 tests, 96%
coverage**.

**Guardrails** (`src/quantagent/guardrails/`, `rules/policy.yaml`) — inbound (rate-limit,
jurisdiction, PII redaction, injection, prohibited-request, scope) + outbound (prohibited
language, advice framing, PII egress, leakage) checks, all deterministic pattern-matching (no LLM
call anywhere in `guardrails/`, enforced by `.importlinter`'s `guardrails-obs-cross-cutting`
contract). Wired into `agent/loop.py`: inbound redacts `question` before `classify_intent` ever
sees it; outbound runs immediately before the single final `yield FinalEvent`, covering both the
normal synthesis path and the FAIL→safe-fallback swap. `apply_disclosures` *replaces*
`AgentAnswer.disclosures` with a code-generated mandatory set rather than appending to the
synthesiser's own free-composed list, closing a guideline.md §9 gap that predated M5. 71 tests,
100% coverage on every guardrails module.

**RAG ingestion** (`data/providers/edgar.py`, `data/providers/embeddings.py`, `rag/chunk.py`,
`data/models.py::Filing`/`FilingChunk`, `migrations/versions/0003_...`,
`data/repositories/filings_repository.py`, `rag/ingest.py`, `scripts/ingest_filings.py`) — direct-
httpx EDGAR client (ticker→CIK, submissions, primary-document fetch, cached, retries only on
429/5xx); `sentence-transformers` bi-encoder embeddings (CPU, lazy-loaded singleton); pure
item-aware chunking (last-occurrence-wins TOC disambiguation so a Table-of-Contents reference
never gets mistaken for the real section heading); pgvector HNSW + a raw-DDL generated `tsvector`
column (deliberately not ORM-mapped, so sqlite unit fixtures keep working). `FilingsRepository`
takes no `tenant_id` and does not inherit `RepositoryBase` — SEC filings are public and identical
for every tenant, so a validated-but-ignored parameter would be an attractive nuisance implying an
isolation guarantee that doesn't exist.

**RAG retrieval + wiring + evals** (`data/providers/reranker.py`, `rag/fusion.py`,
`rag/retrieval.py::HybridRetriever`, `agent/document_index.py::LedgerDocumentIndex`,
`tools/research.py`, `agent/intent.py`'s `RESEARCH` label, real `[RETRIEVED]` prompt block) —
hybrid BM25+dense search fused by RRF, cross-encoder reranked, top-k truncated with a min-score
floor; `DocumentIndex` (the verifier's citation-checking interface, from M4) is built from the
`Ledger` itself rather than re-querying the repository, preserving architecture.md §9.2's
bit-for-bit trace replay; injection screening on every retrieved chunk's full text happens in
`tools/research.py` (not `rag/`, which cannot import `guardrails/`). `retrieve_company_filings`/
`retrieve_filing_section` are real; `search_recent_news`/`get_earnings_transcript_snippets` are
permanently-degraded stubs (no data source configured), registered so the planner's tool catalogue
still resolves them by name. Wired into the real app in `api/app.py`/`api/deps.py` and
`tools/mcp_server.py` — the embedding/reranker model wrappers are constructed once and shared
across requests, never rebuilt per call, since they lazily cache loaded weights.

**Eval coverage added under `tests/unit/evals/`:** `test_recall_at_k.py` (algorithmic recall@8
through the real `HybridRetriever` against a hand-labelled fake corpus — validates fusion/filter/
rerank/truncation logic, not real semantic quality; no numeric recall@8 threshold is mandated
anywhere in the docs, so the asserted threshold is a placeholder pending real calibration, not an
architecture-mandated figure); `test_citation_precision_filings.py` (closed a real pre-existing
gap — M4's golden set never exercised `kind="filing"` Evidence; one fixture per §7.4 sub-check).
The injection-quarantine adversarial case lives in
`tests/unit/tools/test_research.py::test_retrieve_company_filings_quarantines_injected_chunk`
rather than duplicated into `evals/`.

**Disclosed gaps (environment-only, not logic gaps):**
- `FilingsRepository.search_dense`/`search_bm25`'s Postgres integration tests
  (`tests/integration/data/repositories/test_filings_repository_pg.py`) are written correctly but
  unverified live — no Docker available in this environment, same pattern as every other
  `tests/integration/.../*_pg.py` file in this repo.
- Over-refusal for `RESEARCH`: only a mechanical test that the label flows through
  `classify_intent`'s finalize logic correctly exists — a real over-refusal measurement needs a
  live model, same disclosed limitation as M4's critic false-positive-rate.
- A non-CI-gating benchmark of real retrieval quality (real `sentence-transformers` models against
  a larger real-filing corpus) was not built — recall@8 today is validated algorithmically only.

**Notable bug caught during implementation:** `.importlinter`'s `data-purity` contract caught a
real layering violation — `FilingsRepository` (in `data/`) briefly imported `rag.chunk.Chunk`.
Fixed via a `data/`-local `NewChunk` write-side type; `rag/ingest.py` converts at the call site.

## M6 — Observability, evals, hardening: not started

### Scope (architecture.md §9; guideline.md's M6 milestone)

Full trace persistence and the trace-viewer endpoint (`GET /v1/traces/{id}`), hash-chained
append-only audit log, all SLIs exported (§9.3: hallucinated_number_rate, verifier_fail_rate,
repair_success_rate, safe_fallback_rate, citation_precision, tool_error_rate,
schema_violation_rate, confidence_calibration_brier, latency/cost percentiles), alert rules (§9.4),
the nightly reproducibility replay (`make replay TRACE=...` reproducing every `MetricValue`
bit-for-bit from a pinned data snapshot), confidence calibration measurement (reliability curve +
Brier score against the eval set), a load test, and full CI-wiring of the §10.4 gate table
(currently exercised only via `tests/unit/evals/` per M4/M5's precedent — `evals/README.md`
explicitly deferred the CI-wiring itself to M6).

**DoD (guideline.md):** `GET /v1/traces/{id}` renders a complete audit view; `make replay
TRACE=...` reproduces metrics bit-for-bit; all §10.4 gates wired into CI; calibration curve and
Brier score reported; a committed eval scorecard.

### Not yet started — nothing to resume beyond this section

No exploration or design work has been done for M6 yet. When picking this up: read
architecture.md §9 (Observability and Governance) and §10.4 (CI gates) in full, and check
`src/quantagent/obs/` (`logging.py`, `tracing.py` — currently thin OTel/structlog bootstrap from
M0) for what already exists versus what M6 needs to add.
