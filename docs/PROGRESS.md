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
| M5 — Guardrails and RAG | **Planning paused** (see below) | |
| M6 — Observability, evals, hardening | Not started | |
| M7 — Portfolio engine and polish | Not started | |

`make check` (lint + typecheck + arch + test) passes as of end of M4.

## M5 — Guardrails and RAG: current state

**Phase 1 (exploration) is complete.** Phase 2 (design — spawning parallel Plan agents per
subsystem, the pattern used successfully for every milestone through M4) has **not started yet**,
paused by explicit request to move the embeddings decision (below) to a machine with more compute.

### Scope (architecture.md §4.7, §8, §11.3; guideline.md §9)

- **Guardrails:** inbound (scope classification, prohibited-request types, PII detection/redaction,
  prompt-injection screening on user input, rate/budget limits, jurisdiction check) + outbound
  (prohibited language, advice framing, mandatory disclosures, PII egress, leakage) + refusal
  templates + `rules/policy.yaml`.
- **RAG:** EDGAR filing ingestion with item-aware chunking (never splits Item 1A/Item 7),
  Postgres+pgvector HNSW index, hybrid retrieval (BM25 `tsvector` ∪ dense, RRF fusion,
  cross-encoder rerank top-50→top-8), freshness as a hard pre-retrieval filter, source tiering
  (T1-T4), verbatim-excerpt offsets, injection classifier on every retrieved chunk (quarantine +
  log).

**DoD:** adversarial suite 100% safe behaviour; injection payloads in retrieved chunks quarantined
and logged; citation precision ≥0.98; over-refusal slice within threshold; retrieval quality
measured (recall@8 on a labelled set).

### Architectural constraints already confirmed (binding — do not redesign these)

1. **`guardrails/` cannot import `quantagent.llm`** (`.importlinter`'s `guardrails-obs-cross-cutting`
   contract). Combined with `guideline.md` §9 ("each check is a pure function `(payload, context)
   → GuardrailDecision`"; "deterministic layers must not call an LLM"), **every guardrail check,
   including the injection classifier, must be deterministic pattern-matching**, never an LLM call.
   Patterns live in `rules/policy.yaml` (path already reserved in `rules/README.md`).
2. **`rag/` may import only `contracts` + `data`** — not `llm`. Embedding generation must go
   through a `data/providers/` adapter (I/O layer, like `YFinancePriceProvider`), not
   `llm/client.py`'s Anthropic wrapper.
3. **Layering:** `api → agent → tools → rag → data → quant → contracts`. A RAG tool
   (`retrieve_company_filings` etc.) lives in `tools/`, calling into `rag/` — `agent/` never calls
   `rag/` directly.
4. **`verify/citation.py`'s `DocumentIndex` Protocol already exists** (built in M4, real code) and
   is exactly what M5's RAG retrieval must implement from the outside (dependency inversion, since
   `verify/` can't import `rag/`):
   ```python
   class DocumentIndex(Protocol):
       def get_metadata(self, document_id: str) -> DocumentMetadata | None: ...
       def get_chunk_text(self, document_id: str) -> str | None: ...
       def resolves_source_url(self, document_id: str, source_url: str) -> bool: ...
   ```
   `run_v3_checks` already implements every §7.4 sub-check for real against this Protocol — nothing
   to change there. M5 just needs a concrete implementation wired into `run_verification`'s
   `document_index=` param (currently always `None` in `agent/loop.py`).

### Existing scaffolding to plug into (don't rebuild)

- `contracts/errors.py` already declares `GuardrailError`/`OutOfScopeError`/`ProhibitedRequestError`/
  `InjectionDetectedError` — unused so far; M5's guardrails module is the first consumer.
- `prompts/synthesis/answer.v1.jinja` has a `[RETRIEVED]` section today rendering a static
  placeholder string — needs to become real conditional Jinja. `Evidence`'s `excerpt`/`char_span`/
  `source_url`/`source_tier`/`retrieval_score` fields already exist, unused — no contract change
  needed, just population.
- `agent/synthesizer.py`'s `SynthesisInput` needs a new `retrieved` field, threaded through to the
  template render call.
- `agent/intent.py`'s `IntentLabel` is `Literal["SIMPLE_LOOKUP", "PORTFOLIO_ANALYSIS",
  "OUT_OF_SCOPE"]` — `RESEARCH` was deliberately dropped pending RAG (see the docstring). Adding it
  touches: the literal, `prompts/intent/classify.v1.jinja`'s rules, and a new branch in
  `agent/loop.py::_run_agent_loop_inner` (currently only branches on `OUT_OF_SCOPE`/
  `SIMPLE_LOOKUP`, else falls through to `PORTFOLIO_ANALYSIS`).
- Guardrail insertion points in `agent/loop.py::_run_agent_loop_inner`: **inbound** at the very
  top, before `classify_intent` (PII must be redacted before the question reaches any LLM call) —
  same "yield `FinalEvent` and return early" shape already used for the `OUT_OF_SCOPE` refusal.
  **Outbound** after `_synthesize_verify_repair` resolves, right before the final
  `yield FinalEvent(answer=answer)`.
- `rag/__init__.py` and `guardrails/__init__.py` are empty (0 bytes) — clean slate.
- `pgvector` and `sec-edgar-downloader` are already pinned dependencies but unused anywhere in
  `src/`. `config.py`'s `Settings.sec_user_agent` exists but nothing reads it yet. No embeddings
  library, no `rank-bm25`, no HTML/filing-parsing library, no reranker library exists yet.

### Open decision — paused here

Which embedding approach to use for hybrid dense retrieval:
- **sentence-transformers (local)** — real semantic embeddings + a real cross-encoder reranker,
  fully offline/deterministic, no new API key. Adds ~200-300MB CPU-only `torch` dependency.
- **fastembed (local, lighter)** — ONNX-runtime based, no torch, smaller footprint; less
  battle-tested in this ecosystem, thinner reranker story.
- **TF-IDF/hashing pseudo-embeddings (scikit-learn, zero new deps)** — reuses an existing
  dependency, but "dense" becomes highly correlated with BM25 rather than a complementary signal;
  weakens the hybrid design and recall@8 quality. Would be a disclosed scope compromise.

**Decide this first when resuming**, then proceed to Design phase.

### Next steps (Design phase — not yet started)

Spawn 3 parallel Plan agents (the pattern used for every milestone through M4 — each design
sub-plan given the others' target interfaces up front, then reconciled explicitly before
implementation):

1. **Guardrails** — inbound + outbound + injection classifier (deterministic; shared machinery
   between input-screening and RAG-chunk-screening) + `rules/policy.yaml` + refusal templates +
   disclosure assembly. Self-contained (only imports `contracts`).
2. **RAG ingestion** — EDGAR provider (`data/providers/edgar.py`; use `factors.py`'s
   httpx-based-provider pattern for test mocking via `respx`, not yfinance's curl_cffi pattern
   which `vcrpy` can't mock), item-aware chunking, embedding provider (per the decision above),
   pgvector schema + Alembic migration (follow `data/models.py`'s `Mapped`/`mapped_column`
   conventions), filings/chunks repository.
   **Open question to resolve here:** every existing repository method calls
   `RepositoryBase._require_tenant` (I9) even for tenant-agnostic children, via a join to the
   tenant-owning parent — but SEC filings are public documents, not tenant-owned. Whether a
   filings/chunks repository should still take a `tenant_id` it structurally ignores (interface
   uniformity) or explicitly break from `RepositoryBase`'s pattern is unresolved — flag explicitly.
3. **RAG retrieval + wiring + evals** — hybrid BM25+dense+RRF+rerank, freshness filter, source
   tiering, a concrete `DocumentIndex` implementation, `RESEARCH` intent wiring, new RAG tools
   registered in `tools/registry.py` (recommend scoping the real build to
   `retrieve_company_filings`+`retrieve_filing_section` — what architecture.md §16's worked
   example actually exercises; `search_recent_news`/`get_earnings_transcript_snippets` have no
   configured data source and should likely degrade gracefully with a documented limitation,
   mirroring how M4 handled rules R-005/R-009 — confirm this scope call rather than assuming it),
   `agent/loop.py` guardrail+RAG wiring, and the eval harness (adversarial suite, citation
   precision, over-refusal slice, recall@8) under `tests/unit/evals/` (real pytest assertions
   exercised by `make check` today), matching M4's already-established precedent — full CI-wiring
   of `evals/` itself stays M6 scope per `evals/README.md`.

### Full exploration detail

The complete fact-finding from 3 background Explore agents (exact file contents, function
signatures, conventions for providers/repositories/models/migrations/tools/prompts) is preserved
in this project's Claude memory (`project_ai_quant_agent_m5_inprogress.md`) — re-read the specific
source files listed above directly if that memory isn't available in a future session, rather than
re-running exploration from scratch.
