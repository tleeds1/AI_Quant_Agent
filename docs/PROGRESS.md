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
| M5 — Guardrails and RAG | **Done** | 629 unit/e2e/property tests, 96% coverage. `FilingsRepository.search_dense`/`search_bm25` integration tests verified live 2026-08-31 (see below) — no longer a disclosed gap. |
| M6 — Observability, evals, hardening | **Partial** | Trace persistence + hash-chained audit log + `GET /v1/traces/{id}` + `scripts/replay_trace.py` + `evals/run_scorecard.py` exist and pass their own tests. NOT done: alert rules, full §10.4 CI-wiring, calibration/Brier score, load test. Newly found 2026-08-31: the outer exception-handler exit path never persists a trace — see below. |
| M7 — Portfolio engine and polish | **Partial** | CVXPY optimizer (`quant/optimization.py`) + `optimize_portfolio`/`simulate_trade_impact` tools + `scripts/demo_rebalance.py` + `docs/adr_retrospective.md` exist and pass their own tests; `demo_rebalance.py` verified live against real Postgres+yfinance data 2026-08-31. NOT done: README requirements fully met (see disclosed issue below), a 5-minute-quickstart verification. See below. |

`make check` (lint + typecheck + arch + test) passes as of 2026-08-31: **651 tests, 96% coverage**
(unit/e2e/property; +25 more in `tests/integration`, verified live against real Docker infra the
same day, see the new section below). The M6/M7 quality-gate debt described below (formatting,
lint, `B904`/`F821`/broad-except issues) has been fixed; Citation Precision in the scorecard/README
is now a real computed number (was fabricated); the Makefile's
`eval`/`eval-tools`/`eval-adversarial`/`replay` targets are wired to the scripts that already
existed but weren't invoked. Not committed yet — still needs a decision on git history/authorship
before committing (see the chat record for that discussion).

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
- ~~`FilingsRepository.search_dense`/`search_bm25`'s Postgres integration tests unverified live~~ —
  closed 2026-08-31, see the new section below.
- Over-refusal for `RESEARCH`: only a mechanical test that the label flows through
  `classify_intent`'s finalize logic correctly exists — a real over-refusal measurement needs a
  live model, same disclosed limitation as M4's critic false-positive-rate.
- A non-CI-gating benchmark of real retrieval quality (real `sentence-transformers` models against
  a larger real-filing corpus) was not built — recall@8 today is validated algorithmically only.

**Notable bug caught during implementation:** `.importlinter`'s `data-purity` contract caught a
real layering violation — `FilingsRepository` (in `data/`) briefly imported `rag.chunk.Chunk`.
Fixed via a `data/`-local `NewChunk` write-side type; `rag/ingest.py` converts at the call site.

## Uncommitted M6/M7 work (2026-08-30) — done outside this assistant's sessions, audited now

Between the "M5 finished" commit (`af8ac68`, authored by the repo owner) and now, substantial new
code appeared on disk that this assistant did not write and has no prior conversation record of —
audited in this session by reading the files and actually running the tests/scripts. It is real,
functioning work, not vaporware, but it has **not** been through this project's own `make check`
gate yet and has a few rough edges. Recorded here so it isn't lost and so the next session (any
author) knows exactly what's outstanding.

**What's real and working:**
- `quant/optimization.py::optimize_portfolio` — genuine CVXPY convex optimization (min-variance,
  max-utility, risk-parity via a log-barrier formulation), constraint-tested via Hypothesis
  property tests (`tests/property/test_optimization_properties.py`).
- `tools/optimization_tools.py` — `optimize_portfolio`/`simulate_trade_impact` MCP tools wrapping
  the above, registered and tested.
- `data/repositories/trace_repository.py` + `migrations/versions/0004_...` + `data/models.py`
  additions — real `Trace`/`AuditLogEntry` tables, a genuine SHA-256 hash-chain
  (`previous_hash`/`hash`) matching architecture.md §9.5's "tamper-evident" requirement.
  `agent/loop.py::_persist_trace` calls it from every exit path of the loop.
- `api/routes/traces.py::GET /v1/traces/{id}` — real, tenant-scoped, wired into `api/app.py`.
- `scripts/replay_trace.py` — loads a saved trace, re-executes its plan, diffs metric values
  bit-for-bit (architecture.md §9.2). Needs a real Postgres with a saved trace to actually run —
  unverified live in this environment (no Docker).
- `evals/run_scorecard.py` + `evals/scorecard.json`/`.md` — runs the same 8 golden fixtures
  M4 already had (duplicated into `evals/fixtures.py` from `tests/unit/evals/fixtures.py`) and
  writes a committed scorecard. Confirmed by reading the script: it only computes
  precision/recall on those 8 hand-crafted mocked fixtures (verifier-logic correctness against a
  controlled set) — **the same disclosed M4 caveat applies** (not a live-model measurement).
- `scripts/demo_rebalance.py` + the README's "Worked Example" section — **actually ran this in
  session; output matches the README verbatim, byte for byte.** Falls back to a synthetic/offline
  price panel when Postgres isn't reachable (it wasn't, here) — the printed "As of: 2021-11-30" is
  that synthetic panel's date, not live market data. The script also imports test-double builders
  (`tests/unit/tools/builders.py`) as its offline fallback path, which is unusual for something
  under `scripts/` (test doubles are normally test-only) — worth tidying, not a correctness bug.
- `docs/adr_retrospective.md` — reasonable, accurate summary of 4 real ADRs.

**⚠️ Fixed (2026-08-30), was a real problem — the README's "Reliability Scorecard" table
presented four numbers with no computation behind them anywhere in the codebase:** Tool-Selection
F1 (98.2%), Citation Precision (99.4%), p95 Latency (3.8s), Cost per Request ($0.015), each marked
"PASS" against a target as if independently audited. Only the Hallucinated-Number-Rate row (0.00%)
was real. Resolved as follows, not by inventing the missing numbers:
- **Citation precision is now real**: `evals/citation_fixtures.py` (5 hand-labelled `kind="filing"`
  Evidence cases — clean, fabricated, excerpt-mismatch, wrong-url, wrong-tier; M4's golden set
  never exercised this kind of Evidence at all) + `tests/unit/evals/citation_harness.py`
  (deterministic, no LLM call — V3 never calls a model) + `evals/run_scorecard.py` now reports it
  for real: **precision=1.000, recall=1.000 on 5 fixtures**, written into `evals/scorecard.md`.
  Same disclosed caveat as the rest of the golden set applies (hand-curated, unambiguous ground
  truth — proves verifier-logic correctness, not real-world performance against live model output).
- **Tool-selection F1**: `evals/eval_tool_selection.py` + `evals/tool_selection_fixtures.py` (8
  golden traces, real `classify_intent`/`create_plan` calls, exactly mirroring
  `agent/loop.py`'s own branching) now exist and are runnable (`make eval-tools`) — but genuinely
  need a live `ANTHROPIC_API_KEY`, which isn't configured in this environment, so no number is
  reported here. Run it yourself to get the real one.
- **p95 latency / cost per request**: no bespoke harness was built — a correct measurement needs
  real end-to-end traffic (live key + real Postgres/Redis/yfinance), and `agent/loop.py::
  _persist_trace` already faithfully records `latency_ms`/`cost_usd` per LLM and tool call in the
  persisted trace. Reading those after running real traffic is more honest than a parallel
  re-implementation that could silently diverge from what actually ships.
- README's table and Makefile's `eval-latency` target updated to state this plainly instead of
  showing a placeholder number.

**Quality-gate debt — fixed (2026-08-30):** `black`/`ruff` reformatted/fixed; `quant/
optimization.py`/`tools/optimization_tools.py`'s bare `except Exception as e: raise ...` now use
`raise ... from e`; `tests/property/test_optimization_properties.py`'s property test now narrows
its `except Exception: assume(False)` to `except OptimizationError` specifically (was silently
discarding a Hypothesis-generated case on *any* exception, including a genuine optimizer bug) and
imports `pandas as pd` (was an undefined-name, previously masked by a blanket
`# mypy: ignore-errors` at the top of the file, now removed). The Makefile's
`eval`/`eval-tools`/`eval-adversarial`/`replay` targets are now wired to the scripts that already
existed. `make check` verified green: 652 tests, 96% coverage.

## LLM backend swap + first live-infra run (2026-08-31)

**LLM backend swapped from the `anthropic` SDK to a self-hosted, OpenAI-Chat-Completions-compatible
endpoint** (a company Open WebUI instance proxying a local Ollama model, `gemma4:26b`): the
Anthropic-Messages-compatible route (`/v1/messages`) returned 403 on the available API key (an
Open WebUI permission the account doesn't have), while `/api/chat/completions` (OpenAI-compatible)
worked. Rather than wait on an admin grant, `llm/client.py` was rewritten to speak that protocol
directly over plain `httpx` (matching the rest of the codebase's provider pattern, e.g.
`data/providers/edgar.py`) instead of the `anthropic` SDK — `LLMClient` replaces `AsyncAnthropic`
everywhere (9 call sites, mechanical type-hint swap only); forced structured-output now uses
OpenAI's `tools`/`tool_choice` function-calling shape instead of Anthropic's `tool_choice={"type":
"tool"}`; the temperature→reasoning-`effort` workaround is gone since this endpoint accepts real
`temperature` directly. `contracts/errors.py` gained `LLMTransportError` for transport/HTTP
failures, separate from `StructuredOutputError` (schema failures). Test fixtures
(`tests/unit/llm/fixtures.py`) now use plain `httpx.MockTransport` (simpler than the old
`httpx2.MockTransport` shim the Anthropic SDK required). `make check` equivalent verified green
after the swap: black/ruff/mypy/import-linter/**651 tests, 96% coverage**.
Config: `Settings.anthropic_base_url` (new field, empty by default) — set alongside
`ANTHROPIC_API_KEY` in `.env` to route through an OpenAI-compatible proxy instead of Anthropic's
API directly; see `.env.example`'s comment block for the exact Open WebUI wiring.

**Docker Desktop set up and used for the first time this project has ever run against real
infrastructure.** Sequence: `docker compose up -d` (pgvector/pg16 + redis:7-alpine, both healthy)
→ `alembic upgrade head` (all 4 migrations, including the two — `0003_create_filings_and_chunks`,
`0004_create_traces_and_audit_log` — that had never been run against a real Postgres before) →
`scripts/seed_portfolio.py`. Found and fixed one real Windows-specific bug this surfaced:
**`REDIS_URL=redis://localhost:...` timed out** even though the container was healthy and reachable
via raw TCP — Windows resolves `localhost` to the IPv6 loopback (`::1`) first, which Docker
Desktop's port forwarding doesn't listen on; `redis-py`'s async client doesn't fall back to IPv4
quickly the way `asyncpg` apparently does. Fixed by pinning `REDIS_URL=redis://127.0.0.1:...` in
`.env`/`.env.example`, with the reasoning left as a comment so it isn't rediscovered blind.

**Results, now genuinely verified live (not just "written correctly"):**
- `scripts/demo_rebalance.py` ran against the **real** seeded Postgres portfolio and **real**
  yfinance price history (not the offline/synthetic fallback the README's Worked Example section
  documents) — real numbers, real `as_of` date, confirming the real-data path in the README's
  Quickstart §3 genuinely works end-to-end.
- `tests/integration/` (25 tests across `data/providers/`, `data/repositories/`, `data/
  test_cache_redis.py`, `scripts/test_print_risk_report_smoke.py`, `tools/test_mcp_server.py`,
  `tools/test_tools_end_to_end.py`) — **all pass against real Docker Postgres/pgvector/Redis**,
  first time ever. One real gap found and fixed in the process:
  `tests/integration/tools/test_tools_end_to_end.py`'s `_ARGS_BY_TOOL` dict (asserts it covers
  every registered tool) had never been updated for the 6 tools M5/M6/M7 added
  (`optimize_portfolio`, `simulate_trade_impact`, `retrieve_company_filings`,
  `retrieve_filing_section`, `search_recent_news`, `get_earnings_transcript_snippets`) — this test
  had simply never run before, since it needs real Postgres. Fixed by adding args for all 6 (the
  two RAG tools deliberately exercise their "`ctx.retrieval` not configured" fallback path, since
  seeding real filing chunks is out of scope for a registry-wiring test).
- A real, full `/v1/analyze` request was run against the real LLM (`gemma4:26b` via Open WebUI),
  real Postgres, real Redis. **It did not produce a successful analysis** — `gemma4:26b`'s forced
  tool-call output for the DAG-planning stage (`agent/planner.py`'s `Plan` schema, several nested
  steps) was schema-invalid on both the first attempt and the one permitted retry (observed mangled
  field names, e.g. `tool_args` instead of `args`), so the system correctly raised
  `StructuredOutputError` rather than accept malformed output. This is a genuine finding about this
  particular local model's structured-output reliability on a moderately complex schema, not a bug
  in the swap above — the same model *did* succeed on simpler schemas (a smoke-tested trivial tool
  call, and `classify_intent`'s simpler schema succeeded in one run). A second finding, real and
  worth fixing deliberately rather than patched in passing: **`run_agent_loop`'s outer fail-closed
  exception handler (`agent/loop.py` ~line 98) does not call `_persist_trace` before yielding the
  unrecoverable-error answer** — every other exit path does (guardrail block, OUT_OF_SCOPE, missing
  direct_tool, and the normal RELEASE/SAFE_FALLBACK path all persist), but an unhandled exception
  mid-loop (exactly what a schema-invalid planner output produces) currently leaves **zero trace
  record** for that request, contradicting architecture.md §9's "every request produces a complete,
  tamper-evident trace" intent. Confirmed by direct `TraceRepository.get_trace` lookups for both
  real trace_ids generated above — neither was found. Not fixed this session (needs a deliberate
  decision on what a best-effort trace write from inside an exception handler should look like, not
  a rushed patch) — next session should treat this as a real M6 item, not a "nice to have".
- `scripts/replay_trace.py` remains unverified live — no successful (or even persisted) trace was
  produced this session to replay against, for the reason above.

**Tool-Selection F1 (`make eval-tools`) measured live, closing the README's last fabricated-metric
gap, then improved based on what the first run showed:**
- First run: **93.3%** aggregate (target ≥ 90.0%, architecture.md §10.4) over the 8 golden traces,
  but every single trace needed its one permitted schema-retry before succeeding — a 100% retry
  rate is a real reliability signal worth investigating, not something to wave off just because the
  aggregate cleared target. The dominant recurring error: `_IntentClassification.rationale` (a
  Pydantic `max_length=280` field) came back over-length repeatedly — the prompt
  (`prompts/intent/classify.v1.jinja`) never told the model this in plain language, only via the
  JSON schema's `maxLength`, which `gemma4:26b` doesn't reliably respect the way Claude does.
- Fixed `prompts/intent/classify.v1.jinja` (explicit rule: keep `rationale` to one short sentence)
  and `prompts/planner/dag.v1.jinja` (added a worked JSON example of a correctly-shaped
  multi-step plan — the other recurring error was steps missing `args` or `depends_on`, and this
  model benefits from a concrete shape to copy, not just an abstract schema description).
- Re-running exposed a second, separate real bug: `evals/eval_tool_selection.py` let one trace's
  unrecoverable `StructuredOutputError` crash the entire script instead of recording that trace as
  a genuine F1=0 miss and continuing — so one bad trace destroyed the whole run's signal. Fixed by
  wrapping `_tools_chosen_for` in a `try/except LLMError` per trace.
- Final verified run (both prompt fixes + the harness fix): **91.7%** aggregate, all 8 traces
  completed without crashing. The confirmed-fixed `rationale`-length error did not recur in either
  re-run. The number moving 93.3% → 91.7% between runs is real LLM non-determinism (this model, at
  `temperature=0.0`), not a regression — both clear target. README's Reliability Scorecard updated
  with the real number and this full caveat (model-dependent, non-deterministic, what was fixed).

**Chasing one full, successful `/v1/analyze` run (2026-08-31, later same day) — real progress, not
yet a clean success:**
- Root-caused why every real request so far had hit `SAFE_FALLBACK`: `.env`'s `MAX_WALL_MS=12000`
  (architecture.md's 9s target, for a fast provider) cut tool execution off before this slow local
  model's own LLM calls (20-90s+ each) even returned, so `execute_plan` aborted mid-DAG on budget
  exhaustion and SYNTHESIZE got an empty ledger — a guaranteed FAIL. Not a logic bug; a config
  mismatched to this model. Bumped to `MAX_WALL_MS=300000` for this model (documented in `.env`/
  `.env.example` as model-specific, to be lowered against a faster provider).
  Re-running a `SIMPLE_LOOKUP` question (`"How many holdings are in my portfolio?"`, chosen to skip
  the multi-step DAG planner entirely) then got past PLAN and EXECUTE for real (`get_holdings`
  really ran against real Postgres, `status: OK`) — first time any real request got that far.
- New failure surfaced at SYNTHESIZE: the model answered with a `` ```json ... ``` `` fenced code
  block in plain `content` instead of calling the forced `emit_structured_output` tool at all —
  `gemma4:26b` doesn't reliably honor `tool_choice` on a long generation (13k+ char system prompt,
  ~5-6k output tokens). Fixed generally in `llm/client.py::_parse_response`: when no matching
  `tool_calls` entry exists, `_extract_json_from_text` now recovers a JSON object from `content`
  (fenced block preferred, else the whole trimmed string) before falling through to the original
  "no tool call" failure — a real robustness improvement to the shared primitive, not a one-off
  patch, covered by a new unit test (`test_recovers_json_written_as_fenced_text_instead_of_a_tool_call`).
- With that fix, the next re-run got further still: the fenced JSON was recovered, but it failed
  **schema** validation (7 errors) against `_DraftAnswer` — missing `claim_id` on `claims[0]` and
  five required fields on `evidence[0]` (`evidence_id`, `char_span`, `source_url`, `source_tier`,
  `published_at`, `retrieval_score`). Unlike the planner's `Plan` schema (fixed earlier with a
  worked example), the synthesis stage's schema is considerably larger (claim-evidence citation
  linking) and `prompts/synthesis/answer.v1.jinja` has no worked example either — the likely next
  fix, not yet made. **Stopped here rather than keep iterating unprompted**: each attempt costs
  ~2-3 real minutes against this model, and this is a genuinely deep schema for a 26B local model
  writing free-form JSON (not going through real function-calling machinery at this point, since
  `tool_choice` isn't being honored) to get exactly right.
- Added a worked-example `_DraftAnswer` JSON block to `prompts/synthesis/answer.v1.jinja` (same
  technique that fixed the planner), showing every `Evidence` field explicit-even-when-null and the
  full nested `MetricValue`/`Provenance` shape. Re-running: attempt 4 got the schema-retry down from
  7 validation errors to **1** (`published_at` missing on the retry too); attempt 5 produced a fully
  schema-valid `_DraftAnswer` on the first try — **SYNTHESIZE succeeded for the first time against
  a real request.**
- That draft then correctly **failed VERIFY** (verdict FAIL, 14 checks run — V4's 10 constraint
  rules all PASS/NOT_APPLICABLE, so the failure is a V1/V2/V3 hard-stop) and correctly triggered
  SAFE_FALLBACK rather than release an answer with a claim/evidence mismatch (the model's one claim
  listed the portfolio's tickers but cited a `market_value_AAPL` metric evidence entry that doesn't
  actually substantiate that claim). The exact failing check wasn't root-caused (verify/ doesn't log
  per-check detail below WARN; would need reading the persisted trace, which needs the crash-path
  trace-persistence gap above fixed first) — but this is now a **complete, honest demonstration of
  the full architecture working end-to-end for real**: a real (if imperfect) model produced a
  plausible-looking but subtly ungrounded answer, and the verifier caught it and refused to release
  it, exactly per architecture.md §7's design intent. This is arguably more convincing evidence of
  the system's core value proposition than a clean success would have been.
- Net result: no `/v1/analyze` response has reached RELEASE yet (every real run so far ends in
  SAFE_FALLBACK), but the pipeline now demonstrably runs every stage for real — INTAKE → PLAN →
  EXECUTE (real Postgres data) → SYNTHESIZE (real schema-valid output) → VERIFY (real check
  failure, correctly caught) → SAFE_FALLBACK — and three of the four failures found along the way
  were real, generally-useful fixes, not workarounds specific to one question: the `MAX_WALL_MS`
  config note, the `_extract_json_from_text` fallback, and the synthesis worked example all help any
  future request against a slow/imperfect model.

## M6 — Observability, evals, hardening: partial (see above)

### Scope (architecture.md §9; guideline.md's M6 milestone)

Full trace persistence and the trace-viewer endpoint (`GET /v1/traces/{id}`) — **done**. Still
outstanding: all SLIs exported (§9.3: hallucinated_number_rate, verifier_fail_rate,
repair_success_rate, safe_fallback_rate, citation_precision, tool_error_rate,
schema_violation_rate, confidence_calibration_brier, latency/cost percentiles), alert rules (§9.4),
confidence calibration measurement (reliability curve + Brier score against the eval set), a load
test, and full CI-wiring of the §10.4 gate table (currently exercised only via `tests/unit/evals/`
per M4/M5's precedent — `evals/README.md` explicitly deferred the CI-wiring itself to M6).

**DoD (guideline.md):** `GET /v1/traces/{id}` renders a complete audit view — done, but unverified
live (no Docker here). `make replay TRACE=...` reproduces metrics bit-for-bit — script exists,
unverified live. All §10.4 gates wired into CI — not done. Calibration curve and Brier score
reported — not done. A committed eval scorecard — done (`evals/scorecard.md`), now with a real
citation-precision number alongside the golden-set precision/recall (see above for exactly what
it does and doesn't prove).

### Resuming M6

Read architecture.md §9 (Observability and Governance) and §10.4 (CI gates) in full. Check
`src/quantagent/obs/` (`logging.py`, `tracing.py` — thin OTel/structlog bootstrap from M0) for
what exists versus what's still needed: real SLI export (a `/v1/metrics` endpoint or OTel exporter
wired to something that scrapes it), alert rule definitions, calibration/Brier score computation,
and a load test. Fix the quality-gate debt listed above first — cheap, mechanical, and blocks
`make check` today.

## M7 — Portfolio engine and polish: partial (see above)

### Scope (guideline.md's M7 milestone)

Constrained mean-variance/min-variance/risk-parity optimiser — **done** (`quant/optimization.py`,
CVXPY). Rebalance proposals with ex-ante Δrisk — **done** (`optimize_portfolio` tool returns
current/target return/volatility/Sharpe and `ex_ante_delta_risk`). Trade-impact simulation —
**done** (`simulate_trade_impact` tool). A deterministic report artefact — not found; `utility.py`'s
existing `generate_risk_report` (M2) was not extended to include optimizer/rebalance sections.
A demo script — done (`scripts/demo_rebalance.py`, verified to actually run and match the README).
An architecture-decision retrospective — done (`docs/adr_retrospective.md`).

**DoD (guideline.md):** optimiser respects every declared constraint (property-tested) — done.
Rebalance proposals always include ex-ante deltas — done. README lets a new reader run the demo
in under 5 minutes — **not verified**: the README's quickstart doesn't mention `docker compose up
-d`/`make migrate`/`make seed` at all anymore (it was in the pre-M5 README), so a reader following
it literally gets the offline-fallback demo, not the real-data path, with no explanation of that
distinction. The Reliability Scorecard table issue (flagged above) also means the README's opening
claims are not yet trustworthy as written.

### Resuming M7

The core optimizer/tools work is solid and tested. Remaining: fix the quality-gate debt above,
decide what "under 5 minutes, real data" quickstart should look like (with or without Docker), and
either extend `generate_risk_report` to include a rebalance section or explicitly scope that out.
