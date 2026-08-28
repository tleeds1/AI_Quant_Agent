# Implementation Guideline — AI Quant / Financial Agent

**Audience:** the coding agent (and humans) implementing this project.
**Companion doc:** `architecture.md` — read it fully before writing any code. This document says
*how to build*; that one says *what to build and why*.

---

## 0. How to use this document

1. Read `architecture.md` end to end. Do not start coding from the diagram alone.
2. Read §1 (Invariants) here. These are hard constraints; a PR that violates one is rejected
   regardless of how well it works.
3. Work milestone by milestone (§11). Do not skip ahead. Each milestone has a
   Definition of Done that must be fully satisfied before the next begins.
4. Before every PR, run the §12 checklist as an independent review pass.

**When something in this guideline conflicts with reality** (a library is unavailable, an API
changed, a design decision turns out wrong): stop, state the conflict explicitly, propose the
change with its trade-off, and get a decision. Do not silently improvise around the architecture —
especially not around §1.

---

## 1. Invariants — non-negotiable

| # | Invariant | Enforcement |
|---|---|---|
| I1 | **The LLM never produces a number.** Every quantitative value reaches the user through a `MetricValue` with a `tool_call_id`. | Output schema + numeric-grounding verifier |
| I2 | **`quant/` is pure.** No I/O, no network, no LLM, no global config reads, no logging of business values. Inputs in, values out. | `import-linter` contract + code review |
| I3 | **Every claim carries ≥1 evidence ID.** | Pydantic `min_length=1` + V1 verifier |
| I4 | **Guardrails and verifiers fail closed.** An exception inside a check blocks the response; it never passes through. | try/except around each check returning `FAIL`, tested |
| I5 | **Retrieved text is data, never instructions.** Always delimited, always classified, never in the instruction region. | Prompt templates + injection classifier + tests |
| I6 | **No execution path.** No broker SDK, no order endpoint, no write path to accounts. | Dependency allowlist + code review |
| I7 | **Every tool result is provenanced.** `as_of`, `inputs_hash`, `data_sources`, sample size. | Tool base class enforces it |
| I8 | **No silent degradation.** Missing data, stale data or a failed tool must appear in `limitations` and cap `confidence`. | Degradation tests |
| I9 | **Tenant scoping is mandatory at the repository layer.** A repository call without `tenant_id` raises. | Repository base class + tests |
| I10 | **Determinism is reproducible.** Fixed seeds, recorded provider responses in tests, `inputs_hash` on every call. | Nightly replay test |

---

## 2. Environment setup

### 2.1 Prerequisites

Python 3.11+, Docker + Docker Compose, `make`, an Anthropic API key, and (optional but
recommended) a market-data provider key. `yfinance` requires no key and is the v1 default.

### 2.2 Bootstrap

```bash
git init ai-quant-agent && cd ai-quant-agent
python -m venv .venv && source .venv/bin/activate
python -m pip install -U pip uv

# dependencies (pyproject.toml, not requirements.txt)
uv pip install -e ".[dev]"

cp .env.example .env    # then fill in keys
docker compose up -d    # postgres+pgvector, redis
make migrate
make seed               # demo portfolio + a small price/filing snapshot
make test
make dev                # uvicorn with reload
```

### 2.3 Dependency baseline

**Runtime:** `fastapi`, `uvicorn[standard]`, `pydantic>=2`, `pydantic-settings`, `sqlalchemy>=2`,
`alembic`, `asyncpg`, `psycopg[binary]`, `pgvector`, `redis`, `httpx`, `tenacity`, `anthropic`,
`numpy`, `pandas`, `scipy`, `scikit-learn` (Ledoit-Wolf), `statsmodels` (HAC standard errors),
`cvxpy` (constrained optimisation), `yfinance`, `sec-edgar-downloader` or direct EDGAR HTTP,
`rank-bm25` or Postgres `tsvector`, `structlog`, `opentelemetry-sdk`, `jinja2`, `pyyaml`.

**Dev:** `pytest`, `pytest-asyncio`, `pytest-cov`, `hypothesis`, `vcrpy`/`respx`, `ruff`, `black`,
`mypy`, `import-linter`, `pre-commit`.

Pin exact versions in a lockfile. Do not add a dependency that duplicates something already
present, and do not add an agent framework (see ADR-002).

### 2.4 `.env.example`

```
ANTHROPIC_API_KEY=
MARKET_DATA_PROVIDER=yfinance
POLYGON_API_KEY=
DATABASE_URL=postgresql+asyncpg://quant:quant@localhost:5432/quantagent
REDIS_URL=redis://localhost:6379/0
SEC_USER_AGENT="YourName your@email.com"     # EDGAR requires a real contact string
MODEL_PLANNER=claude-sonnet-4-6
MODEL_SYNTHESIZER=claude-sonnet-4-6
MODEL_CRITIC=claude-sonnet-4-6
MODEL_INTENT=claude-haiku-4-5
MAX_TOOL_CALLS=12
MAX_WALL_MS=12000
MAX_USD_PER_REQUEST=0.15
LOG_LEVEL=INFO
ENV=local
```

Never commit `.env`. Never log a secret. Never put a key in a prompt.

### 2.5 Makefile targets (implement these)

```
make dev · test · test-unit · test-integration · test-e2e
make lint · fmt · typecheck · arch (import-linter)
make migrate · seed · ingest-filings
make eval · eval-grounding · eval-tools · eval-adversarial · eval-latency
make replay TRACE=tr_xxx
make check      # lint + typecheck + arch + test + eval gates — must pass before any PR
```

---

## 3. Layering rules

```
contracts/  ← imported by everything, imports nothing internal
quant/      ← imports contracts + numpy/pandas/scipy ONLY
data/       ← imports contracts; owns all I/O
rag/        ← imports contracts, data
tools/      ← imports contracts, quant, data, rag   (thin adapters only)
agent/      ← imports contracts, tools, llm, verify, guardrails, obs
api/        ← imports contracts, agent
verify/ guardrails/ obs/  ← cross-cutting; import contracts (+ llm for the critic only)
```

Encode this in `.importlinter` and run it in CI. The forbidden edges that matter most:
`quant → data`, `quant → llm`, `quant → tools`, `tools → agent`.

**Three-layer rule inside every module:** I/O · business logic · orchestration. A data-fetching
function must not compute a risk metric; a quant function must not fetch data; a tool must not
contain a formula.

---

## 4. Coding standards

Baseline: `black` + `ruff`, `snake_case` functions/vars, `PascalCase` classes, `pytest`.
`mypy --strict` on `contracts/`, `quant/` and `verify/`.

### 4.1 Comment discipline

This is the most common failure mode, so it is stated bluntly.

**Do not write** comments that restate the code, comments describing a self-explanatory name,
editing-history comments, or comments that paper over unclear code (rewrite the code instead).
A 100-line file with moderately complex logic should have **0–5 comments**.

**Do write** comments for: the *why* behind a non-obvious choice; a side effect or gotcha not
inferable from the name; a TODO with concrete context (`# TODO: switch to Polygon when daily
requests exceed the yfinance soft limit`); docstrings on public functions describing the
**contract** (inputs, outputs, exceptions, units, and — for quant functions — the sign convention);
and any formula/business rule with a reference.

```python
# BAD
# calculate the VaR
def calculate_var(returns, alpha):
    q = np.quantile(returns, 1 - alpha)   # get the quantile
    return -q                              # make it positive

# GOOD
def calculate_historical_var(returns: NDArray[np.float64], alpha: float) -> float:
    """Empirical 1-period VaR at confidence `alpha`.

    Returns a POSITIVE loss fraction (0.024 == a 2.4% loss), matching the
    convention in `contracts.MetricValue`. Raises InsufficientDataError below
    MIN_VAR_OBSERVATIONS because an empirical quantile on a short sample is
    not a risk estimate.
    """
    if returns.size < MIN_VAR_OBSERVATIONS:
        raise InsufficientDataError(...)
    return float(-np.quantile(returns, 1.0 - alpha))
```

### 4.2 Naming

Business meaning, not type: `active_holdings`, not `holding_list`. Booleans prefixed `is_`/`has_`/
`should_`. Functions are verb+noun: `calculate_component_var`, `fetch_filing_chunks`. No
`data`/`temp`/`result`/`utils.py`-as-junk-drawer. Domain terms match finance usage exactly — do not
invent a synonym for "drawdown".

### 4.3 Structure limits

Functions ≤ 40 lines, classes ≤ 300 lines, nesting ≤ 3 levels (use guard clauses and early
returns). No magic numbers in logic: `TRADING_DAYS_PER_YEAR = 252`,
`Z_SCORE_95 = 1.6449`, `MIN_VAR_OBSERVATIONS = 250`, `CORRELATION_SHRINKAGE = "ledoit_wolf"`
live in `quant/constants.py` or config.

### 4.4 Errors and logging

Catch specific exceptions. No bare `except:`, no empty `except: pass`. Validate inputs at the top
of the function (fail fast). A domain exception hierarchy:

```
QuantAgentError
├── DataError            (ProviderUnavailable, StaleData, InsufficientData, UnknownTicker)
├── ToolError            (ToolTimeout, ToolValidationError, BudgetExhausted)
├── VerificationError    (UngroundedNumber, InvalidCitation, ConstraintBreach)
└── GuardrailError       (OutOfScope, ProhibitedRequest, InjectionDetected)
```

`structlog` with `trace_id` bound to context; log with context
(`logger.error("filing_fetch_failed", ticker=ticker, form=form, error=str(e))`), never bare
`print`. Never log secrets, full prompts in production, or other tenants' identifiers.

### 4.5 Async

All I/O is `async`. Quant functions are synchronous and CPU-bound; if one exceeds ~200 ms, run it
in a thread/process executor rather than making it `async` (making a CPU-bound function `async`
solves nothing and hides the blocking).

---

## 5. Tool authoring contract

Every tool follows this recipe. Deviating breaks the planner, the cache and the verifier
simultaneously.

1. Define `XInput` / `XOutput` Pydantic models in `contracts/tools.py`. Output must contain
   `MetricValue`s or typed records — never bare floats, never `dict[str, Any]`, never a DataFrame.
2. Declare the tool spec: `name`, description written *for the planner* (say when to use it and
   when not to), `p95_latency_ms`, `est_cost_usd`, `cache_ttl_s`, `side_effects="READ_ONLY"`.
3. Implement the adapter: validate → fetch via `data/` → compute via `quant/` → wrap with
   `Provenance`. Adapter body should be ~15–30 lines. If it is longer, logic has leaked in.
4. Register in `tools/registry.py`. The JSON schema handed to the LLM is generated from the
   Pydantic model — never hand-written, or it will drift.
5. Tests: unit (mocked data layer), integration (recorded provider), and a schema round-trip test.

```python
class CalculatePortfolioVarInput(BaseModel):
    portfolio_id: str
    alpha: float = Field(0.95, ge=0.90, le=0.999)
    horizon_days: int = Field(1, ge=1, le=20)
    method: Literal["historical", "parametric", "monte_carlo"] = "historical"
    lookback_days: int = Field(504, ge=250, le=2520)


@registry.tool(
    name="calculate_portfolio_var",
    description=(
        "Value-at-Risk of an entire portfolio at a given confidence level. "
        "Use for portfolio-level downside risk. Do NOT use for single-position "
        "risk (use calculate_component_var) or for realised losses (use "
        "calculate_max_drawdown)."
    ),
    p95_latency_ms=250,
    est_cost_usd=0.0,
    cache_ttl_s=900,
    side_effects="READ_ONLY",
)
async def calculate_portfolio_var(
    inp: CalculatePortfolioVarInput, ctx: ToolContext
) -> MetricValue:
    holdings = await ctx.portfolios.get_holdings(inp.portfolio_id, tenant_id=ctx.tenant_id)
    panel = await ctx.prices.get_return_panel(
        tickers=[h.ticker for h in holdings], lookback_days=inp.lookback_days
    )
    var = quant.var.portfolio_var(
        weights=to_weights(holdings),
        returns=panel.matrix,
        alpha=inp.alpha,
        horizon_days=inp.horizon_days,
        method=inp.method,
    )
    return ctx.wrap_metric(
        metric_id=f"portfolio_var_{int(inp.alpha*100)}_{inp.horizon_days}d",
        value=var.value,
        unit="ratio",
        method=var.method,
        window=f"{inp.lookback_days}d",
        sample_size=panel.n_obs,
        warnings=panel.warnings,
    )
```

`ctx.wrap_metric` builds `Provenance` automatically (call ID, `as_of`, `inputs_hash`, sources).
Do not construct `Provenance` by hand in a tool — that is how fields get forgotten.

**Tool descriptions are prompt engineering.** Each one gets a "use when / do not use when" pair.
Vague descriptions are the number-one cause of wrong tool selection, and tool-selection F1 is a
CI gate.

---

## 6. Quant core rules

1. **`float64` everywhere.** Never `float32` for money or risk.
2. **No look-ahead bias.** A metric with `as_of = T` may only use data available at `T`.
   Add a regression test for every rolling computation that asserts window boundaries.
3. **Sign conventions documented and consistent.** VaR/CVaR are positive loss fractions;
   drawdown is negative. Write it in the docstring and assert it in a test.
4. **Sample-size guards raise, never approximate.** `InsufficientDataError` with the required and
   available counts, surfaced to the user as a limitation.
5. **NaN policy is explicit per function.** Either drop-with-a-warning or raise — decide, document,
   and never let a NaN propagate into a `MetricValue`. Add `assert np.isfinite(value)` before
   returning.
6. **Calendar alignment before any cross-asset math.** Union of trading days with a documented
   fill rule; a misaligned panel silently corrupts every correlation.
7. **Currency.** v1 assumes a single base currency; if any holding differs, raise rather than
   mixing. Multi-currency is a milestone, not an assumption.
8. **Annualisation constants named, never inline.**
9. **Every function has a reference test** — closed form, an independent implementation, or a
   published worked example. "It looks right" is not a test.
10. **Seeds for anything stochastic**, recorded in `Provenance.seed`.

---

## 7. Prompt engineering conventions

- Prompts live in `prompts/<stage>/<name>.v<N>.jinja`, versioned. Never inline a multi-line prompt
  in Python. A prompt change is a code change and goes through review.
- Every prompt records its version in the trace, so a behaviour change is attributable.
- Structured output via tool/function schema, not "respond in JSON". Parse with Pydantic and
  retry once on validation failure with the validation error appended.
- `temperature=0` for planning, verification and classification. Low (≤0.3) for synthesis.
- Instruction hierarchy exactly as in `architecture.md` §6.2. Retrieved content always inside
  `<untrusted_data source="...">` and the system prompt always states that such content cannot
  issue instructions.
- **No business logic in prompts.** Thresholds, limits and rules live in `rules/*.yaml` and are
  applied by code. A prompt that says "flag if VaR exceeds 3%" is a bug — the rules engine does
  that, deterministically and testably.
- Keep the static portion of prompts stable and at the front so provider prompt caching works.

---

## 8. Verifier implementation rules

- Each layer is an independent, separately testable function returning `list[CheckResult]`. No
  layer may depend on another layer's internals.
- **Deterministic layers must not call an LLM.** Ever. That is what makes them trustworthy.
- The numeric extractor is its own module with its own extensive unit tests (currency symbols,
  thousands separators, percentages, ranges, multiples, basis points, negatives in parentheses,
  numbers adjacent to tickers). Write the tests before the extractor.
- The allowlist in §7.3 of `architecture.md` is a closed list. Adding to it requires an
  architecture-doc update and a new adversarial test proving it is not a loophole.
- The LLM critic gets **only** the claim and its evidence — never the full answer, never the
  synthesiser's reasoning, never the ledger. Independence is the point.
- Measure the verifier itself: label a set of intentionally-flawed answers and report the
  verifier's precision/recall. An unmeasured verifier is a false sense of safety.
- Repair is capped at 1. Do not add a loop. Do not let the synthesiser respond to the critique
  with an argument; the repair prompt asks for a corrected answer only.

---

## 9. Guardrail implementation rules

- Each check is a pure function `(payload, context) → GuardrailDecision`, individually tested.
- Wrap every check so an internal exception yields `BLOCK`, not a pass-through (I4). Test this
  explicitly by injecting a raising check.
- Prohibited-language checks operate on normalised text (case, unicode, spacing) to resist trivial
  evasion, and their patterns live in `rules/policy.yaml`.
- Disclosures are appended by code from a template, never generated by the LLM.
- Refusals use the §8.3 format: what cannot be answered, plus what can.
- Track over-refusal as a defect with its own eval slice. A system that refuses everything passes
  every safety test and is worthless.

---

## 10. Testing strategy

### 10.1 Shape

```
        /   E2E (~20)     \    full loop, recorded providers + replayed LLM, golden answers
       / Integration (~80) \   tool↔data↔quant wiring, RAG retrieval quality, DB
      /    Unit (400+)      \  quant formulas, verifier layers, guardrails, schemas, extractor
     /   Property-based      \  invariants over generated portfolios (hypothesis)
```

### 10.2 What must be covered

- **`quant/`:** ≥90% line coverage plus the property tests in `architecture.md` §10.2. This is the
  highest-value test surface in the project.
- **`verify/`:** ≥90%, including a dedicated suite of hand-crafted flawed answers — one per failure
  class (ungrounded number, fabricated citation, real citation misquoted, constraint breach,
  causal overclaim, contradictory claims, schema violation).
- **Tools:** input validation, provenance completeness, cache key stability, degraded-provider path.
- **Agent loop:** plan validation rejects cycles/unknown tools/over-budget plans; DAG executor
  parallelises independent branches (assert on concurrency, not just on the result); budget
  exhaustion mid-DAG; single retry semantics.
- **Guardrails:** each prohibited category; injection payloads in both user input and retrieved
  chunks; fail-closed on internal error.
- **Security:** cross-tenant access attempt returns nothing and raises; repository call without
  `tenant_id` raises.

### 10.3 Determinism

No test touches a live market-data or LLM endpoint. Providers are recorded cassettes; LLM calls in
E2E tests are replayed fixtures. For judge-dependent evals, pin the model version and report
variance across 3 runs. Fix every seed.

### 10.4 Eval harness (separate from tests)

`evals/` holds golden traces, the adversarial suite and the fixture portfolios described in
`architecture.md` §10. `make eval` produces a scorecard (markdown + JSON) with all §10.4 gate
metrics. The scorecard is committed per release so quality is trackable over time — this artefact
is also the single most persuasive thing to show a reviewer.

---

## 11. Milestones

Each milestone ships something runnable and fully tested. Do not begin the next until the DoD is
completely satisfied.

### M0 — Skeleton and contracts (foundation)
Repo layout, `pyproject.toml`, Makefile, Docker Compose (Postgres+pgvector, Redis), CI pipeline,
`import-linter` contracts, `structlog`+OTel bootstrap, and **all of `contracts/`** written first.

**DoD:** `make check` passes on an empty implementation; import contracts enforced in CI;
`/healthz` returns 200; every model in `architecture.md` §5 exists with round-trip tests.

### M1 — Data layer and quant core (the trustworthy part)
Price/fundamentals providers behind one interface, portfolio repository with tenant scoping,
Redis caching keyed on `inputs_hash`, and the full `quant/` module: returns, calendar alignment,
Ledoit-Wolf covariance, historical/parametric/MC VaR, CVaR, component VaR, drawdown, beta,
concentration, FF5+MOM factor regression, stress scenarios.

**DoD:** every quant function has a reference test and passes; all property tests pass; ≥90%
coverage on `quant/`; a CLI script prints a full risk report for the seeded portfolio with zero
LLM involvement. **This milestone must be correct before an LLM is introduced.**

### M2 — Tool layer and MCP surface
All v1 tools per the catalogue, registry with auto-generated schemas, `ToolContext`,
`compute_expression` safe evaluator, and the MCP server exposing the same tools.

**DoD:** every tool has unit + integration tests; provenance completeness asserted for all;
schemas validate; the MCP server is usable from an MCP client; cache hit/miss behaviour tested.

### M3 — Agent loop
Intent classifier and fast path, DAG planner with plan validation, async DAG executor with
timeouts/retries/circuit breakers, ledger, budget controller, synthesiser with strict structured
output, SSE streaming.

**DoD:** the worked example in `architecture.md` §16 runs end to end; plan validation rejects the
full set of malformed plans; parallelism asserted by test; budget exhaustion degrades gracefully;
p95 within the §13.1 budget on the E2E suite; golden-trace tool-selection F1 ≥ 0.90.

### M4 — Verifier (the differentiator — do not compress this)
V1–V5 with verdict aggregation, the numeric extractor, the citation checker, the YAML rules engine,
the independent LLM critic, the single repair pass, and the safe fallback.

**DoD:** `hallucinated_number_rate == 0` on the golden set; every flawed-answer fixture is caught
by the correct layer with the correct span; verifier precision/recall measured and reported;
critic false-positive rate < 8%; safe fallback path tested and alerted.

### M5 — Guardrails and RAG
Inbound/outbound guardrails, injection classifier, disclosure assembly, refusal templates; EDGAR
ingestion with item-aware chunking, hybrid retrieval + RRF + rerank, freshness filters, source
tiering, verbatim-excerpt offsets.

**DoD:** adversarial suite 100% safe behaviour; injection payloads in retrieved chunks quarantined
and logged; citation precision ≥ 0.98; over-refusal slice within threshold; retrieval quality
measured (recall@8 on a labelled set).

### M6 — Observability, evals, hardening
Full trace persistence and the trace viewer endpoint, hash-chained audit log, all SLIs exported,
alert rules, the nightly reproducibility replay, confidence calibration measurement, load test.

**DoD:** `GET /v1/traces/{id}` renders a complete audit view; `make replay TRACE=...` reproduces
metrics bit-for-bit; all §10.4 gates wired into CI; calibration curve and Brier score reported;
a committed eval scorecard.

### M7 — Portfolio engine and polish
Constrained mean-variance / min-variance / risk-parity optimiser, rebalance proposals with ex-ante
Δrisk, trade-impact simulation, deterministic report artefact, README with the worked example,
a demo script, and an architecture-decision retrospective.

**DoD:** optimiser respects every declared constraint (property-tested); rebalance proposals always
include ex-ante deltas; README lets a new reader run the demo in under 5 minutes.

---

## 12. Definition of Done for any PR

Run this as an independent review pass. "It runs" is not done.

- [ ] `make check` passes (lint, format, `mypy`, import contracts, tests, eval gates)
- [ ] No invariant in §1 violated
- [ ] New quant logic has a reference test and, where applicable, a property test
- [ ] New numbers reaching the user flow through `MetricValue` with complete provenance
- [ ] New tool has "use when / do not use when" in its description, plus tests and a cache TTL
- [ ] Errors handled explicitly; no bare `except`; no silent failure
- [ ] No dead code, no commented-out code, no leftover `print`/debug
- [ ] Comments audited against §4.1 — for each one: is this a *why* or a *what*?
- [ ] I/O · logic · orchestration remain separated
- [ ] Structured logging with `trace_id`; no secrets, no PII, no cross-tenant identifiers logged
- [ ] `architecture.md` updated if a decision, contract, rule or budget changed (and a new ADR row
      added if a decision was reversed)
- [ ] Eval scorecard re-run if agent behaviour, prompts or rules changed; deltas explained
- [ ] Commit is atomic with a clear message; one logical change per commit

---

## 13. Anti-patterns — do not do these

**Architectural**
- Letting the LLM compute, round, convert or compare a number "just this once".
- Putting a formula inside a tool adapter, or a fetch inside a quant function.
- Passing raw provider JSON or a DataFrame across a layer boundary instead of a typed model.
- Adding an agent framework to "simplify" the orchestrator (ADR-002).
- Making the verifier advisory because it is blocking a demo. Fix the answer, not the verifier.
- Adding more agents before single-agent metrics are stable.

**Quant**
- Sample covariance at low `T/N`; mixing log and simple returns; annualising without stating the
  convention; computing VaR from a historical NAV series that embeds past weights; silently
  shortening a lookback window because data is missing; answering "am I overexposed?" with GICS
  sector weight alone.

**Agent**
- A linear tool loop where a DAG would parallelise.
- Retrying non-transient errors.
- Unbounded repair loops.
- Truncating the ledger to fit a context window without recording that truncation.

**Reliability**
- `except Exception: pass` anywhere, especially in a guardrail.
- A confidence score that nothing computes and nothing validates.
- An empty `limitations` list.
- Caching without an `as_of`-aware key, then serving stale risk numbers as current.
- Logging a full prompt containing portfolio positions to a shared log sink.

**Process**
- Building M3 before M1's quant tests pass.
- Adding a tool because it sounds impressive rather than because a golden query needs it.
- Changing a prompt without versioning it or re-running evals.

---

## 14. README requirements (M7)

The README is what a reviewer reads first. It must contain, in this order: one-paragraph statement
of what the system does and the governance problem it solves; the architecture diagram; the
determinism-boundary explanation; the §16 worked example with the real JSON output; the **eval
scorecard table** (hallucinated-number rate, tool-selection F1, citation precision, p95 latency,
cost per request); quickstart in under 5 minutes; explicit limitations and the "this is not
investment advice / not a trading system" statement.

Lead with the reliability numbers. Anyone can build an agent that calls tools; the scorecard and
the verifier are the parts that are hard, and they are what distinguishes this project.

---

## 15. Open questions for a human to decide

Flag these rather than guessing:

1. Market-data provider for anything beyond a demo — `yfinance` terms of use make it unsuitable
   for production, and licensing needs a decision.
2. Retention policy for full prompts and portfolio data in traces (privacy vs. auditability).
3. Whether `require_review` mode is on by default, and what the materiality threshold is.
4. Jurisdictions in scope, which drives the disclosure and advice-framing configuration.
5. Initial `theme_map.yaml` ownership and review cadence — it is a model input that goes stale.
6. Benchmark and mandate defaults for the demo portfolio (they determine which constraint rules
   ever fire).
