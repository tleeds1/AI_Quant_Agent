# AI Quant / Financial Agent

A governed financial analysis agent: the LLM plans and explains, a deterministic quant core
computes every number, a hybrid verifier proves every claim traces back to data, and a guardrail
layer enforces compliance before anything reaches the user.

Full design: [`docs/architecture.md`](docs/architecture.md). Implementation guideline and
milestone plan: [`docs/guideline.md`](docs/guideline.md).

**Status:** M0-M4 done and verified end-to-end. M5 (guardrails + RAG) planning in progress — see
[`docs/PROGRESS.md`](docs/PROGRESS.md) for exact resume state and next steps.

## Quickstart

```bash
uv sync --extra dev
cp .env.example .env
docker compose up -d
make check
make dev
curl http://localhost:8000/v1/healthz
```

This is analysis, not investment advice, and not a trading system — see `docs/architecture.md`
§1.2 and §11.4.
