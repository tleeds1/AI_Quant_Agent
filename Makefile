SHELL := sh
.PHONY: dev test test-unit test-integration test-e2e lint fmt typecheck arch \
        migrate seed risk-report mcp-server ingest-filings eval eval-grounding eval-tools \
        eval-adversarial eval-latency replay check

dev:
	uv run uvicorn quantagent.api.app:app --reload --app-dir src --port 8000

lint:
	uv run black --check src tests evals
	uv run ruff check src tests evals

fmt:
	uv run black src tests evals
	uv run ruff check --fix src tests evals

typecheck:
	uv run mypy src

arch:
	uv run lint-imports

test:
	uv run pytest

test-unit:
	uv run pytest tests/unit

test-integration:
	uv run pytest tests/integration

test-e2e:
	uv run pytest tests/e2e

migrate:
	uv run alembic upgrade head

seed:
	uv run python scripts/seed_portfolio.py

risk-report:
	uv run python scripts/print_risk_report.py

mcp-server:
	uv run python -m quantagent.tools.mcp_server

ingest-filings:
	uv run python scripts/ingest_filings.py $(TICKER)

eval:
	uv run python -m evals.run_scorecard

eval-grounding:
	uv run pytest tests/unit/evals/test_golden_set.py -q -s

eval-tools:
	uv run python -m evals.eval_tool_selection

eval-adversarial:
	uv run pytest tests/unit/tools/test_research.py::test_retrieve_company_filings_quarantines_injected_chunk -q

eval-latency:
	@echo "no live-latency/cost harness yet: a correct end-to-end measurement needs a real"
	@echo "ANTHROPIC_API_KEY plus real Postgres/Redis/yfinance, not a re-implementation that"
	@echo "could silently diverge from what actually ships. Run real traffic through the app"
	@echo "(make dev) with a real key, then read latency_ms/cost_usd per call from the"
	@echo "persisted traces via GET /v1/traces/{id} (agent/loop.py::_persist_trace already"
	@echo "records both, faithfully, for every LLM and tool call)."

replay:
	uv run python scripts/replay_trace.py $(TRACE)

check: lint typecheck arch test
