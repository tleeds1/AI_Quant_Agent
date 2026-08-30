SHELL := sh
.PHONY: dev test test-unit test-integration test-e2e lint fmt typecheck arch \
        migrate seed risk-report mcp-server ingest-filings eval eval-grounding eval-tools \
        eval-adversarial eval-latency replay check

dev:
	uv run uvicorn quantagent.api.app:app --reload --app-dir src --port 8000

lint:
	uv run black --check src tests
	uv run ruff check src tests

fmt:
	uv run black src tests
	uv run ruff check --fix src tests

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
	@echo "no eval harness yet - introduced in M4-M6"

eval-grounding:
	@echo "no eval harness yet - introduced in M4"

eval-tools:
	@echo "no eval harness yet - introduced in M2/M3"

eval-adversarial:
	@echo "no eval harness yet - introduced in M5"

eval-latency:
	@echo "no eval harness yet - introduced in M3/M6"

replay:
	@echo "trace replay not implemented yet - introduced in M6; TRACE=$(TRACE) ignored"

check: lint typecheck arch test
