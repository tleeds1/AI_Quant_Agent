"""tests/e2e/test_worked_example.py -- the literal M3 DoD requirement: "the
worked example in architecture.md §16 runs end to end". §16's literal
example uses 2 tools not built in M2 (`get_theme_exposure`,
`retrieve_company_filings` -- the latter is RAG, out of reach until M5); per
the M3 plan's scope decision, this test substitutes a closely-analogous
worked example built entirely from the 18 real M2 tools:

    s1: get_holdings                                          depends_on=[]
    s2: get_sector_exposure                                   depends_on=[s1]
    s3: calculate_portfolio_var                                depends_on=[s1]
    s4: calculate_component_var(group_by="ticker")             depends_on=[s2, s3]

A genuine diamond: s2/s3 are independent (both only need s1's holdings) and
must run concurrently; s4 fans back in.

Real app, real SQLite-backed seeded portfolio, real tool/quant execution
against monkeypatched provider network fetches (M1/M2's established
pattern) -- only the LLM calls are mocked (via `httpx.MockTransport` injected
into `LLMClient`, see `tests/unit/llm/fixtures.py`).
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from quantagent.api.app import create_app
from quantagent.api.deps import get_app_resources
from quantagent.contracts.answer import AgentAnswer
from quantagent.data.cache import CacheClient
from quantagent.data.models import Base
from quantagent.data.providers.factors import KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import YFinanceFundamentalsProvider
from quantagent.data.providers.prices import YFinancePriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.context import ToolContext
from tests.unit.llm.fixtures import MockLLMSession, tool_use_response

PORTFOLIO_ID = "pf_e2e"
TENANT_ID = "tenant_e2e"
TICKERS = ["AAPL", "JPM", "XOM"]
BENCHMARK = "SPY"
SECTOR_BY_TICKER = {"AAPL": "Technology", "JPM": "Financial Services", "XOM": "Energy"}
_OUTPUT_TOOL = "emit_structured_output"


def _synthetic_price_frame(n: int, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range(end=date.today(), periods=n)
    frames = {}
    for ticker in [*TICKERS, BENCHMARK]:
        closes = 100.0 * (1.0 + rng.normal(0.0003, 0.015, size=n)).cumprod()
        frames[ticker] = pd.DataFrame(
            {"Open": closes, "High": closes, "Low": closes, "Close": closes, "Volume": 1000},
            index=index,
        )
    return pd.concat(frames, axis=1)


def _fake_fundamentals_info(ticker: str) -> dict[str, object]:
    return {
        "sector": SECTOR_BY_TICKER.get(ticker, "Technology"),
        "industry": "n/a",
        "totalRevenue": 1_000_000.0,
        "profitMargins": 0.2,
        "trailingPE": 20.0,
        "shortName": ticker,
    }


@pytest.fixture
async def session_factory(
    monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(engine, expire_on_commit=False)

    repository = PortfolioRepository(factory)
    await repository.upsert_portfolio(
        id=PORTFOLIO_ID,
        tenant_id=TENANT_ID,
        name="E2E Worked Example Portfolio",
        base_currency="USD",
        benchmark_ticker=BENCHMARK,
        mandate_constraints={},
    )
    await repository.upsert_holdings(
        PORTFOLIO_ID, TENANT_ID, date.today(), [(t, 10.0, 100.0) for t in TICKERS]
    )

    monkeypatch.setattr(
        YFinancePriceProvider,
        "_download_sync",
        staticmethod(lambda *a, **k: _synthetic_price_frame(700)),
    )
    monkeypatch.setattr(
        YFinanceFundamentalsProvider,
        "_fetch_info_sync",
        staticmethod(lambda ticker: _fake_fundamentals_info(ticker)),
    )

    yield factory
    await engine.dispose()


def _worked_example_plan_payload() -> dict[str, object]:
    return {
        "steps": [
            {
                "id": "s1",
                "tool": "get_holdings",
                "args": {"portfolio_id": PORTFOLIO_ID},
                "depends_on": [],
            },
            {
                "id": "s2",
                "tool": "get_sector_exposure",
                "args": {"portfolio_id": PORTFOLIO_ID},
                "depends_on": ["s1"],
            },
            {
                "id": "s3",
                "tool": "calculate_portfolio_var",
                "args": {
                    "portfolio_id": PORTFOLIO_ID,
                    "alpha": 0.95,
                    "horizon_days": 1,
                    "method": "historical",
                },
                "depends_on": ["s1"],
            },
            {
                "id": "s4",
                "tool": "calculate_component_var",
                "args": {
                    "portfolio_id": PORTFOLIO_ID,
                    "alpha": 0.95,
                    "method": "parametric",
                    "group_by": "ticker",
                },
                "depends_on": ["s2", "s3"],
            },
        ],
        "success_criteria": "quantify sector concentration and its portfolio VaR contribution",
    }


def _intent_response() -> dict[str, object]:
    return tool_use_response(
        _OUTPUT_TOOL,
        {
            "label": "PORTFOLIO_ANALYSIS",
            "confidence": 0.9,
            "rationale": "needs multi-step risk analysis",
        },
    )


def _plan_response() -> dict[str, object]:
    return tool_use_response(_OUTPUT_TOOL, _worked_example_plan_payload())


def _metric_value_dict(metric_id: str, value: float) -> dict[str, object]:
    return {
        "metric_id": metric_id,
        "value": value,
        "unit": "ratio",
        "method": "historical",
        "window": None,
        "ci_95": None,
        "provenance": {
            "tool_call_id": "tc_mock",
            "tool_name": "calculate_portfolio_var",
            "as_of": date(2026, 8, 25).isoformat(),
            "computed_at": datetime(2026, 8, 25, 12, 0, 0).isoformat(),
            "inputs_hash": "h",
            "data_sources": ["yfinance"],
            "estimator": None,
            "sample_size": None,
            "seed": None,
            "warnings": [],
        },
    }


def _synthesis_payload(*, evidence_id: str = "ev1") -> dict[str, object]:
    return {
        "trace_id": "placeholder",
        "scope": "PORTFOLIO",
        "decision": "HOLD",
        "confidence": 0.85,
        "confidence_basis": ["ledger fully populated"],
        "risk_level": "MEDIUM",
        "horizon": "n/a",
        "summary": "The portfolio's VaR is concentrated across its sector buckets.",
        "claims": [
            {
                "claim_id": "c1",
                # Deliberately no digits here: V2 numeric grounding has its
                # own dedicated, thorough test suite (tests/unit/verify/
                # test_numeric_grounding.py) -- this e2e test only proves the
                # full pipeline's WIRING works end-to-end, not V2's regex.
                "text": "The portfolio's VaR is driven by its sector concentration.",
                "claim_type": "numeric",
                "evidence_ids": [evidence_id],
                "hedge": "none",
            }
        ],
        "evidence": [
            {
                "evidence_id": "ev1",
                "kind": "metric",
                "ref": "portfolio_var",
                "excerpt": None,
                "char_span": None,
                "source_title": "calculate_portfolio_var",
                "source_url": None,
                "source_tier": None,
                "published_at": None,
                "retrieval_score": None,
            }
        ],
        # V1's metric-ref-resolution check requires evidence.ref ("portfolio_var")
        # to be a real key here -- self-consistency within the answer, not a
        # cross-check against the real ledger's own metric_id naming (that's
        # V2's separate numeric-value job).
        "quant_metrics": {"portfolio_var": _metric_value_dict("portfolio_var", 0.025)},
        "constraints_checked": [],
        "limitations": ["Based on a 504-day historical lookback."],
        "disclosures": ["Analysis only, not investment advice."],
    }


def _synthesis_response(*, evidence_id: str = "ev1") -> dict[str, object]:
    return tool_use_response(_OUTPUT_TOOL, _synthesis_payload(evidence_id=evidence_id))


def _critic_supported_response() -> dict[str, object]:
    return tool_use_response(
        _OUTPUT_TOOL,
        {
            "claim_verdicts": [
                {
                    "claim_id": "c1",
                    "verdict": "SUPPORTED",
                    "reason": "Matches the ledger VaR figure.",
                    "severity": "low",
                }
            ],
            "contradictions": [],
        },
    )


class _E2EResources:
    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        anthropic_session: MockLLMSession,
    ) -> None:
        self.anthropic_client = anthropic_session.build_client()
        self.prompt_loader = PromptLoader()
        self._session_factory = session_factory

    def tool_context(self, tenant_id: str) -> ToolContext:
        # Never touched: every provider below is given cache=None, so no
        # tool call in this DAG ever reaches into ctx.cache. CacheClient's
        # constructor is lazy (redis-py doesn't connect until a command is
        # issued), so this is safe without a running Redis.
        cache = CacheClient.from_settings()
        return ToolContext(
            tenant_id=tenant_id,
            portfolios=PortfolioRepository(self._session_factory),
            prices=YFinancePriceProvider(cache=None),
            fundamentals=YFinanceFundamentalsProvider(cache=None),
            factors=KenFrenchFactorDataProvider(cache=None),
            cache=cache,
        )


def _run_and_collect(client: TestClient) -> list[dict[str, object]]:
    with client.stream(
        "POST",
        "/v1/analyze",
        json={
            "question": "How concentrated is my portfolio's risk by sector?",
            "portfolio_id": PORTFOLIO_ID,
        },
        headers={"X-Tenant-Id": TENANT_ID},
    ) as response:
        assert response.status_code == 200
        body = b"".join(response.iter_bytes())
    frames = [f for f in body.split(b"\n\n") if f.strip()]
    events = []
    for frame in frames:
        text = frame.decode()
        events.append(json.loads(text.split("data: ", 1)[1]))
    return events


async def test_worked_example_runs_end_to_end(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    anthropic_session = MockLLMSession(
        [_intent_response(), _plan_response(), _synthesis_response(), _critic_supported_response()]
    )
    app = create_app()
    app.dependency_overrides[get_app_resources] = lambda: _E2EResources(
        session_factory, anthropic_session
    )

    with TestClient(app) as client:
        events = _run_and_collect(client)

    kinds = [e["event"] for e in events]
    assert kinds[0] == "plan"
    assert kinds[-3:] == ["draft", "verdict", "final"]
    assert kinds.count("tool_start") == 4
    assert kinds.count("tool_done") == 4

    plan_event = events[0]
    assert [s["tool"] for s in plan_event["steps"]] == [
        "get_holdings",
        "get_sector_exposure",
        "calculate_portfolio_var",
        "calculate_component_var",
    ]

    tool_done_events = [e for e in events if e["event"] == "tool_done"]
    assert all(e["status"] == "OK" for e in tool_done_events)

    final = events[-1]
    answer = AgentAnswer.model_validate(final["answer"])
    assert answer.verification.verdict in ("PASS", "PASS_WITH_WARNINGS")
    assert len(answer.limitations) >= 1
    assert answer.decision != ""

    verdict_event = events[-2]
    assert verdict_event["verdict"] == answer.verification.verdict
    assert anthropic_session.call_count == 4  # intent + plan + synthesis + critic, no repair needed


async def test_worked_example_repair_path_fires_on_dangling_evidence(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    anthropic_session = MockLLMSession(
        [
            _intent_response(),
            _plan_response(),
            _synthesis_response(
                evidence_id="ev_dangling"
            ),  # claim references an evidence_id not in evidence[] -- V1
            # FAILs and short-circuits before V2-V5, so no critic call is
            # consumed for this (rejected) attempt.
            _synthesis_response(evidence_id="ev1"),  # corrected on repair
            _critic_supported_response(),  # V5 runs on the repaired attempt
        ]
    )
    app = create_app()
    app.dependency_overrides[get_app_resources] = lambda: _E2EResources(
        session_factory, anthropic_session
    )

    with TestClient(app) as client:
        events = _run_and_collect(client)

    assert anthropic_session.call_count == 5  # intent + plan + synthesis + repair + critic
    final = events[-1]
    answer = AgentAnswer.model_validate(final["answer"])
    assert answer.verification.verdict == "PASS"
    assert answer.verification.repair_attempts == 1


async def test_worked_example_safe_fallback_on_repeated_verification_failure(
    session_factory: async_sessionmaker[AsyncSession],
) -> None:
    """M4 DoD: "safe fallback path tested" at the full-loop level (the
    harness-level `test_safe_fallback_is_alerted` in tests/unit/evals/
    proves the alert half in isolation; this proves the path is genuinely
    reachable end-to-end). Both synthesis attempts reference a dangling
    evidence id -- V1 FAILs on the initial AND the one repair attempt, so
    verification never passes and SAFE_FALLBACK must trigger. V1 is a
    hard-stop layer, so V2-V5 never run on either attempt -- no critic
    response is needed at all.
    """
    anthropic_session = MockLLMSession(
        [
            _intent_response(),
            _plan_response(),
            _synthesis_response(evidence_id="ev_dangling"),
            _synthesis_response(evidence_id="ev_still_dangling"),
        ]
    )
    app = create_app()
    app.dependency_overrides[get_app_resources] = lambda: _E2EResources(
        session_factory, anthropic_session
    )

    with TestClient(app) as client:
        events = _run_and_collect(client)

    assert (
        anthropic_session.call_count == 4
    )  # intent + plan + synthesis + 1 repair, no critic calls
    final = events[-1]
    answer = AgentAnswer.model_validate(final["answer"])
    assert answer.decision == "INSUFFICIENT_EVIDENCE"
    assert answer.risk_level == "EXTREME"
    assert answer.verification.verdict == "FAIL"
    assert answer.verification.repair_attempts == 1
