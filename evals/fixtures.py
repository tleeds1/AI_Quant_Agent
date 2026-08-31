"""evals/fixtures.py -- hand-authored (Ledger, AgentAnswer-draft)
golden/flawed-answer pairs the M4 DoD is measured against (guideline.md's
M4 DoD; architecture.md §10's "hallucination probes" / "golden set").

Each `GoldenFixture` is tagged with the layer/check_id that SHOULD catch it
(or None for a clean-pass fixture) so tests/unit/evals/test_golden_set.py
can assert "caught by the correct layer with the correct check" literally,
not just "verdict == FAIL".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime

from tests.unit.llm.fixtures import tool_use_response

from quantagent.contracts.answer import AgentAnswer, Decision, RiskLevel
from quantagent.contracts.evidence import Claim, Evidence
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.provenance import Provenance
from quantagent.contracts.verification import VerificationReport
from quantagent.verify.types import Layer

_OUTPUT_TOOL = "emit_structured_output"


@dataclass(frozen=True, slots=True)
class GoldenFixture:
    name: str
    ledger: Ledger
    answer: AgentAnswer
    expected_verdict: str  # "PASS" | "PASS_WITH_WARNINGS" | "FAIL"
    expected_layer: Layer | None  # None for a clean-pass fixture
    expected_check_id: str | None
    critic_responses: list[dict[str, object]] = field(default_factory=list)
    # Mock Anthropic response envelopes for V5, in the order it will be
    # called. Empty when the fixture is expected to hard-stop before V5.


def _provenance(call_id: str = "tc_1") -> Provenance:
    return Provenance(
        tool_call_id=call_id,
        tool_name="calculate_portfolio_var",
        as_of=date(2026, 8, 25),
        computed_at=datetime(2026, 8, 25, 12, 0, 0),
        inputs_hash="h1",
        data_sources=["yfinance"],
        estimator="historical",
        sample_size=504,
        seed=None,
        warnings=[],
    )


def _clean_ledger() -> Ledger:
    return Ledger(
        trace_id="tr_golden",
        calls=[
            ToolCallRecord(
                call_id="tc_1",
                tool_name="calculate_portfolio_var",
                args={"portfolio_id": "pf_1"},
                args_hash="h1",
                status="OK",
                latency_ms=100,
                cost_usd=0.0,
                result={
                    "metric_id": "portfolio_var",
                    "value": 0.025,
                    "unit": "ratio",
                    "method": "historical",
                    "provenance": _provenance("tc_1").model_dump(mode="json"),
                },
                error=None,
            )
        ],
        numeric_index={"tc_1.result.value": 0.025},
    )


def _var_metric() -> MetricValue:
    return MetricValue(
        metric_id="portfolio_var",
        value=0.025,
        unit="ratio",
        method="historical",
        provenance=_provenance(),
    )


def _evidence(evidence_id: str = "ev1", ref: str = "portfolio_var") -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        kind="metric",
        ref=ref,
        excerpt=None,
        char_span=None,
        source_title="calculate_portfolio_var",
        source_url=None,
        source_tier=None,
        published_at=None,
        retrieval_score=None,
    )


def _base_answer(
    *,
    claims: list[Claim],
    evidence: list[Evidence],
    decision: Decision = "HOLD",
    risk_level: RiskLevel = "MEDIUM",
    summary: str = "Portfolio VaR is stable given current holdings.",
    quant_metrics: dict[str, MetricValue] | None = None,
) -> AgentAnswer:
    return AgentAnswer(
        trace_id="tr_golden",
        scope="PORTFOLIO",
        decision=decision,
        confidence=0.7,
        confidence_basis=["ledger fully populated"],
        risk_level=risk_level,
        horizon="n/a",
        summary=summary,
        claims=claims,
        evidence=evidence,
        quant_metrics=(
            quant_metrics if quant_metrics is not None else {"portfolio_var": _var_metric()}
        ),
        constraints_checked=[],
        limitations=["Based on a 504-day historical lookback."],
        disclosures=["Analysis only, not investment advice."],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


_ALL_SUPPORTED_CRITIC_RESPONSE = tool_use_response(
    _OUTPUT_TOOL,
    {
        "claim_verdicts": [
            {
                "claim_id": "c1",
                "verdict": "SUPPORTED",
                "reason": "Evidence directly states the VaR figure claimed.",
                "severity": "low",
            }
        ],
        "contradictions": [],
    },
)


# ---------------------------------------------------------------- (a) clean --

_CLEAN_CLAIM = Claim(
    claim_id="c1", text="Portfolio VaR is stable.", claim_type="factual", evidence_ids=["ev1"]
)

FIXTURE_CLEAN_PASS = GoldenFixture(
    name="clean_pass",
    ledger=_clean_ledger(),
    answer=_base_answer(claims=[_CLEAN_CLAIM], evidence=[_evidence()]),
    expected_verdict="PASS",
    expected_layer=None,
    expected_check_id=None,
    critic_responses=[_ALL_SUPPORTED_CRITIC_RESPONSE],
)

# ---------------------------------------------- (b) hallucinated number (V2) --

FIXTURE_HALLUCINATED_NUMBER = GoldenFixture(
    name="hallucinated_number",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[
            Claim(
                claim_id="c1",
                text=(
                    "The portfolio's Sharpe ratio is 1.8, indicating strong risk-adjusted returns."
                ),
                claim_type="numeric",
                evidence_ids=["ev1"],
            )
        ],
        evidence=[_evidence()],
        summary="The portfolio's Sharpe ratio is 1.8.",
    ),
    expected_verdict="FAIL",
    expected_layer="V2",
    expected_check_id="v2.numeric_grounding",
    critic_responses=[],  # never reached -- V2 hard-stops
)

# --------------------------------------------- (c) dangling evidence id (V1) --

FIXTURE_DANGLING_EVIDENCE = GoldenFixture(
    name="dangling_evidence_id",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[
            Claim(
                claim_id="c1",
                text="Portfolio VaR is stable.",
                claim_type="factual",
                evidence_ids=["ev_missing"],
            )
        ],
        evidence=[_evidence()],
    ),
    expected_verdict="FAIL",
    expected_layer="V1",
    expected_check_id="v1.evidence_resolution",
    critic_responses=[],
)

# ---------------------------------------- (d) metric ref doesn't resolve (V1) --

FIXTURE_UNRESOLVED_METRIC_REF = GoldenFixture(
    name="unresolved_metric_ref",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[_CLEAN_CLAIM],
        evidence=[_evidence("ev1", ref="metric_that_does_not_exist")],
    ),
    expected_verdict="FAIL",
    expected_layer="V1",
    expected_check_id="v1.metric_ref_resolution",
    critic_responses=[],
)

# ----------------------------------------------- (e) R-001 EXTREME+BUY (V4) --

FIXTURE_R001_EXTREME_BUY = GoldenFixture(
    name="r001_extreme_risk_buy",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[_CLEAN_CLAIM], evidence=[_evidence()], decision="BUY", risk_level="EXTREME"
    ),
    expected_verdict="FAIL",
    expected_layer="V4",
    expected_check_id="R-001",
    critic_responses=[_ALL_SUPPORTED_CRITIC_RESPONSE],  # V4 is not a hard-stop -> V5 still runs
)

# --------------------------------------------- (f) R-008 guaranteed language --

FIXTURE_R008_GUARANTEED_LANGUAGE = GoldenFixture(
    name="r008_guaranteed_language",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[_CLEAN_CLAIM],
        evidence=[_evidence()],
        summary="This portfolio is guaranteed to outperform its benchmark.",
    ),
    expected_verdict="FAIL",
    expected_layer="V4",
    expected_check_id="R-008",
    critic_responses=[_ALL_SUPPORTED_CRITIC_RESPONSE],
)

# --------------------------------------------------- (g) unsupported claim (V5) --

FIXTURE_UNSUPPORTED_CLAIM = GoldenFixture(
    name="unsupported_claim",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[
            Claim(
                claim_id="c1",
                text="Portfolio VaR is stable.",
                claim_type="factual",
                evidence_ids=["ev1"],
            )
        ],
        evidence=[_evidence()],
    ),
    expected_verdict="FAIL",
    expected_layer="V5",
    expected_check_id="v5.entailment",
    critic_responses=[
        tool_use_response(
            _OUTPUT_TOOL,
            {
                "claim_verdicts": [
                    {
                        "claim_id": "c1",
                        "verdict": "UNSUPPORTED",
                        "reason": "The linked evidence does not contain a VaR figure at all.",
                        "severity": "high",
                    }
                ],
                "contradictions": [],
            },
        )
    ],
)

# ------------------------------------------------- (h) contradictory claims (V5) --

FIXTURE_CONTRADICTORY_CLAIMS = GoldenFixture(
    name="contradictory_claims",
    ledger=_clean_ledger(),
    answer=_base_answer(
        claims=[
            Claim(
                claim_id="c1",
                text="Portfolio VaR is stable at current levels.",
                claim_type="factual",
                evidence_ids=["ev1"],
            ),
            Claim(
                claim_id="c2",
                text="Portfolio VaR has spiked well above the mandate limit.",
                claim_type="factual",
                evidence_ids=["ev1"],
            ),
        ],
        evidence=[_evidence()],
        summary="Portfolio VaR is stable, and separately, VaR has spiked above the mandate limit.",
    ),
    expected_verdict="FAIL",
    expected_layer="V5",
    expected_check_id="v5.contradiction",
    critic_responses=[
        tool_use_response(
            _OUTPUT_TOOL,
            {
                "claim_verdicts": [
                    {
                        "claim_id": "c1",
                        "verdict": "SUPPORTED",
                        "reason": "matches ledger",
                        "severity": "low",
                    },
                    {
                        "claim_id": "c2",
                        "verdict": "SUPPORTED",
                        "reason": "presented as if separately supported",
                        "severity": "low",
                    },
                ],
                "contradictions": [
                    {
                        "claim_ids": ["c1", "c2"],
                        "explanation": (
                            "c1 states VaR is stable and c2 states VaR has spiked for the same "
                            "portfolio and window -- these cannot both be true."
                        ),
                        "severity": "high",
                    }
                ],
            },
        )
    ],
)

GOLDEN_FIXTURES: list[GoldenFixture] = [
    FIXTURE_CLEAN_PASS,
    FIXTURE_HALLUCINATED_NUMBER,
    FIXTURE_DANGLING_EVIDENCE,
    FIXTURE_UNRESOLVED_METRIC_REF,
    FIXTURE_R001_EXTREME_BUY,
    FIXTURE_R008_GUARANTEED_LANGUAGE,
    FIXTURE_UNSUPPORTED_CLAIM,
    FIXTURE_CONTRADICTORY_CLAIMS,
]
