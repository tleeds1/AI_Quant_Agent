from __future__ import annotations

from datetime import date, datetime

import pytest

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import Claim, Evidence
from quantagent.contracts.ledger import Ledger, ToolCallRecord
from quantagent.contracts.verification import VerificationReport
from quantagent.verify.numeric_grounding import (
    NumericToken,
    _tokenize_field,
    hallucinated_number_rate,
    round_to_sig_figs,
    run_v2_numeric_grounding,
)
from quantagent.verify.types import CheckResult


def _provenance(call_id: str = "tc_1") -> dict[str, object]:
    return {
        "tool_call_id": call_id,
        "tool_name": "calculate_portfolio_var",
        "as_of": date(2026, 8, 25).isoformat(),
        "computed_at": datetime(2026, 8, 25, 12, 0, 0).isoformat(),
        "inputs_hash": "h",
        "data_sources": ["yfinance"],
        "estimator": None,
        "sample_size": None,
        "seed": None,
        "warnings": [],
    }


def _ledger(value: float = 0.034, *, ci_95: list[float] | None = None) -> Ledger:
    result: dict[str, object] = {
        "metric_id": "var",
        "value": value,
        "unit": "ratio",
        "method": "historical",
        "provenance": _provenance(),
    }
    if ci_95 is not None:
        result["ci_95"] = ci_95
    return Ledger(
        trace_id="tr",
        calls=[
            ToolCallRecord(
                call_id="tc_1",
                tool_name="calculate_portfolio_var",
                args={},
                args_hash="h",
                status="OK",
                latency_ms=1,
                cost_usd=0.0,
                result=result,
                error=None,
            )
        ],
        numeric_index={"tc_1.result.value": value},
    )


def _answer(
    summary: str, *, claim_text: str | None = None, excerpt: str | None = None
) -> AgentAnswer:
    claims = (
        [Claim(claim_id="c1", text=claim_text, claim_type="numeric", evidence_ids=["ev1"])]
        if claim_text
        else []
    )
    return AgentAnswer(
        trace_id="tr",
        scope="PORTFOLIO",
        decision="HOLD",
        confidence=0.5,
        confidence_basis=[],
        risk_level="LOW",
        horizon="n/a",
        summary=summary,
        claims=claims,
        evidence=[
            Evidence(
                evidence_id="ev1",
                kind="metric",
                ref="var",
                excerpt=excerpt,
                char_span=None,
                source_title="x",
                source_url=None,
                source_tier=None,
                published_at=None,
                retrieval_score=None,
            )
        ],
        quant_metrics={},
        constraints_checked=[],
        limitations=["none"],
        disclosures=[],
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )


def _by_text(results: list[CheckResult], text: str) -> CheckResult:
    matches = [r for r in results if r.offending_text == text]
    assert matches, f"no CheckResult with offending_text={text!r} in {results}"
    return matches[0]


# ---------------------------------------------------------------- tokenizer --


def _tok(text: str) -> NumericToken:
    toks = _tokenize_field(text, "x", claim_id=None)
    assert len(toks) == 1, f"expected exactly one token in {text!r}, got {toks}"
    return toks[0]


def test_tokenize_thousands_separator() -> None:
    t = _tok("1,234.5 is here")
    assert (t.value, t.unit) == (1234.5, "bare")


def test_tokenize_percent() -> None:
    t = _tok("4.3% is here")
    assert (t.value, t.unit) == (pytest.approx(0.043), "ratio")


def test_tokenize_dollar_magnitude() -> None:
    t = _tok("$1.2B is here")
    assert (t.value, t.unit) == (1_200_000_000.0, "usd")


def test_tokenize_basis_points() -> None:
    toks = _tokenize_field("43bp and 43bps", "x", claim_id=None)
    assert [(t.value, t.unit) for t in toks] == [(pytest.approx(0.0043), "ratio")] * 2


def test_tokenize_multiple() -> None:
    t = _tok("1.37x leverage")
    assert (t.value, t.unit) == (1.37, "ratio")


def test_tokenize_bare_decimal() -> None:
    t = _tok("0.71 alone")
    assert (t.value, t.unit) == (0.71, "bare")


def test_tokenize_range_en_dash_and_hyphen() -> None:
    for text in ("12-15% range", "12–15% range"):  # noqa: RUF001 -- en dash intentional
        toks = _tokenize_field(text, "x", claim_id=None)
        assert [round(t.value, 2) for t in toks] == [0.12, 0.15]


def test_tokenize_negative_in_parens() -> None:
    t = _tok("(3.2%) negative")
    assert t.value == pytest.approx(-0.032)


def test_tokenize_bare_negative() -> None:
    t = _tok("-3.2% bare negative")
    assert t.value == pytest.approx(-0.032)


def test_tokenize_trailing_period_excluded_from_span() -> None:
    t = _tok("...0.71.")
    assert t.text == "0.71"
    assert t.span == (3, 7)


def test_tokenize_percent_then_trailing_period() -> None:
    t = _tok("...0.71%.")
    assert t.text == "0.71%"


def test_tokenize_percent_immediately_followed_by_word_degrades_to_bare() -> None:
    # Documented known limitation: "5%ish" doesn't parse "%" as the unit
    # suffix since a word character follows it directly; the bare "5" is
    # still extracted (rather than silently dropping the token entirely).
    t = _tok("5%ish")
    assert (t.text, t.value, t.unit) == ("5", 5.0, "bare")


def test_tokenize_company_name_false_positive_documented() -> None:
    # Documented known limitation: "3M" (the company) parses identically to
    # a $3,000,000 magnitude-suffixed figure.
    t = _tok("3M reported strong earnings")
    assert (t.text, t.value, t.unit) == ("3M", 3_000_000.0, "usd")


# ------------------------------------------------------------- sig figs -----


def test_round_to_sig_figs() -> None:
    assert round_to_sig_figs(1234567.89, 2) == 1_200_000.0
    assert round_to_sig_figs(1234567.89, 3) == 1_230_000.0
    assert round_to_sig_figs(1234567.89, 1) == 1_000_000.0


def test_round_to_sig_figs_zero_does_not_crash() -> None:
    assert round_to_sig_figs(0.0, 2) == 0.0


def test_round_to_sig_figs_negative() -> None:
    assert round_to_sig_figs(-0.15873, 3) == -0.159


# ------------------------------------------------------- allowed-set/closure --


def test_ci_bounds_are_grounded_as_a_range() -> None:
    ledger = _ledger(value=0.095, ci_95=[0.08, 0.11])
    answer = _answer("expected return of 8-11%")
    results = run_v2_numeric_grounding(answer, ledger)
    assert all(r.verdict == "PASS" for r in results)
    assert _by_text(results, "8-11%").nearest_ledger_key in {
        "tc_1.result.ci_95_lo",
        "tc_1.result.ci_95_hi",
    }


def test_value_matches_via_percent_scaling_closure() -> None:
    ledger = _ledger(value=0.15)  # e.g. an HHI stored as a raw ratio
    answer = _answer("HHI of 15%")
    results = run_v2_numeric_grounding(answer, ledger)
    assert _by_text(results, "15%").verdict == "PASS"


def test_unrelated_value_does_not_spuriously_match() -> None:
    ledger = _ledger(value=0.15)
    answer = _answer("a totally different figure of 8.3%")
    results = run_v2_numeric_grounding(answer, ledger)
    assert _by_text(results, "8.3%").verdict == "FAIL"


# ----------------------------------------------------------------- tolerance --


def test_display_rounded_dollar_value_matches_within_tolerance() -> None:
    ledger = _ledger(value=1234567.89)
    answer = _answer("total exposure of $1.2M")
    results = run_v2_numeric_grounding(answer, ledger)
    assert _by_text(results, "$1.2M").verdict == "PASS"


def test_just_inside_tolerance_passes_just_outside_fails() -> None:
    ledger = _ledger(value=0.20)
    inside = _answer("value near 0.201")  # 0.5% of 0.20 rounded-candidate is small
    outside = _answer("value near 0.30")
    assert _by_text(run_v2_numeric_grounding(inside, ledger), "0.201").verdict == "PASS"
    assert _by_text(run_v2_numeric_grounding(outside, ledger), "0.30").verdict == "FAIL"


# ----------------------------------------------------------------- allowlist --


def test_standalone_year_allowlisted() -> None:
    ledger = _ledger()
    results = run_v2_numeric_grounding(_answer("in 2026 the portfolio grew"), ledger)
    r = _by_text(results, "2026")
    assert r.verdict == "PASS"
    assert "standalone year" in r.message


def test_quarter_label_allowlisted() -> None:
    ledger = _ledger()
    results = run_v2_numeric_grounding(_answer("Q3 results were strong"), ledger)
    r = _by_text(results, "3")
    assert r.verdict == "PASS"
    assert "quarter label" in r.message


def test_count_of_items_allowlisted_when_adjacent_to_count_noun() -> None:
    ledger = _ledger()
    results = run_v2_numeric_grounding(_answer("the portfolio has 3 holdings"), ledger)
    r = _by_text(results, "3")
    assert r.verdict == "PASS"
    assert "count of listed items" in r.message


def test_small_integer_without_count_noun_is_not_allowlisted() -> None:
    # Documents the stated tradeoff: "3 breaches" isn't recognized (no
    # count-noun match), so it falls through to normal grounding.
    # Ledger value deliberately avoids 0.034 (the module-default fixture
    # value): 0.034 * 100 = 3.4, which 1-sig-fig-rounds to 3.0 and would
    # coincidentally "match" the bare token "3" -- a real closure collision
    # unrelated to what this test is checking.
    ledger = _ledger(value=0.567)
    results = run_v2_numeric_grounding(_answer("3 breaches occurred"), ledger)
    assert _by_text(results, "3").verdict == "FAIL"


def test_analysis_window_length_allowlisted() -> None:
    ledger = _ledger()
    results = run_v2_numeric_grounding(_answer("over a 5-year window"), ledger)
    r = _by_text(results, "5")
    assert r.verdict == "PASS"
    assert "analysis window" in r.message


def test_number_inside_verified_excerpt_allowlisted() -> None:
    ledger = _ledger()
    answer = _answer("revenue grew 4.3% year over year", excerpt="revenue grew 4.3% year over year")
    results = run_v2_numeric_grounding(answer, ledger)
    r = _by_text(results, "4.3%")
    assert r.verdict == "PASS"
    assert "verified excerpt" in r.message


# --------------------------------------------------------------- FAIL path ---


def test_transposed_digit_is_flagged_with_nearest_ledger_value() -> None:
    ledger = _ledger(value=0.034)
    answer = _answer("VaR of 0.043", claim_text="VaR of 0.043")
    results = run_v2_numeric_grounding(answer, ledger)
    # The same offending text appears in both `summary` and the claim's own
    # `text`, so two FAIL CheckResults are produced (one per source field) --
    # pick the claim-sourced one explicitly rather than assuming list order.
    failing = [r for r in results if r.verdict == "FAIL" and r.claim_id is not None]
    assert failing
    assert failing[0].offending_text == "0.043"
    assert failing[0].nearest_ledger_value == pytest.approx(0.034)
    assert failing[0].claim_id == "c1"


# --------------------------------------------------------- hallucination rate --


def test_hallucinated_number_rate_computation() -> None:
    results = [
        CheckResult(layer="V1", check_id="v1.x", verdict="PASS", message="m"),
        CheckResult(layer="V2", check_id="v2.numeric_grounding", verdict="PASS", message="m"),
        CheckResult(layer="V2", check_id="v2.numeric_grounding", verdict="PASS", message="m"),
        CheckResult(layer="V2", check_id="v2.numeric_grounding", verdict="PASS", message="m"),
        CheckResult(layer="V2", check_id="v2.numeric_grounding", verdict="FAIL", message="m"),
    ]
    assert hallucinated_number_rate(results) == 250.0


def test_hallucinated_number_rate_empty_is_zero() -> None:
    assert hallucinated_number_rate([]) == 0.0


def test_hallucinated_number_rate_zero_on_a_clean_answer() -> None:
    ledger = _ledger(value=0.034)
    answer = _answer("VaR is 3.4%.", claim_text="VaR is 3.4%.")
    results = run_v2_numeric_grounding(answer, ledger)
    assert hallucinated_number_rate(results) == 0.0
