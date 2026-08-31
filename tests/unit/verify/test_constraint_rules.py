from __future__ import annotations

from datetime import date, datetime

import pytest

from quantagent.contracts.metrics import MetricValue
from quantagent.verify.constraint_rules import RulesConfigError, RulesEngine, run_v4_checks
from tests.unit.agent.builders import build_provenance
from tests.unit.verify.builders import (
    build_answer,
    build_claim,
    build_evidence,
    build_ledger,
    build_tool_call_record,
)


def _metric(metric_id: str, value: float, *, as_of=None, computed_at=None) -> MetricValue:
    return MetricValue(
        metric_id=metric_id,
        value=value,
        unit="ratio",
        method="test",
        provenance=build_provenance(
            **({"as_of": as_of} if as_of else {}),
            **({"computed_at": computed_at} if computed_at else {}),
        ),
    )


def _result(results, rule_id: str):
    return next(r for r in results if r.check_id == rule_id)


def _cc(checks, rule_id: str):
    return next(c for c in checks if c.rule_id == rule_id)


def test_run_v4_checks_always_returns_all_ten_rules_sorted() -> None:
    answer = build_answer()
    ledger = build_ledger()
    results, checks = run_v4_checks(answer, ledger)
    assert [r.check_id for r in results] == sorted(f"R-{i:03d}" for i in range(1, 11))
    assert len(checks) == 10


def test_r001_pass_and_breach() -> None:
    ledger = build_ledger()
    ok = build_answer(risk_level="HIGH", decision="BUY")
    results, checks = run_v4_checks(ok, ledger)
    assert _result(results, "R-001").verdict == "PASS"
    assert _cc(checks, "R-001").status == "PASS"

    bad = build_answer(risk_level="EXTREME", decision="BUY")
    results, checks = run_v4_checks(bad, ledger)
    assert _result(results, "R-001").verdict == "FAIL"
    assert _cc(checks, "R-001").status == "BREACH"


def test_r002_pass_and_breach() -> None:
    ledger = build_ledger()
    within_cap = build_answer(
        decision="BUY", quant_metrics={"top_5_weight": _metric("top_5_weight", 0.20)}
    )
    results, _ = run_v4_checks(
        within_cap, ledger, mandate_constraints={"max_concentration_pct": 0.25}
    )
    assert _result(results, "R-002").verdict == "PASS"

    over_cap_no_hedge = build_answer(
        decision="BUY", quant_metrics={"top_5_weight": _metric("top_5_weight", 0.30)}
    )
    results, checks = run_v4_checks(
        over_cap_no_hedge, ledger, mandate_constraints={"max_concentration_pct": 0.25}
    )
    assert _result(results, "R-002").verdict == "FAIL"
    assert _cc(checks, "R-002").observed == pytest.approx(0.30)
    assert _cc(checks, "R-002").limit == pytest.approx(0.25)

    over_cap_hedged = build_answer(
        decision="BUY",
        quant_metrics={"top_5_weight": _metric("top_5_weight", 0.30)},
        summary="Recommend a hedge alongside this position.",
    )
    results, _ = run_v4_checks(
        over_cap_hedged, ledger, mandate_constraints={"max_concentration_pct": 0.25}
    )
    assert _result(results, "R-002").verdict == "PASS"


def test_r003_pass_and_breach() -> None:
    ledger_clean = build_ledger()
    low_confidence = build_answer(confidence=0.5)
    results, _ = run_v4_checks(low_confidence, ledger_clean)
    assert _result(results, "R-003").verdict == "PASS"

    ledger_degraded = build_ledger(calls=[build_tool_call_record(status="DEGRADED")])
    high_confidence_degraded = build_answer(confidence=0.9)
    results, _ = run_v4_checks(high_confidence_degraded, ledger_degraded)
    assert _result(results, "R-003").verdict == "WARN"  # R-003's severity is WARN in the YAML


def test_r004_pass_and_breach() -> None:
    ledger = build_ledger()
    under_limit = build_answer(
        quant_metrics={"portfolio_var_95_1d": _metric("portfolio_var_95_1d", 0.03)}
    )
    results, _ = run_v4_checks(under_limit, ledger, mandate_constraints={"max_var_pct": 0.05})
    assert _result(results, "R-004").verdict == "PASS"

    over_limit_buy = build_answer(
        decision="BUY", quant_metrics={"portfolio_var_95_1d": _metric("portfolio_var_95_1d", 0.08)}
    )
    results, checks = run_v4_checks(
        over_limit_buy, ledger, mandate_constraints={"max_var_pct": 0.05}
    )
    assert _result(results, "R-004").verdict == "FAIL"
    assert _cc(checks, "R-004").status == "BREACH"

    over_limit_reduce = build_answer(
        decision="REDUCE",
        quant_metrics={"portfolio_var_95_1d": _metric("portfolio_var_95_1d", 0.08)},
    )
    results, _ = run_v4_checks(over_limit_reduce, ledger, mandate_constraints={"max_var_pct": 0.05})
    assert _result(results, "R-004").verdict == "PASS"


def test_r004_missing_mandate_key_is_pass_not_fail_or_unknown() -> None:
    ledger = build_ledger()
    answer = build_answer(
        decision="BUY", quant_metrics={"portfolio_var_95_1d": _metric("portfolio_var_95_1d", 0.99)}
    )
    results, checks = run_v4_checks(answer, ledger, mandate_constraints={})
    assert _result(results, "R-004").verdict == "PASS"
    assert _cc(checks, "R-004").status == "PASS"


def test_r005_always_not_applicable() -> None:
    ledger = build_ledger()
    answer = build_answer(decision="BUY", risk_level="EXTREME")
    results, checks = run_v4_checks(answer, ledger)
    assert _result(results, "R-005").verdict == "PASS"
    assert _cc(checks, "R-005").status == "NOT_APPLICABLE"


def test_r009_pass_and_breach() -> None:
    # 1. No BUY/SELL decision -> PASS
    ledger_empty = build_ledger()
    no_trade = build_answer(decision="NO_ACTION")
    results, checks = run_v4_checks(no_trade, ledger_empty)
    assert _result(results, "R-009").verdict == "PASS"
    assert _cc(checks, "R-009").status == "PASS"

    # 2. BUY decision but no simulation -> FAIL / BREACH
    trade_no_sim = build_answer(decision="BUY")
    results, checks = run_v4_checks(trade_no_sim, ledger_empty)
    assert _result(results, "R-009").verdict == "FAIL"
    assert _cc(checks, "R-009").status == "BREACH"

    # 3. BUY decision with successful simulation -> PASS
    ledger_sim = build_ledger(
        calls=[build_tool_call_record(tool_name="simulate_trade_impact", status="OK")]
    )
    trade_with_sim = build_answer(decision="BUY")
    results, checks = run_v4_checks(trade_with_sim, ledger_sim)
    assert _result(results, "R-009").verdict == "PASS"
    assert _cc(checks, "R-009").status == "PASS"


def test_r006_pass_and_breach() -> None:
    ledger = build_ledger()
    good_evidence = build_evidence("ev1", kind="filing", source_tier="T1")
    causal_claim_good = build_claim("c1", ["ev1"], claim_type="causal")
    ok = build_answer(claims=[causal_claim_good], evidence=[good_evidence])
    results, _ = run_v4_checks(ok, ledger)
    assert _result(results, "R-006").verdict == "PASS"

    weak_evidence = build_evidence("ev2", kind="filing", source_tier="T4")
    causal_claim_bad = build_claim("c2", ["ev2"], claim_type="causal")
    bad = build_answer(claims=[causal_claim_bad], evidence=[weak_evidence])
    results, _ = run_v4_checks(bad, ledger)
    assert _result(results, "R-006").verdict == "WARN"  # R-006 is severity WARN


def test_r007_pass_and_breach() -> None:
    ledger = build_ledger()
    fresh = build_answer(
        decision="BUY",
        quant_metrics={
            "m1": _metric("m1", 1.0, as_of=date(2026, 8, 25), computed_at=datetime(2026, 8, 25, 12))
        },
    )
    results, _ = run_v4_checks(fresh, ledger)
    assert _result(results, "R-007").verdict == "PASS"

    stale = build_answer(
        decision="BUY",
        quant_metrics={
            "m1": _metric("m1", 1.0, as_of=date(2026, 8, 1), computed_at=datetime(2026, 8, 25, 12))
        },
    )
    results, _ = run_v4_checks(stale, ledger)
    assert _result(results, "R-007").verdict == "WARN"


def test_r008_pass_and_breach() -> None:
    ledger = build_ledger()
    ok = build_answer(summary="The portfolio's risk is well diversified.")
    results, _ = run_v4_checks(ok, ledger)
    assert _result(results, "R-008").verdict == "PASS"

    bad = build_answer(summary="This investment is guaranteed to perform well.")
    results, _ = run_v4_checks(bad, ledger)
    assert _result(results, "R-008").verdict == "FAIL"


def test_r010_pass_and_breach() -> None:
    ledger = build_ledger()
    ok = build_answer(confidence=0.9)
    results, _ = run_v4_checks(ok, ledger)
    assert _result(results, "R-010").verdict == "PASS"

    correctly_rewritten = build_answer(confidence=0.2, decision="INSUFFICIENT_EVIDENCE")
    results, _ = run_v4_checks(correctly_rewritten, ledger)
    assert _result(results, "R-010").verdict == "PASS"

    not_rewritten = build_answer(confidence=0.2, decision="BUY")
    results, _ = run_v4_checks(not_rewritten, ledger)
    assert _result(results, "R-010").verdict == "FAIL"


def test_rules_engine_loads_real_yaml_and_matches_predicates() -> None:
    engine = RulesEngine()
    assert set(engine.specs()) == {f"R-{i:03d}" for i in range(1, 11)}


def _rule_entry(rule_id: str, description: str = "x") -> str:
    return (
        f"  - id: {rule_id}\n"
        f"    description: {description}\n"
        "    action: block\n"
        "    severity: BREACH\n"
        "    params: {}\n"
    )


def test_rules_engine_missing_predicate_raises(tmp_path) -> None:
    bad_yaml = tmp_path / "constraints.yaml"
    bad_yaml.write_text("version: 1\nrules:\n" + _rule_entry("R-999"), encoding="utf-8")
    with pytest.raises(RulesConfigError):
        RulesEngine(bad_yaml)


def test_rules_engine_duplicate_id_raises(tmp_path) -> None:
    bad_yaml = tmp_path / "constraints.yaml"
    content = "version: 1\nrules:\n" + _rule_entry("R-001", "x") + _rule_entry("R-001", "y")
    bad_yaml.write_text(content, encoding="utf-8")
    with pytest.raises(RulesConfigError):
        RulesEngine(bad_yaml)
