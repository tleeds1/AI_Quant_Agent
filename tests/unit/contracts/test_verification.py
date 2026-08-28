from __future__ import annotations

from quantagent.contracts.verification import ConstraintCheck, VerificationReport


def build_constraint_check(**overrides: object) -> ConstraintCheck:
    defaults: dict[str, object] = {
        "rule_id": "R-005",
        "description": "thematic exposure cap",
        "status": "BREACH",
        "observed": 0.38,
        "limit": 0.25,
    }
    defaults.update(overrides)
    return ConstraintCheck(**defaults)


def build_verification_report(**overrides: object) -> VerificationReport:
    defaults: dict[str, object] = {
        "verdict": "PASS_WITH_WARNINGS",
        "checks": 27,
        "warnings": 2,
        "repair_attempts": 0,
    }
    defaults.update(overrides)
    return VerificationReport(**defaults)


def test_constraint_check_round_trips_through_json() -> None:
    original = build_constraint_check()

    restored = ConstraintCheck.model_validate_json(original.model_dump_json())

    assert restored == original


def test_verification_report_round_trips_through_json() -> None:
    original = build_verification_report()

    restored = VerificationReport.model_validate_json(original.model_dump_json())

    assert restored == original
