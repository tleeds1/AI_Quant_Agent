"""tests/unit/guardrails/test_policy.py"""

from __future__ import annotations

from pathlib import Path

import pytest

from quantagent.guardrails.policy import PolicyConfig, PolicyConfigError, get_default_policy


def test_default_policy_loads_and_compiles() -> None:
    policy = get_default_policy()
    assert policy.prohibited_request_groups()
    assert policy.pii_patterns()
    assert policy.injection_groups()
    assert policy.prohibited_language_patterns()
    assert policy.advice_framing_patterns()
    assert policy.leakage_patterns()


def test_disclosure_template_missing_key_raises() -> None:
    policy = get_default_policy()
    with pytest.raises(PolicyConfigError):
        policy.disclosure_template("does_not_exist")


def test_refusal_template_missing_subcategory_raises() -> None:
    policy = get_default_policy()
    with pytest.raises(PolicyConfigError):
        policy.refusal_template("prohibited_request", "does_not_exist")


def test_refusal_template_missing_category_raises() -> None:
    policy = get_default_policy()
    with pytest.raises(PolicyConfigError):
        policy.refusal_template("does_not_exist")


def test_duplicate_pii_pattern_id_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "policy.yaml"
    bad_file.write_text(
        "pii_patterns:\n"
        "  - {id: dup, pattern: 'a', replacement: '[X]'}\n"
        "  - {id: dup, pattern: 'b', replacement: '[Y]'}\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyConfigError):
        PolicyConfig(path=bad_file)


def test_malformed_yaml_top_level_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "policy.yaml"
    bad_file.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(PolicyConfigError):
        PolicyConfig(path=bad_file)


def test_group_with_no_patterns_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "policy.yaml"
    bad_file.write_text(
        "prohibited_request_patterns:\n  - id: empty_group\n    patterns: []\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyConfigError):
        PolicyConfig(path=bad_file)


def test_duplicate_group_id_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "policy.yaml"
    bad_file.write_text(
        "prohibited_request_patterns:\n"
        "  - id: dup\n    patterns: ['a']\n"
        "  - id: dup\n    patterns: ['b']\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyConfigError):
        PolicyConfig(path=bad_file)


def test_invalid_regex_raises(tmp_path: Path) -> None:
    bad_file = tmp_path / "policy.yaml"
    bad_file.write_text(
        "prohibited_request_patterns:\n  - id: bad_regex\n    patterns: ['(unclosed']\n",
        encoding="utf-8",
    )
    with pytest.raises(PolicyConfigError):
        PolicyConfig(path=bad_file)
