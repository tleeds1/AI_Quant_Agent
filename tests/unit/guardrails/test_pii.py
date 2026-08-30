"""tests/unit/guardrails/test_pii.py"""

from __future__ import annotations

from quantagent.guardrails.pii import redact_pii


def test_redacts_ssn() -> None:
    result = redact_pii("my ssn is 123-45-6789")
    assert "123-45-6789" not in result.redacted_text
    assert "ssn_us" in result.matched_pattern_ids


def test_redacts_account_number() -> None:
    result = redact_pii("my account #12345678 needs review")
    assert "12345678" not in result.redacted_text
    assert "account_number" in result.matched_pattern_ids


def test_redacts_credit_card() -> None:
    result = redact_pii("card 4111 1111 1111 1111 was charged")
    assert "4111 1111 1111 1111" not in result.redacted_text
    assert "credit_card" in result.matched_pattern_ids


def test_redacts_email() -> None:
    result = redact_pii("contact me at jane.doe@example.com")
    assert "jane.doe@example.com" not in result.redacted_text
    assert "email" in result.matched_pattern_ids


def test_redacts_phone_number() -> None:
    result = redact_pii("call me at (415) 555-0192")
    assert "555-0192" not in result.redacted_text
    assert "phone_us" in result.matched_pattern_ids


def test_no_match_returns_original_text_unchanged() -> None:
    result = redact_pii("what is my portfolio's 1-day VaR at 95%?")
    assert result.redacted_text == "what is my portfolio's 1-day VaR at 95%?"
    assert result.matched_pattern_ids == []


def test_matched_pattern_ids_never_contain_the_raw_value() -> None:
    result = redact_pii("ssn 123-45-6789")
    for pattern_id in result.matched_pattern_ids:
        assert "123-45-6789" not in pattern_id
