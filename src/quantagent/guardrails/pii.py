"""guardrails/pii.py -- deterministic PII detection and redaction
(architecture.md §8.1: "account numbers, national IDs ... stripped before
any LLM call").

Matches against the raw, un-normalized string (unlike injection.py/
inbound.py's other checks): redaction must replace substrings in the
original text the caller will go on to use, and SSNs/emails/phone numbers
don't need casefolding or unicode-folding to match reliably.

Third-party full-name detection (architecture.md §8.1's "full names of
third parties") is NOT implemented: reliable name detection needs NER, an
ML dependency forbidden here (guardrails/ must stay deterministic
pattern-matching only). Documented limitation, not a silent gap.
"""

from __future__ import annotations

from pydantic import BaseModel

from quantagent.guardrails.policy import PolicyConfig, get_default_policy


class PIIRedactionResult(BaseModel):
    redacted_text: str
    matched_pattern_ids: list[str]


def redact_pii(text: str, *, policy: PolicyConfig | None = None) -> PIIRedactionResult:
    """Applies every PII pattern's substitution in `rules/policy.yaml`'s
    fixed order. `matched_pattern_ids` never carries the raw matched value,
    only the pattern id, so a redaction result is itself safe to log.
    """
    resolved_policy = policy or get_default_policy()
    redacted = text
    matched_pattern_ids: list[str] = []
    for pii_pattern in resolved_policy.pii_patterns():
        redacted, count = pii_pattern.compiled.subn(pii_pattern.replacement, redacted)
        if count:
            matched_pattern_ids.append(pii_pattern.pattern_id)
    return PIIRedactionResult(redacted_text=redacted, matched_pattern_ids=matched_pattern_ids)
