"""guardrails/normalize.py -- text normalization shared by every
boolean-detection check (injection, prohibited-request, prohibited-language,
advice-framing, leakage). guideline.md §9: "operate on normalised text
(case, unicode, spacing) to resist trivial evasion."

Not used by PII redaction (guardrails/pii.py): redaction must replace
substrings in the *original* string, and SSNs/emails/phone numbers don't
need casefolding to match reliably.
"""

from __future__ import annotations

import re
import unicodedata

# Zero-width space (200B), ZWNJ (200C), ZWJ (200D), BOM-as-ZWNBSP (FEFF) --
# built from code points rather than pasted as literal invisible characters,
# so the set stays legible in any editor/diff.
_ZERO_WIDTH_CODE_POINTS = (0x200B, 0x200C, 0x200D, 0xFEFF)
_ZERO_WIDTH_CHARS = re.compile("[" + "".join(chr(cp) for cp in _ZERO_WIDTH_CODE_POINTS) + "]")
_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_for_matching(text: str) -> str:
    """NFKC-normalize (folds full-width/lookalike unicode to ASCII where
    possible), strip zero-width characters, casefold, and collapse repeated
    whitespace to a single space.
    """
    stripped = _ZERO_WIDTH_CHARS.sub("", text)
    folded = unicodedata.normalize("NFKC", stripped).casefold()
    return _WHITESPACE_RUN.sub(" ", folded).strip()
