"""verify/numeric_grounding.py -- V2: numeric grounding (architecture.md
§7.2-7.3).

An LLM given a table of correct numbers will still, occasionally, write a
number that is not in the table -- a transposed digit, a plausible-looking
derived ratio, a remembered figure. This is completely detectable
deterministically, so we detect it deterministically: extract every numeric
token from the answer's free text, build the set of numbers the ledger
actually supports (plus a deterministically-derived closure of legitimate
restatements), and flag anything that doesn't match.

Design decisions (see the M4 plan for full rationale):
- Canonical units are a local 3-value taxonomy (ratio/usd/bare), not
  `contracts.metrics.MetricUnit` -- matching never needs to guess between
  count/zscore/days from text alone.
- `1.37x` multiples normalize to "ratio" (a multiple IS a dimensionless
  ratio; including it is strictly more protective than excluding it).
- The unit-conversion closure is one-directional (upward only: V, V*100,
  V*10000) -- every current tool call site uses unit="ratio" for
  proportions, never "pct"/"bps", so only "a ratio legitimately redisplayed
  as a percent/bp figure" is a real need; adding the downward direction
  would only inflate the allowed set speculatively.
- `Ledger.numeric_index` never contains `MetricValue.ci_95` bounds
  (agent/executor.py's flattener only captures `.value`) -- a CI range is a
  legitimate restatement pattern, so this is the one place V2 looks past
  `numeric_index` into `Ledger.calls[].result` directly.
- Count-of-listed-items allowlisting requires adjacency to a count noun
  ("holdings", "positions", etc.), not a blanket small-integer allowlist --
  a blanket rule would mask a genuinely wrong metric that happens to be
  small.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.ledger import Ledger
from quantagent.verify.types import CheckResult

CanonicalUnit = Literal["ratio", "usd", "bare"]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_INT = r"\d{1,3}(?:,\d{3})*|\d+"
_DEC = rf"(?:{_INT})(?:\.\d+)?"
_RANGE_SEP = r"[-–—]"  # noqa: RUF001 -- hyphen, en dash, em dash (intentional literal match)


def _core(tag: str) -> str:
    """One numeric "shape": optional sign, optional leading $, a number, an
    optional range second-number, an optional trailing unit suffix. `tag`
    disambiguates group names between the parenthesized-negative and bare
    alternatives below (Python `re` forbids duplicate group names in one
    pattern).
    """
    return rf"""
        (?P<sign_{tag}>-)?
        (?P<dollar_{tag}>\$)?
        (?P<num1_{tag}>{_DEC})
        (?:\s*{_RANGE_SEP}\s*(?P<num2_{tag}>{_DEC}))?
        (?P<unit_{tag}>%|bps?|x|[BMKT])?
    """


_TOKEN_RE = re.compile(
    rf"""
    (?<!\w)                         # not preceded by a word/digit char (a
                                     # preceding literal "." -- e.g. an
                                     # ellipsis or the end of a prior
                                     # sentence -- is allowed)
    (?:
        \((?P<is_neg>{_core("n")})\)    # negative-in-parentheses, e.g. (3.2%)
        |
        (?P<is_pos>{_core("p")})
    )
    (?![\w])                        # not immediately followed by a word char
    """,
    re.VERBOSE | re.IGNORECASE,
)

_MAGNITUDE = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}

# "Q3" is a quarter label, not a numeric claim -- but the digit is fused
# directly to "Q" with no separating whitespace, so the main _TOKEN_RE's
# `(?<!\w)` lookbehind (which deliberately excludes any digit run fused to a
# preceding word character, e.g. to avoid mis-tokenizing part of an
# identifier) never extracts it as a token in the first place. A dedicated
# pass captures exactly this shape, matching a quarter digit only when the
# quarter allowlist in `_is_allowlisted` can subsequently recognize it via
# its own adjacent-"Q" check.
_QUARTER_TOKEN_RE = re.compile(r"\bQ([1-4])\b", re.IGNORECASE)


@dataclass(frozen=True)
class NumericToken:
    text: str
    span: tuple[int, int]  # offsets into `source_field`'s own string
    source_field: str
    value: float  # canonical numeric value
    unit: CanonicalUnit
    claim_id: str | None = None


def _parse_number(num_str: str) -> float:
    return float(num_str.replace(",", ""))


def _normalize_one(
    num_str: str, unit_raw: str | None, dollar: bool, negative: bool
) -> tuple[float, CanonicalUnit]:
    magnitude = _parse_number(num_str)
    if negative:
        magnitude = -magnitude
    # Magnitude suffixes (K/M/B/T) are case-sensitively upper in _MAGNITUDE;
    # %/bp/bps/x are matched case-insensitively via the lowercased form.
    unit_upper = (unit_raw or "").upper()
    unit_lower = (unit_raw or "").lower()
    if dollar or unit_upper in _MAGNITUDE:
        return magnitude * _MAGNITUDE.get(unit_upper, 1.0), "usd"
    if unit_lower == "%":
        return magnitude / 100.0, "ratio"
    if unit_lower in ("bp", "bps"):
        return magnitude / 10_000.0, "ratio"
    if unit_lower == "x":
        return magnitude, "ratio"
    return magnitude, "bare"


def _tokenize_field(text: str, source_field: str, *, claim_id: str | None) -> list[NumericToken]:
    tokens: list[NumericToken] = []
    for m in _TOKEN_RE.finditer(text):
        tag = "n" if m.group("is_neg") is not None else "p"
        negative = m.group("is_neg") is not None or m.group(f"sign_{tag}") is not None
        dollar = m.group(f"dollar_{tag}") is not None
        unit_raw = m.group(f"unit_{tag}")
        for num_group in (f"num1_{tag}", f"num2_{tag}"):
            raw = m.group(num_group)
            if raw is None:
                continue
            value, unit = _normalize_one(raw, unit_raw, dollar, negative)
            tokens.append(
                NumericToken(
                    text=m.group(0),
                    span=m.span(num_group),
                    source_field=source_field,
                    value=value,
                    unit=unit,
                    claim_id=claim_id,
                )
            )
    for qm in _QUARTER_TOKEN_RE.finditer(text):
        tokens.append(
            NumericToken(
                text=qm.group(1),
                span=qm.span(1),
                source_field=source_field,
                value=float(qm.group(1)),
                unit="bare",
                claim_id=claim_id,
            )
        )
    return tokens


def _iter_sources(answer: AgentAnswer) -> Iterator[tuple[str, str, str | None]]:
    yield answer.summary, "summary", None
    for claim in answer.claims:
        yield claim.text, f"claims[{claim.claim_id}].text", claim.claim_id
    for i, limitation in enumerate(answer.limitations):
        yield limitation, f"limitations[{i}]", None


# ---------------------------------------------------------------------------
# Allowed-set builder (Ledger.numeric_index + deterministic closure)
# ---------------------------------------------------------------------------

_SCALE_FAMILIES: tuple[float, ...] = (1.0, 100.0, 10_000.0)  # identity, "as %", "as bp"
_SIG_FIGS: tuple[int, ...] = (1, 2, 3)


def round_to_sig_figs(value: float, sig_figs: int) -> float:
    """Round `value` to `sig_figs` significant figures. `round()` alone
    rounds to decimal PLACES, not sig figs, hence this.
    """
    if value == 0.0 or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    decimals = sig_figs - 1 - magnitude
    factor = 10.0**decimals  # float base: avoids an int**negative-int surprise
    return round(value * factor) / factor


def _extract_ci_bounds(ledger: Ledger) -> dict[str, tuple[float, float]]:
    """`Ledger.numeric_index` flattens only `MetricValue.value`
    (agent/executor.py::_flatten_metric_values), never `.ci_95`. A CI bound
    is nonetheless a legitimate number to restate (e.g. "expected return of
    8-11%"), so this is the one place V2 looks past `numeric_index` into
    `Ledger.calls[].result` directly -- those are already-dumped dicts
    (`ToolCallRecord.result: dict[str, Any] | None`), so this is a
    dict-shape walk, not the typed pre-dump walk executor.py uses.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for call in ledger.calls:
        _walk_for_ci95(f"{call.call_id}.result", call.result, bounds)
    return bounds


def _walk_for_ci95(prefix: str, obj: Any, out: dict[str, tuple[float, float]]) -> None:
    if isinstance(obj, dict):
        ci = obj.get("ci_95")
        if (
            isinstance(ci, (list, tuple))
            and len(ci) == 2
            and all(isinstance(x, (int, float)) for x in ci)
        ):
            out[prefix] = (float(ci[0]), float(ci[1]))
        for key, value in obj.items():
            _walk_for_ci95(f"{prefix}.{key}", value, out)
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            _walk_for_ci95(f"{prefix}.{idx}", item, out)


@dataclass(frozen=True)
class _Candidate:
    value: float
    source_key: str
    exact: bool  # True -> rel_tol=1e-9 tier; False -> rel_tol=0.005 (display-rounded) tier


def _build_allowed_set(ledger: Ledger) -> tuple[list[_Candidate], dict[str, float]]:
    """Returns (candidates, source_values) where source_values maps every
    key that contributed a candidate back to its own raw value, for
    "nearest ledger value" reporting.
    """
    source_values: dict[str, float] = dict(ledger.numeric_index)
    for key, (lo, hi) in _extract_ci_bounds(ledger).items():
        source_values[f"{key}.ci_95_lo"] = lo
        source_values[f"{key}.ci_95_hi"] = hi

    candidates: list[_Candidate] = []
    for key, raw_value in source_values.items():
        for scale in _SCALE_FAMILIES:
            scaled = raw_value * scale
            candidates.append(_Candidate(scaled, key, exact=True))
            for sig in _SIG_FIGS:
                candidates.append(_Candidate(round_to_sig_figs(scaled, sig), key, exact=False))
    return candidates, source_values


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def _match(token_value: float, candidates: list[_Candidate]) -> _Candidate | None:
    for c in candidates:
        tol = 1e-9 if c.exact else 0.005
        if math.isclose(token_value, c.value, rel_tol=tol, abs_tol=1e-12):
            return c
    return None


def _nearest_candidate(token_value: float, candidates: list[_Candidate]) -> _Candidate | None:
    if not candidates:
        return None
    return min(candidates, key=lambda c: abs(c.value - token_value))


# ---------------------------------------------------------------------------
# Context allowlist (architecture.md §7.3 point 5)
# ---------------------------------------------------------------------------

_MONTH_NAMES = (
    r"january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
_DATE_CONTEXT_RE = re.compile(rf"\b(?:{_MONTH_NAMES})\b|\b\d{{4}}-\d{{2}}-\d{{2}}\b", re.IGNORECASE)
_COUNT_WORDS = (
    r"holdings?|positions?|claims?|items?|sectors?|steps?|tickers?|"
    r"constraints?|limitations?|disclosures?|assets?"
)
_COUNT_CONTEXT_RE = re.compile(rf"^\s*(?:{_COUNT_WORDS})\b", re.IGNORECASE)
_WINDOW_UNIT_RE = re.compile(r"^-(?:day|days|month|months|year|years|week|weeks)\b", re.IGNORECASE)


def _is_allowlisted(token: NumericToken, source_text: str, evidence: Sequence[Any]) -> str | None:
    """Returns the allowlist reason, or None if the token must be grounded."""
    start, end = token.span
    nearby = source_text[max(0, start - 12) : min(len(source_text), end + 12)]
    window_after = source_text[end : end + 20]

    if _DATE_CONTEXT_RE.search(nearby):
        return "calendar date context"

    if token.unit == "bare" and re.fullmatch(r"(?:19|20)\d{2}", token.text):
        return "standalone year"

    if start > 0 and source_text[start - 1] in "Qq" and re.fullmatch(r"[1-4]", token.text):
        return "quarter label"

    if _WINDOW_UNIT_RE.match(window_after):
        return "analysis window length"

    if (
        token.unit == "bare"
        and re.fullmatch(r"\d+", token.text)
        and _COUNT_CONTEXT_RE.match(window_after)
    ):
        return "count of listed items"

    # Ticker-embedded digits: currently vacuous in practice -- US equity
    # tickers used throughout this codebase (yfinance/SEC conventions) are
    # letter-only -- but implemented for robustness against alphanumeric
    # symbols, same spirit as V3's mostly-vacuous-pre-M5 citation layer.
    if (
        start > 0
        and end < len(source_text)
        and source_text[start - 1].isupper()
        and source_text[end].isupper()
    ):
        return "embedded in an alphanumeric ticker-like token"

    # Numbers inside a verified Evidence.excerpt: real code, but mostly
    # vacuous pre-M5 since RAG/real filing-news evidence doesn't exist yet.
    for ev in evidence:
        excerpt = getattr(ev, "excerpt", None)
        if excerpt and token.text in excerpt:
            return f"present in verified excerpt of {ev.evidence_id}"

    return None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def run_v2_numeric_grounding(answer: AgentAnswer, ledger: Ledger) -> list[CheckResult]:
    candidates, source_values = _build_allowed_set(ledger)
    results: list[CheckResult] = []

    for source_text, source_field, claim_id in _iter_sources(answer):
        for token in _tokenize_field(source_text, source_field, claim_id=claim_id):
            reason = _is_allowlisted(token, source_text, answer.evidence)
            if reason is not None:
                results.append(
                    CheckResult(
                        layer="V2",
                        check_id="v2.numeric_grounding",
                        verdict="PASS",
                        message=f"allowlisted ({reason}): '{token.text}'",
                        claim_id=token.claim_id,
                        source_field=token.source_field,
                        offending_text=token.text,
                        span=token.span,
                    )
                )
                continue

            match = _match(token.value, candidates)
            if match is not None:
                results.append(
                    CheckResult(
                        layer="V2",
                        check_id="v2.numeric_grounding",
                        verdict="PASS",
                        message=f"'{token.text}' grounds to {match.source_key}",
                        claim_id=token.claim_id,
                        source_field=token.source_field,
                        offending_text=token.text,
                        span=token.span,
                        nearest_ledger_key=match.source_key,
                        nearest_ledger_value=source_values.get(match.source_key),
                    )
                )
                continue

            nearest = _nearest_candidate(token.value, candidates)
            results.append(
                CheckResult(
                    layer="V2",
                    check_id="v2.numeric_grounding",
                    verdict="FAIL",
                    message=(
                        f"'{token.text}' in {token.source_field} does not match any ledger "
                        "value within tolerance"
                        + (f"; nearest is {nearest.source_key}" if nearest else "")
                    ),
                    claim_id=token.claim_id,
                    source_field=token.source_field,
                    offending_text=token.text,
                    span=token.span,
                    nearest_ledger_key=nearest.source_key if nearest else None,
                    nearest_ledger_value=(
                        source_values.get(nearest.source_key) if nearest else None
                    ),
                )
            )
    return results


def hallucinated_number_rate(check_results: Sequence[CheckResult]) -> float:
    """architecture.md §7.3: unmatched numbers per 1,000 numeric tokens.
    "Numeric tokens" = ALL v2.numeric_grounding CheckResults, including
    allowlisted PASSes (they were still extracted numeric tokens; they just
    didn't need grounding) -- this keeps the denominator well-defined and
    comparable across answers.
    """
    v2 = [r for r in check_results if r.layer == "V2" and r.check_id == "v2.numeric_grounding"]
    if not v2:
        return 0.0
    unmatched = sum(1 for r in v2 if r.verdict == "FAIL")
    return 1000.0 * unmatched / len(v2)
