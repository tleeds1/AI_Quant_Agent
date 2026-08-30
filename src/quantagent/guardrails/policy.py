"""guardrails/policy.py -- loads and validates `rules/policy.yaml` once
(architecture.md §8; guideline.md §9's "patterns live in rules/policy.yaml").

Mirrors verify/constraint_rules.py::RulesEngine's load-validate-compile-once
discipline: regexes are compiled at construction time (case-insensitive, so
evasion by casing alone doesn't require a separate normalization pass at
match time), and a malformed file raises `PolicyConfigError` immediately
rather than failing silently per-request.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from quantagent.contracts.errors import QuantAgentError

_REPO_ROOT = Path(__file__).resolve().parents[3]  # mirrors verify/constraint_rules.py's convention
_DEFAULT_POLICY_PATH = _REPO_ROOT / "rules" / "policy.yaml"


class PolicyConfigError(QuantAgentError):
    """`rules/policy.yaml` is malformed, or a pattern fails to compile.
    Raised at `PolicyConfig` construction (load time), not at check time.
    """


@dataclass(frozen=True, slots=True)
class PatternGroup:
    group_id: str
    description: str
    compiled: tuple[re.Pattern[str], ...]


@dataclass(frozen=True, slots=True)
class PIIPattern:
    pattern_id: str
    compiled: re.Pattern[str]
    replacement: str


def _compile(pattern: str, *, context: str) -> re.Pattern[str]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise PolicyConfigError(f"{context}: invalid regex {pattern!r}: {exc}") from exc


def _load_pattern_groups(raw: list[dict[str, Any]], *, section: str) -> tuple[PatternGroup, ...]:
    groups = []
    seen_ids: set[str] = set()
    for entry in raw:
        group_id = str(entry["id"])
        if group_id in seen_ids:
            raise PolicyConfigError(f"{section}: duplicate group id {group_id!r}")
        seen_ids.add(group_id)
        patterns = entry.get("patterns") or []
        if not patterns:
            raise PolicyConfigError(f"{section}: group {group_id!r} has no patterns")
        groups.append(
            PatternGroup(
                group_id=group_id,
                description=str(entry.get("description", "")),
                compiled=tuple(_compile(p, context=f"{section}.{group_id}") for p in patterns),
            )
        )
    return tuple(groups)


def _load_pii_patterns(raw: list[dict[str, Any]]) -> tuple[PIIPattern, ...]:
    patterns = []
    seen_ids: set[str] = set()
    for entry in raw:
        pattern_id = str(entry["id"])
        if pattern_id in seen_ids:
            raise PolicyConfigError(f"pii_patterns: duplicate pattern id {pattern_id!r}")
        seen_ids.add(pattern_id)
        patterns.append(
            PIIPattern(
                pattern_id=pattern_id,
                compiled=_compile(str(entry["pattern"]), context=f"pii_patterns.{pattern_id}"),
                replacement=str(entry["replacement"]),
            )
        )
    return tuple(patterns)


def _load_flat_patterns(raw: list[str], *, section: str) -> tuple[re.Pattern[str], ...]:
    return tuple(_compile(p, context=section) for p in raw)


class PolicyConfig:
    """Loads, validates, and compiles `rules/policy.yaml`. Constructor takes
    an optional `path` (mirrors `RulesEngine`) so tests can point at a
    fixture file without touching the real one.
    """

    def __init__(self, path: Path | None = None) -> None:
        resolved_path = path or _DEFAULT_POLICY_PATH
        raw: Any = yaml.safe_load(resolved_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise PolicyConfigError(f"{resolved_path}: expected a top-level mapping")

        self._prohibited_request_groups = _load_pattern_groups(
            raw.get("prohibited_request_patterns") or [], section="prohibited_request_patterns"
        )
        self._pii_patterns = _load_pii_patterns(raw.get("pii_patterns") or [])
        self._injection_groups = _load_pattern_groups(
            raw.get("injection_patterns") or [], section="injection_patterns"
        )
        self._prohibited_language_patterns = _load_flat_patterns(
            raw.get("prohibited_language_patterns") or [], section="prohibited_language_patterns"
        )
        self._advice_framing_patterns = _load_flat_patterns(
            raw.get("advice_framing_patterns") or [], section="advice_framing_patterns"
        )
        leakage_patterns = list(raw.get("leakage_patterns") or [])
        leakage_patterns += [re.escape(p) for p in (raw.get("prompt_marker_phrases") or [])]
        self._leakage_patterns = _load_flat_patterns(leakage_patterns, section="leakage_patterns")
        self._disclosure_templates: dict[str, str] = dict(raw.get("disclosure_templates") or {})
        self._refusal_templates: dict[str, Any] = dict(raw.get("refusal_templates") or {})

    def prohibited_request_groups(self) -> tuple[PatternGroup, ...]:
        return self._prohibited_request_groups

    def pii_patterns(self) -> tuple[PIIPattern, ...]:
        return self._pii_patterns

    def injection_groups(self) -> tuple[PatternGroup, ...]:
        return self._injection_groups

    def prohibited_language_patterns(self) -> tuple[re.Pattern[str], ...]:
        return self._prohibited_language_patterns

    def advice_framing_patterns(self) -> tuple[re.Pattern[str], ...]:
        return self._advice_framing_patterns

    def leakage_patterns(self) -> tuple[re.Pattern[str], ...]:
        return self._leakage_patterns

    def disclosure_template(self, key: str) -> str:
        try:
            return self._disclosure_templates[key]
        except KeyError:
            raise PolicyConfigError(f"no disclosure_templates entry for {key!r}") from None

    def refusal_template(self, category: str, subcategory: str | None = None) -> str:
        entry = self._refusal_templates.get(category)
        if subcategory is not None:
            if not isinstance(entry, dict) or subcategory not in entry:
                raise PolicyConfigError(
                    f"no refusal_templates entry for {category!r}.{subcategory!r}"
                )
            return str(entry[subcategory])
        if not isinstance(entry, str):
            raise PolicyConfigError(f"no refusal_templates entry for {category!r}")
        return entry


_default_policy: PolicyConfig | None = None


def get_default_policy() -> PolicyConfig:
    global _default_policy
    if _default_policy is None:
        _default_policy = PolicyConfig()
    return _default_policy
