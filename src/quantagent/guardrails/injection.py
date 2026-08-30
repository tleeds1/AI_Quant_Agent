"""guardrails/injection.py -- deterministic prompt-injection classifier
(architecture.md §11.3, §3.5/P5; guideline.md §9 -- "deterministic layers
must not call an LLM", binding per .importlinter's guardrails-obs-cross-
cutting contract).

Single entrypoint, pure function of its input text: used both for inbound
user-input screening (guardrails/inbound.py) and for retrieved-chunk
screening (called from tools/, which is allowed to import guardrails/;
rag/ is not -- .importlinter's rag-scope contract forbids it).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from quantagent.guardrails.normalize import normalize_for_matching
from quantagent.guardrails.policy import PolicyConfig, get_default_policy

InjectionConfidence = Literal["low", "medium", "high"]


class InjectionVerdict(BaseModel):
    is_injection: bool
    matched_group_ids: list[str]
    confidence: InjectionConfidence


def _confidence_for(matched_count: int) -> InjectionConfidence:
    if matched_count >= 3:
        return "high"
    if matched_count == 2:
        return "medium"
    return "low"


def classify_injection(text: str, *, policy: PolicyConfig | None = None) -> InjectionVerdict:
    """Matches `text` against every injection pattern group in
    `rules/policy.yaml`. Confidence scales with the number of distinct
    groups matched, not the number of individual pattern hits within a
    group -- three variants of the same override phrase is still one
    signal, not three.
    """
    resolved_policy = policy or get_default_policy()
    normalized = normalize_for_matching(text)
    matched_group_ids = [
        group.group_id
        for group in resolved_policy.injection_groups()
        if any(pattern.search(normalized) for pattern in group.compiled)
    ]
    return InjectionVerdict(
        is_injection=bool(matched_group_ids),
        matched_group_ids=matched_group_ids,
        confidence=_confidence_for(len(matched_group_ids)),
    )
