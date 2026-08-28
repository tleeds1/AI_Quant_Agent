"""verify/v5_critic.py -- V5: entailment critique (architecture.md §7.6).

Design resolution -- an apparent spec tension: §7.6 says the critic
receives "only the claim text and its linked evidence -- not the rest of
the answer, and not the synthesiser's reasoning", which reads as strict
per-claim isolation; the same paragraph requires checking "does any claim
contradict another claim in the same answer", impossible without seeing
multiple claims together. Resolved the same way this project has resolved
every prior milestone's doc ambiguity: ONE LLM call per verification pass,
not one call per claim -- far cheaper, and the only way cross-claim
contradiction detection can work at all -- whose prompt lists every
`{claim_id, claim_text, linked_evidence}` tuple but deliberately excludes
`summary`, `confidence`, `decision`, and any synthesiser narrative outside
the claims+evidence list. See prompts/critic/entailment.v1.jinja.

Skips entirely (`[]`) when `answer.claims` is empty -- the safe-fallback and
any zero-claim answer (e.g. INSUFFICIENT_EVIDENCE with nothing asserted)
have nothing to critique, and an LLM call against an empty claim list would
be a wasted, meaningless one.
"""

from __future__ import annotations

from typing import Literal

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from quantagent.config import settings
from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.evidence import Evidence
from quantagent.llm.client import get_structured_completion
from quantagent.llm.prompts import PromptLoader
from quantagent.verify.types import CheckResult, CheckVerdict

PROMPT_STAGE = "critic"
PROMPT_NAME = "entailment"
PROMPT_VERSION = 1
CRITIC_TEMPERATURE = 0.0  # architecture.md §7: "V5 Entailment critique  LLM (temp 0)"

_ClaimVerdictLiteral = Literal["SUPPORTED", "PARTIALLY_SUPPORTED", "UNSUPPORTED", "CONTRADICTED"]
_SeverityLiteral = Literal["low", "medium", "high"]

# §7.6/§7.7: blocking-vs-advisory is purely a function of the critic's own
# verdict value; `severity` is diagnostic metadata and never changes this.
_BLOCKING_VERDICTS: frozenset[str] = frozenset({"UNSUPPORTED", "CONTRADICTED"})


class _ClaimVerdict(BaseModel):
    claim_id: str
    verdict: _ClaimVerdictLiteral
    reason: str = Field(min_length=1, max_length=500)
    severity: _SeverityLiteral


class _ContradictionFinding(BaseModel):
    """A contradiction is inherently about >=2 claims together, so it
    cannot be represented as one claim's own verdict -- a separate
    top-level list rather than forced onto one claim's `reason`.
    """

    claim_ids: list[str] = Field(min_length=2)
    explanation: str = Field(min_length=1, max_length=500)
    severity: _SeverityLiteral


class _CriticResponse(BaseModel):
    claim_verdicts: list[_ClaimVerdict]
    contradictions: list[_ContradictionFinding] = Field(default_factory=list)


def _evidence_row(evidence: Evidence) -> dict[str, str | None]:
    return {
        "evidence_id": evidence.evidence_id,
        "kind": evidence.kind,
        "source_title": evidence.source_title,
        "source_tier": evidence.source_tier,
        "excerpt": evidence.excerpt,
    }


def _claim_rows(answer: AgentAnswer) -> list[dict[str, object]]:
    evidence_by_id = {e.evidence_id: e for e in answer.evidence}
    return [
        {
            "claim_id": claim.claim_id,
            "claim_type": claim.claim_type,
            "hedge": claim.hedge,
            "claim_text": claim.text,
            "evidence": [
                _evidence_row(evidence_by_id[eid])
                for eid in claim.evidence_ids
                if eid in evidence_by_id
            ],
        }
        for claim in answer.claims
    ]


async def run_v5_critique(
    answer: AgentAnswer,
    *,
    client: AsyncAnthropic,
    prompts: PromptLoader,
    model: str | None = None,
) -> list[CheckResult]:
    if not answer.claims:
        return []

    resolved_model = model or settings.model_critic
    rendered = prompts.render(PROMPT_STAGE, PROMPT_NAME, PROMPT_VERSION, claims=_claim_rows(answer))
    critique, _meta = await get_structured_completion(
        client,
        model=resolved_model,
        system=rendered.text,
        messages=[
            {
                "role": "user",
                "content": "Evaluate every claim listed in [CLAIMS] per your instructions.",
            }
        ],
        output_schema=_CriticResponse,
        prompt_version=rendered.version,
        temperature=CRITIC_TEMPERATURE,
    )
    return _to_check_results(critique)


def _reduce_verdict(critic_verdict: _ClaimVerdictLiteral) -> CheckVerdict:
    if critic_verdict in _BLOCKING_VERDICTS:
        return "FAIL"
    if critic_verdict == "PARTIALLY_SUPPORTED":
        return "WARN"
    return "PASS"


def _to_check_results(critique: _CriticResponse) -> list[CheckResult]:
    results: list[CheckResult] = [
        CheckResult(
            layer="V5",
            check_id="v5.entailment",
            verdict=_reduce_verdict(cv.verdict),
            message=f"[{cv.verdict}/{cv.severity}] {cv.reason}",
            claim_id=cv.claim_id,
        )
        for cv in critique.claim_verdicts
    ]
    for finding in critique.contradictions:
        # Always blocking: §7.6 lists cross-claim contradiction alongside
        # the UNSUPPORTED/CONTRADICTED-grade checklist, never as advisory.
        # CheckResult carries one `claim_id`; the rest of the implicated
        # claims are named in `message` for the repair critique and eval
        # assertions -- keeps CheckResult's shape uniform for V1-V5.
        results.append(
            CheckResult(
                layer="V5",
                check_id="v5.contradiction",
                verdict="FAIL",
                message=(
                    f"[CONTRADICTION/{finding.severity}] claims {finding.claim_ids} "
                    f"contradict each other: {finding.explanation}"
                ),
                claim_id=finding.claim_ids[0],
            )
        )
    return results
