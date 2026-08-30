"""agent/synthesizer.py -- the SYNTHESIZE stage (architecture.md §6, §6.2,
§6.4).

Builds the [SYSTEM]/[MANDATE]/[LEDGER]/[RETRIEVED]/[QUESTION] prompt
(guideline.md §7's instruction hierarchy), calls the shared
`get_structured_completion` primitive against a schema that deliberately
excludes `verification` (the LLM never self-grades verification), and
applies the one confidence-calibration rule M3 owns.
"""

from __future__ import annotations

from typing import Any

from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field

from quantagent.config import settings
from quantagent.contracts.answer import AgentAnswer, Decision, RiskLevel
from quantagent.contracts.evidence import Claim, Evidence
from quantagent.contracts.ledger import Ledger
from quantagent.contracts.metrics import MetricValue
from quantagent.contracts.tools import RETRIEVE_COMPANY_FILINGS, RETRIEVE_FILING_SECTION
from quantagent.contracts.verification import ConstraintCheck, VerificationReport
from quantagent.llm.client import LLMCallMetadata, get_structured_completion
from quantagent.llm.prompts import PromptLoader

# Tool calls whose ledger `result` carries retrieved chunk text -- excluded
# from `_ledger_rows_for_prompt`'s [LEDGER] block (I5: [LEDGER] has no
# untrusted-data delimiter; retrieved text must only ever reach the prompt
# through [RETRIEVED]'s delimited block, built by `_retrieved_rows_for_prompt`).
_RAG_TOOL_NAMES = frozenset({RETRIEVE_COMPANY_FILINGS, RETRIEVE_FILING_SECTION})

PROMPT_STAGE = "synthesis"
PROMPT_NAME = "answer"
PROMPT_VERSION = 1
SYNTHESIS_TEMPERATURE = 0.3  # guideline.md §7: <=0.3 for synthesis
DISPLAY_PRECISION = 4  # architecture.md §6.1: ledger values "rounded to display precision"
TOOL_DEGRADED_CONFIDENCE_CAP = 0.60  # architecture.md §6.4's only M3-detectable calibration row


class SynthesisInput(BaseModel):
    """Everything `synthesize_answer` needs, decoupled from the loop's
    internals so it's unit-testable without a live `ToolContext`/executor.
    """

    question: str
    trace_id: str
    ledger: Ledger
    mandate_summary: str | None = None
    degraded: bool = False
    seeded_limitation: str | None = None
    repair_feedback: str | None = None


class _DraftAnswer(BaseModel):
    """`AgentAnswer` minus `verification` -- the LLM must never author its
    own verification verdict; that is VERIFY's job alone. Field set is
    asserted identical to `AgentAnswer` minus `verification` by
    `tests/unit/agent/test_synthesizer.py::
    test_draft_schema_matches_answer_minus_verification`, so a schema change
    to `contracts/answer.py` fails a test immediately instead of drifting.
    """

    trace_id: str
    scope: str
    decision: Decision
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_basis: list[str]
    risk_level: RiskLevel
    horizon: str
    summary: str
    claims: list[Claim]
    evidence: list[Evidence]
    quant_metrics: dict[str, MetricValue]
    constraints_checked: list[ConstraintCheck]
    limitations: list[str] = Field(min_length=1)
    disclosures: list[str]


async def synthesize_answer(
    inp: SynthesisInput, *, client: AsyncAnthropic, prompts: PromptLoader, model: str | None = None
) -> tuple[AgentAnswer, LLMCallMetadata]:
    resolved_model = model or settings.model_synthesizer
    rendered = prompts.render(
        PROMPT_STAGE,
        PROMPT_NAME,
        PROMPT_VERSION,
        trace_id=inp.trace_id,
        mandate_summary=inp.mandate_summary,
        ledger_calls=_ledger_rows_for_prompt(inp.ledger),
        retrieved=_retrieved_rows_for_prompt(inp.ledger),
        degraded=inp.degraded,
        seeded_limitation=inp.seeded_limitation,
        repair_feedback=inp.repair_feedback,
    )
    draft, meta = await get_structured_completion(
        client,
        model=resolved_model,
        system=rendered.text,
        messages=[{"role": "user", "content": inp.question}],
        output_schema=_DraftAnswer,
        prompt_version=rendered.version,
        temperature=SYNTHESIS_TEMPERATURE,
    )

    confidence, confidence_basis, limitations = _apply_confidence_calibration(
        draft.confidence, draft.confidence_basis, draft.limitations, degraded=inp.degraded
    )
    if inp.seeded_limitation is not None and inp.seeded_limitation not in limitations:
        limitations = [*limitations, inp.seeded_limitation]

    answer = AgentAnswer(
        **draft.model_dump(exclude={"confidence", "confidence_basis", "limitations", "trace_id"}),
        trace_id=inp.trace_id,
        confidence=confidence,
        confidence_basis=confidence_basis,
        limitations=limitations,
        # Placeholder: AgentAnswer.verification is non-optional, so the
        # caller (agent/loop.py) must overwrite this with the real
        # VERIFY-stage report before this answer is ever released. A fixed
        # placeholder here is safer than letting the LLM invent one (P1).
        verification=VerificationReport(verdict="PASS", checks=0, warnings=0, repair_attempts=0),
    )
    return answer, meta


def _apply_confidence_calibration(
    confidence_raw: float, confidence_basis: list[str], limitations: list[str], *, degraded: bool
) -> tuple[float, list[str], list[str]]:
    """Deterministic calibrator (architecture.md §6.4, P6): the LLM proposes
    `confidence_raw`; only this function may cap it.

    M3 builds one of §6.4's seven rows: `tool DEGRADED/ERROR -> cap 0.60`
    (folded together with budget exhaustion via the caller's `degraded`
    flag). The other six rows need signals M3 doesn't compute yet (numeric-
    grounding-parsed sample sizes, theme-estimator spread, retrieval scores,
    source-tier checks) -- explicit M4/M5 TODOs, not silently dropped.
    """
    confidence = confidence_raw
    basis = list(confidence_basis)
    caps = list(limitations)
    if degraded and confidence > TOOL_DEGRADED_CONFIDENCE_CAP:
        confidence = TOOL_DEGRADED_CONFIDENCE_CAP
        basis.append(f"tool_degraded_or_budget_exhausted -> cap {TOOL_DEGRADED_CONFIDENCE_CAP}")
        caps.append(
            "Confidence capped: one or more tool calls were degraded/errored, or the "
            "request budget was exhausted before every planned call completed."
        )
    return confidence, basis, caps


def _round_for_display(value: Any, ndigits: int = DISPLAY_PRECISION) -> Any:
    """Recursively rounds every float leaf for prompt display. Only touches
    a copy built for the prompt -- the authoritative ledger passed to VERIFY
    is never mutated by this function.
    """
    if isinstance(value, float):
        return round(value, ndigits)
    if isinstance(value, dict):
        return {k: _round_for_display(v, ndigits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round_for_display(v, ndigits) for v in value]
    return value


def _ledger_rows_for_prompt(ledger: Ledger) -> list[dict[str, Any]]:
    """Plain dicts for the Jinja template to render via its `tojson` filter.
    Explicitly does NOT include raw provider payloads: `ToolCallRecord.result`
    is already the tool's own typed Pydantic output dumped to JSON, never a
    provider's raw response.

    A RAG tool call's `result` carries retrieved chunk excerpts -- untrusted
    text (P5/I5) that must only ever reach the prompt through [RETRIEVED]'s
    delimited block (`_retrieved_rows_for_prompt`), never [LEDGER], which
    has no untrusted-data delimiter. Its `result` is replaced with a bare
    chunk count here; `call_id`/`tool_name`/`status` stay visible for
    traceability.
    """
    rows: list[dict[str, Any]] = []
    for call in ledger.calls:
        if call.tool_name in _RAG_TOOL_NAMES:
            result: Any = (
                {"chunks_returned": len(call.result.get("chunks", []))}
                if call.result is not None
                else None
            )
        else:
            result = _round_for_display(call.result) if call.result is not None else None
        rows.append(
            {
                "call_id": call.call_id,
                "tool_name": call.tool_name,
                "status": call.status,
                "result": result,
                "error": call.error,
            }
        )
    return rows


def _retrieved_rows_for_prompt(ledger: Ledger) -> list[dict[str, Any]]:
    """Every retrieved chunk across every RAG tool call in the ledger,
    reduced to exactly the citable fields (I5: this is the ONLY chunk text
    that ever reaches the synthesiser -- the full underlying chunk text
    isn't even present in the ledger; see
    `contracts.tools.RetrievedFilingChunk`'s docstring).
    """
    rows: list[dict[str, Any]] = []
    for call in ledger.calls:
        if call.tool_name not in _RAG_TOOL_NAMES or call.result is None:
            continue
        for chunk in call.result.get("chunks", []):
            rows.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "ticker": chunk["ticker"],
                    "form_type": chunk["form_type"],
                    "item": chunk["item"],
                    "section_path": chunk["section_path"],
                    "filed_at": chunk["filed_at"],
                    "excerpt": chunk["excerpt"],
                    "char_span": chunk["char_span"],
                    "source_url": chunk["source_url"],
                    "source_tier": chunk["source_tier"],
                    "retrieval_score": chunk["retrieval_score"],
                }
            )
    return rows
