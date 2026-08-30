"""agent/intent.py -- the INTAKE stage (architecture.md §4.2).

Classifies a question into one of four labels and, for `SIMPLE_LOOKUP`,
resolves it to a one-step `Plan` reusing the exact same `validate_plan` the
planner uses -- one validation code path for both the DAG and this
degenerate one-step case, not a second bespoke check.

`RESEARCH` (added M5, RAG) shares `PORTFOLIO_ANALYSIS`'s full DAG-planning
execution path in `agent/loop.py` -- `create_plan` already lists every
registered tool regardless of intent label, and mandate/portfolio loading
is already conditional on `portfolio_id is None`, independent of intent.
The label exists to steer the intent-classification prompt (a
company-research question that needs no portfolio data should not be
answered "OUT_OF_SCOPE") and for observability, not to select a different
code path -- building a second, lighter-weight execution path for it would
duplicate `create_plan`/`validate_plan` for no measured benefit (YAGNI).
"""

from __future__ import annotations

from typing import Any, Literal

import structlog
from anthropic import AsyncAnthropic
from pydantic import BaseModel, Field, model_validator

from quantagent.agent.planner import Plan, PlanStep, validate_plan
from quantagent.config import settings
from quantagent.llm.client import LLMCallMetadata, get_structured_completion
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.registry import ToolRegistry
from quantagent.tools.registry import registry as tools_registry

logger = structlog.get_logger(__name__)

PROMPT_STAGE = "intent"
PROMPT_NAME = "classify"
PROMPT_VERSION = 1
INTENT_TEMPERATURE = 0.0  # guideline.md §7: temperature=0 for classification

IntentLabel = Literal["SIMPLE_LOOKUP", "PORTFOLIO_ANALYSIS", "RESEARCH", "OUT_OF_SCOPE"]


class _DirectToolSelection(BaseModel):
    """One tool + raw args, as the LLM proposed them for a SIMPLE_LOOKUP
    question. A wrong `tool_name` fails this model's own validator, which
    triggers `get_structured_completion`'s own schema-retry -- no bespoke
    retry loop needed here. `args` stays a loose dict at this LLM-schema
    level (the forcing tool's JSON schema can't know in advance which of the
    real tools' input shapes applies); `_finalize` cross-validates `args`
    against the actually-chosen tool via `validate_plan` immediately after
    parsing.
    """

    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class _IntentClassification(BaseModel):
    label: IntentLabel
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(max_length=280)
    direct_tool: _DirectToolSelection | None = None

    @model_validator(mode="after")
    def _direct_tool_matches_label(self) -> _IntentClassification:
        if self.label == "SIMPLE_LOOKUP" and self.direct_tool is None:
            raise ValueError("SIMPLE_LOOKUP requires a direct_tool selection")
        if self.label != "SIMPLE_LOOKUP" and self.direct_tool is not None:
            raise ValueError(f"direct_tool must be omitted for label={self.label!r}")
        return self


class IntentResult(BaseModel):
    """What `agent/intent.py` hands to the loop-owning state machine.
    `direct_tool`, when set, *is* a one-step `Plan` -- the loop feeds it into
    the identical EXECUTE entrypoint used for a full `PLAN`-produced DAG.
    """

    label: IntentLabel
    confidence: float
    rationale: str
    direct_tool: Plan | None
    llm_call: LLMCallMetadata


async def classify_intent(
    question: str,
    *,
    client: AsyncAnthropic,
    prompts: PromptLoader,
    mandate_summary: str | None = None,
    model: str | None = None,
    registry: ToolRegistry = tools_registry,
) -> IntentResult:
    """INTAKE (architecture.md §4.2). One structured-output call. Never
    refuses anything itself -- `OUT_OF_SCOPE` is just a label; the loop's
    REFUSE transition acts on it.
    """
    resolved_model = model or settings.model_intent
    rendered = prompts.render(
        PROMPT_STAGE,
        PROMPT_NAME,
        PROMPT_VERSION,
        mandate_summary=mandate_summary,
        tools=registry.list_tools(),
    )
    parsed, meta = await get_structured_completion(
        client,
        model=resolved_model,
        system=rendered.text,
        messages=[{"role": "user", "content": question}],
        output_schema=_IntentClassification,
        prompt_version=rendered.version,
        temperature=INTENT_TEMPERATURE,
    )
    return _finalize(parsed, meta, question=question, registry=registry)


def _finalize(
    parsed: _IntentClassification, meta: LLMCallMetadata, *, question: str, registry: ToolRegistry
) -> IntentResult:
    if parsed.direct_tool is None:
        return IntentResult(
            label=parsed.label,
            confidence=parsed.confidence,
            rationale=parsed.rationale,
            direct_tool=None,
            llm_call=meta,
        )

    plan = Plan(
        steps=[
            PlanStep(
                id="s1",
                tool=parsed.direct_tool.tool_name,
                args=parsed.direct_tool.args,
                depends_on=[],
            )
        ],
        success_criteria=f"Directly answer: {question}",
    )
    errors = validate_plan(plan, registry=registry)
    if errors:
        logger.info(
            "direct_tool_plan_invalid_downgrading_to_full_plan",
            tool_name=parsed.direct_tool.tool_name,
            errors=[e.code for e in errors],
        )
        return IntentResult(
            label="PORTFOLIO_ANALYSIS",
            confidence=parsed.confidence,
            rationale=f"{parsed.rationale} (direct-tool plan invalid, downgraded)",
            direct_tool=None,
            llm_call=meta,
        )
    return IntentResult(
        label=parsed.label,
        confidence=parsed.confidence,
        rationale=parsed.rationale,
        direct_tool=plan,
        llm_call=meta,
    )
