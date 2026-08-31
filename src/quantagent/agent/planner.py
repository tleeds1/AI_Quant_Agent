"""agent/planner.py -- the PLAN stage (architecture.md §4.2).

The LLM emits a DAG of tool calls; `validate_plan` -- pure Python, no LLM --
checks it against every malformed-plan category the M3 DoD requires plan
validation to reject before any step ever executes. `create_plan` re-plans
exactly once on a validation failure (guideline.md §7's "retry once with the
validation error appended", applied at the semantic-plan level rather than
`get_structured_completion`'s own lower-level schema-retry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ValidationError

from quantagent.config import settings
from quantagent.contracts.errors import ToolValidationError
from quantagent.llm.client import LLMCallMetadata, LLMClient, get_structured_completion
from quantagent.llm.prompts import PromptLoader
from quantagent.tools.registry import ToolRegistry
from quantagent.tools.registry import registry as tools_registry

PROMPT_STAGE = "planner"
PLAN_PROMPT_NAME = "dag"
REPAIR_PROMPT_NAME = "dag_repair"
PROMPT_VERSION = 1
PLANNING_TEMPERATURE = 0.0  # guideline.md §7: temperature=0 for planning


class PlanStep(BaseModel):
    id: str
    tool: str
    args: dict[str, Any]
    depends_on: list[str]


class Plan(BaseModel):
    steps: list[PlanStep]
    success_criteria: str


@dataclass(frozen=True, slots=True)
class PlanValidationError:
    """One violation found by `validate_plan`. `step_id` is `None` for a
    plan-level violation (e.g. an empty plan or a budget overrun measured
    across the whole DAG) rather than a single step's fault.
    """

    code: str
    message: str
    step_id: str | None = None


def validate_plan(
    plan: Plan,
    *,
    registry: ToolRegistry = tools_registry,
    max_steps: int | None = None,
    max_wall_ms: int | None = None,
    max_usd: float | None = None,
) -> list[PlanValidationError]:
    """Pure Python, no LLM, no I/O. Returns every violation found (not just
    the first) so a repair prompt can address them all in one pass.

    Structural checks (duplicate ids, self/dangling dependencies, cycles,
    unknown tools, invalid args, too many steps) always run. Budget checks
    (critical-path latency, total cost) only run if the DAG is structurally
    sound enough for "critical path" to be well-defined -- a cyclic or
    dangling-reference graph has no meaningful critical path to compute.
    """
    resolved_max_steps = max_steps if max_steps is not None else settings.max_tool_calls
    resolved_max_wall_ms = max_wall_ms if max_wall_ms is not None else settings.max_wall_ms
    resolved_max_usd = max_usd if max_usd is not None else settings.max_usd_per_request

    errors: list[PlanValidationError] = []

    if not plan.steps:
        return [PlanValidationError(code="EMPTY_PLAN", message="plan has no steps")]

    if len(plan.steps) > resolved_max_steps:
        errors.append(
            PlanValidationError(
                code="TOO_MANY_STEPS",
                message=(
                    f"plan has {len(plan.steps)} steps, exceeding max_steps={resolved_max_steps}"
                ),
            )
        )

    seen_ids: set[str] = set()
    for step in plan.steps:
        if step.id in seen_ids:
            errors.append(
                PlanValidationError(
                    code="DUPLICATE_STEP_ID",
                    message=f"step id {step.id!r} used more than once",
                    step_id=step.id,
                )
            )
        seen_ids.add(step.id)

    step_ids = {step.id for step in plan.steps}
    for step in plan.steps:
        if step.id in step.depends_on:
            errors.append(
                PlanValidationError(
                    code="SELF_DEPENDENCY",
                    message=f"step {step.id!r} depends on itself",
                    step_id=step.id,
                )
            )
        for dep in step.depends_on:
            if dep not in step_ids:
                errors.append(
                    PlanValidationError(
                        code="DANGLING_DEPENDENCY",
                        message=f"step {step.id!r} depends on unknown step {dep!r}",
                        step_id=step.id,
                    )
                )

    has_dangling_or_dup = any(
        e.code in {"DANGLING_DEPENDENCY", "DUPLICATE_STEP_ID"} for e in errors
    )
    if not has_dangling_or_dup and _has_cycle(plan):
        errors.append(PlanValidationError(code="CYCLE", message="plan contains a dependency cycle"))

    for step in plan.steps:
        spec = registry.get(step.tool)
        if spec is None:
            errors.append(
                PlanValidationError(
                    code="UNKNOWN_TOOL",
                    message=f"tool {step.tool!r} is not registered",
                    step_id=step.id,
                )
            )
            continue
        try:
            spec.input_model.model_validate(step.args)
        except ValidationError as exc:
            errors.append(
                PlanValidationError(
                    code="INVALID_ARGS",
                    message=f"args for tool {step.tool!r} failed validation: {exc}",
                    step_id=step.id,
                )
            )

    structurally_sound = not any(
        e.code
        in {"UNKNOWN_TOOL", "CYCLE", "DANGLING_DEPENDENCY", "DUPLICATE_STEP_ID", "SELF_DEPENDENCY"}
        for e in errors
    )
    if structurally_sound:
        critical_path_ms = _critical_path_ms(plan, registry)
        if critical_path_ms > resolved_max_wall_ms:
            errors.append(
                PlanValidationError(
                    code="BUDGET_LATENCY_EXCEEDED",
                    message=(
                        f"estimated critical-path latency {critical_path_ms}ms exceeds "
                        f"max_wall_ms={resolved_max_wall_ms}"
                    ),
                )
            )
        specs = [registry.get(step.tool) for step in plan.steps]
        total_cost = sum(spec.est_cost_usd for spec in specs if spec is not None)
        if total_cost > resolved_max_usd:
            errors.append(
                PlanValidationError(
                    code="BUDGET_COST_EXCEEDED",
                    message=(
                        f"estimated total cost ${total_cost:.4f} exceeds max_usd={resolved_max_usd}"
                    ),
                )
            )

    return errors


def _has_cycle(plan: Plan) -> bool:
    steps_by_id = {step.id: step for step in plan.steps}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = dict.fromkeys(steps_by_id, WHITE)

    def visit(step_id: str) -> bool:
        color[step_id] = GRAY
        for dep in steps_by_id[step_id].depends_on:
            if dep not in steps_by_id:
                continue  # dangling dependency, reported separately
            if color[dep] == GRAY:
                return True
            if color[dep] == WHITE and visit(dep):
                return True
        color[step_id] = BLACK
        return False

    return any(color[step_id] == WHITE and visit(step_id) for step_id in steps_by_id)


def _critical_path_ms(plan: Plan, registry: ToolRegistry) -> int:
    """Longest path through the DAG by summed `p95_latency_ms`, NOT a flat
    sum over every step -- independent branches run concurrently (EXECUTE's
    own design), so only the slowest chain of dependencies should count
    against the wall-clock budget.
    """
    steps_by_id = {step.id: step for step in plan.steps}
    memo: dict[str, int] = {}

    def path_length(step_id: str) -> int:
        if step_id in memo:
            return memo[step_id]
        step = steps_by_id[step_id]
        spec = registry.get(step.tool)
        own_latency = spec.p95_latency_ms if spec is not None else 0
        deps = [dep for dep in step.depends_on if dep in steps_by_id]
        longest_dep = max((path_length(dep) for dep in deps), default=0)
        result = own_latency + longest_dep
        memo[step_id] = result
        return result

    return max((path_length(step.id) for step in plan.steps), default=0)


async def create_plan(
    question: str,
    *,
    client: LLMClient,
    prompts: PromptLoader,
    mandate_summary: str | None = None,
    model: str | None = None,
    registry: ToolRegistry = tools_registry,
) -> tuple[Plan, list[LLMCallMetadata]]:
    """One DAG-planning LLM call, `validate_plan`, and -- on failure -- one
    re-plan with the validation errors appended. Raises `ToolValidationError`
    if the repaired plan is still invalid; the plan is never partially
    executed (guideline.md §11's M3 DoD).
    """
    resolved_model = model or settings.model_planner
    tools = registry.list_tools()

    rendered = prompts.render(
        PROMPT_STAGE, PLAN_PROMPT_NAME, PROMPT_VERSION, mandate_summary=mandate_summary, tools=tools
    )
    plan, meta = await get_structured_completion(
        client,
        model=resolved_model,
        system=rendered.text,
        messages=[{"role": "user", "content": question}],
        output_schema=Plan,
        prompt_version=rendered.version,
        temperature=PLANNING_TEMPERATURE,
    )
    metadatas = [meta]

    errors = validate_plan(plan, registry=registry)
    if not errors:
        return plan, metadatas

    repair_rendered = prompts.render(
        PROMPT_STAGE,
        REPAIR_PROMPT_NAME,
        PROMPT_VERSION,
        mandate_summary=mandate_summary,
        tools=tools,
        validation_errors=errors,
    )
    plan, repair_meta = await get_structured_completion(
        client,
        model=resolved_model,
        system=repair_rendered.text,
        messages=[{"role": "user", "content": question}],
        output_schema=Plan,
        prompt_version=repair_rendered.version,
        temperature=PLANNING_TEMPERATURE,
    )
    metadatas.append(repair_meta)

    errors = validate_plan(plan, registry=registry)
    if errors:
        raise ToolValidationError(
            f"plan failed validation after 1 repair attempt: {[e.code for e in errors]}"
        )
    return plan, metadatas
