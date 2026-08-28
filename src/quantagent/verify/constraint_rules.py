"""verify/constraint_rules.py -- V4: constraint consistency / rules engine
(architecture.md §7.5; guideline.md §7 -- "no business logic in prompts").

YAML (`rules/constraints.yaml`) carries only rule metadata: id, description,
action, severity, numeric/wordlist params. Predicate logic is Python,
registered per rule_id in `_PREDICATES`. A full condition-expression DSL
(the kind `tools/compute_expression.py`'s AST evaluator provides) is
over-engineering here: that evaluator exists because its input is
untrusted/user-composed at runtime; these ~10 rules are fixed and
developer-authored.

Each predicate returns both a `CheckResult` (feeds verdict aggregation) and
a `ConstraintCheck` (the frozen `contracts/verification.py` type that
becomes `AgentAnswer.constraints_checked` -- the same rule evaluation
naturally produces both representations; building `ConstraintCheck`
anywhere else would duplicate rule logic). A rule's `severity` (from YAML)
decides whether a breach's `CheckResult.verdict` is `WARN` or `FAIL`; a
structurally-inapplicable rule (R-005/R-009) always reports
`CheckResult(verdict="PASS")` -- it must never block or warn -- while its
`ConstraintCheck.status` is the real `NOT_APPLICABLE` value, preserving the
"documented, not silently missing" intent precisely where the schema
already supports it.

R-005 and R-009 cannot fire for real today (no theme taxonomy /
get_theme_exposure tool; no simulate_trade_impact tool -- both later
milestones). They are registered as real, always-NOT_APPLICABLE predicates
rather than omitted: the golden-set eval harness and any compliance
reviewer can see both rule IDs exist and are explicitly, testably deferred.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml

from quantagent.contracts.answer import AgentAnswer
from quantagent.contracts.errors import QuantAgentError
from quantagent.contracts.ledger import Ledger
from quantagent.contracts.verification import ConstraintCheck, ConstraintStatus
from quantagent.verify.types import CheckResult, CheckVerdict

_REPO_ROOT = Path(__file__).resolve().parents[3]  # mirrors llm/prompts.py's convention
_DEFAULT_RULES_PATH = _REPO_ROOT / "rules" / "constraints.yaml"

_RuleVerdict = Literal["PASS", "FAIL", "NOT_APPLICABLE"]


class RulesConfigError(QuantAgentError):
    """`rules/constraints.yaml` is malformed, or its rule IDs don't exactly
    match the registered Python predicates. Raised at `RulesEngine`
    construction (load time), not at check time -- a broken rules file
    should fail loudly and immediately, never as a silent per-request skip.
    """


@dataclass(frozen=True)
class RuleSpec:
    rule_id: str
    description: str
    action: str
    severity: Literal["BREACH", "WARN"]
    params: dict[str, Any]


@dataclass(frozen=True)
class RuleContext:
    answer: AgentAnswer
    ledger: Ledger
    mandate_constraints: dict[str, Any]
    spec: RuleSpec


RulePredicate = Callable[[RuleContext], tuple[CheckResult, ConstraintCheck]]


def _load_and_validate(path: Path) -> dict[str, RuleSpec]:
    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "rules" not in raw:
        raise RulesConfigError(f"{path}: expected a mapping with a top-level 'rules' list")

    specs: dict[str, RuleSpec] = {}
    for entry in raw["rules"]:
        rule_id = str(entry["id"])
        if rule_id in specs:
            raise RulesConfigError(f"{path}: duplicate rule id {rule_id!r}")
        severity = str(entry.get("severity", "BREACH"))
        if severity not in ("BREACH", "WARN"):
            raise RulesConfigError(f"{path}: rule {rule_id!r} has invalid severity {severity!r}")
        specs[rule_id] = RuleSpec(
            rule_id=rule_id,
            description=str(entry["description"]),
            action=str(entry["action"]),
            severity=severity,  # type: ignore[arg-type]
            params=dict(entry.get("params") or {}),
        )

    yaml_ids, predicate_ids = set(specs), set(_PREDICATES)
    if yaml_ids != predicate_ids:
        raise RulesConfigError(
            f"{path}: rule IDs and registered predicates must match exactly. "
            f"YAML rules with no predicate: {sorted(yaml_ids - predicate_ids)}. "
            f"Predicates with no YAML rule: {sorted(predicate_ids - yaml_ids)}."
        )
    return specs


class RulesEngine:
    """Loads and validates `rules/constraints.yaml` once. Constructor takes
    an optional `path` (mirrors `llm/prompts.py::PromptLoader`'s
    constructor-injected directory) so tests can point at a fixture file
    without touching the real one.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._specs = _load_and_validate(path or _DEFAULT_RULES_PATH)

    def specs(self) -> dict[str, RuleSpec]:
        return self._specs


_default_engine: RulesEngine | None = None


def _get_default_engine() -> RulesEngine:
    global _default_engine
    if _default_engine is None:
        _default_engine = RulesEngine()
    return _default_engine


# ---------------------------------------------------------------- helpers --


def _make_result(
    spec: RuleSpec,
    *,
    verdict: _RuleVerdict,
    message: str,
    observed: float | None = None,
    limit: float | None = None,
) -> tuple[CheckResult, ConstraintCheck]:
    check_verdict: CheckVerdict
    constraint_status: ConstraintStatus
    if verdict == "NOT_APPLICABLE":
        check_verdict, constraint_status = "PASS", "NOT_APPLICABLE"
    elif verdict == "PASS":
        check_verdict, constraint_status = "PASS", "PASS"
    else:  # a real breach -- severity decides WARN vs FAIL on the CheckResult
        check_verdict = "WARN" if spec.severity == "WARN" else "FAIL"
        constraint_status = "BREACH"

    check_result = CheckResult(
        layer="V4",
        check_id=spec.rule_id,
        verdict=check_verdict,
        message=message,
        rule_id=spec.rule_id,
    )
    constraint_check = ConstraintCheck(
        rule_id=spec.rule_id,
        description=spec.description,
        status=constraint_status,
        observed=observed,
        limit=limit,
    )
    return check_result, constraint_check


def _max_metric_matching(answer: AgentAnswer, pattern: str) -> float | None:
    """`quant_metrics` keys are dynamic metric_ids (e.g. `top_5_weight`,
    `portfolio_var_95_1d` -- confirmed against tools/exposure.py and
    tools/risk.py), never a single fixed key. Matches by regex; if several
    metrics match (e.g. two VaR calls at different alphas), the maximum
    (most conservative / most likely to breach) is used.
    """
    regex = re.compile(pattern)
    values = [mv.value for metric_id, mv in answer.quant_metrics.items() if regex.match(metric_id)]
    return max(values) if values else None


_QUALIFIER_KEYWORDS = (
    "hedge",
    "reduce position size",
    "size down",
    "trim",
    "smaller position",
    "partial position",
    "position size",
)


def _has_hedge_or_size_qualifier(answer: AgentAnswer) -> bool:
    """No formal "qualifier" field exists anywhere in the schema -- this is
    a new, documented convention this rule introduces: satisfied by either
    a hedged claim (`Claim.hedge != "none"`) or sizing language in the
    summary text.
    """
    if any(claim.hedge != "none" for claim in answer.claims):
        return True
    summary_lower = answer.summary.lower()
    return any(kw in summary_lower for kw in _QUALIFIER_KEYWORDS)


def _trading_days_between(a: Any, b: Any) -> int:
    """`numpy.busday_count` (numpy is a general, already-pinned dependency,
    not a `quantagent.quant` import -- `verify/` is forbidden from importing
    `quantagent.quant`, so `quant/calendar.py` cannot be reused here).
    """
    lo, hi = (a, b) if a <= b else (b, a)
    return int(np.busday_count(lo, hi))


# ---------------------------------------------------------------- predicates --


def _predicate_r001(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    breached = ctx.answer.risk_level == "EXTREME" and ctx.answer.decision == "BUY"
    return _make_result(
        ctx.spec,
        verdict="FAIL" if breached else "PASS",
        message=(
            "EXTREME risk_level with a BUY decision."
            if breached
            else "risk_level/decision combination is consistent."
        ),
    )


def _predicate_r002(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    if ctx.answer.decision != "BUY":
        return _make_result(
            ctx.spec, verdict="PASS", message="decision is not BUY; rule not triggered."
        )
    cap = ctx.mandate_constraints.get(ctx.spec.params["mandate_key"])
    if cap is None:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            message=(
                f"no {ctx.spec.params['mandate_key']!r} in mandate_constraints; nothing to breach."
            ),
        )
    observed = _max_metric_matching(ctx.answer, ctx.spec.params["metric_id_pattern"])
    if observed is None:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            message="decision is BUY but no concentration metric was cited; cannot evaluate.",
        )
    if observed <= cap:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=observed,
            limit=cap,
            message=f"concentration {observed:.4f} within mandate cap {cap:.4f}.",
        )
    if _has_hedge_or_size_qualifier(ctx.answer):
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=observed,
            limit=cap,
            message="concentration cap breached but BUY includes a hedge/size qualifier.",
        )
    return _make_result(
        ctx.spec,
        verdict="FAIL",
        observed=observed,
        limit=cap,
        message=(
            f"BUY with concentration {observed:.4f} over cap {cap:.4f}, no hedge/size qualifier."
        ),
    )


def _predicate_r003(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    threshold = float(ctx.spec.params["confidence_threshold"])
    if ctx.answer.confidence <= threshold:
        return _make_result(ctx.spec, verdict="PASS", message="confidence at or below threshold.")
    degraded = [c.call_id for c in ctx.ledger.calls if c.status == "DEGRADED"]
    unhedged = [
        c.claim_id
        for c in ctx.answer.claims
        if c.claim_type == "forward_looking" and c.hedge == "none"
    ]
    if not degraded and not unhedged:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=ctx.answer.confidence,
            limit=threshold,
            message=(
                "high confidence supported: no DEGRADED calls, no unhedged "
                "forward-looking claims."
            ),
        )
    reasons = []
    if degraded:
        reasons.append(f"DEGRADED calls: {degraded}")
    if unhedged:
        reasons.append(f"unhedged forward_looking claims: {unhedged}")
    return _make_result(
        ctx.spec,
        verdict="FAIL",
        observed=ctx.answer.confidence,
        limit=threshold,
        message=(
            f"confidence {ctx.answer.confidence:.2f} > {threshold} despite: "
            f"{'; '.join(reasons)}."
        ),
    )


def _predicate_r004(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    limit = ctx.mandate_constraints.get(ctx.spec.params["mandate_key"])
    if limit is None:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            message=(
                f"no {ctx.spec.params['mandate_key']!r} in mandate_constraints; nothing to breach."
            ),
        )
    observed = _max_metric_matching(ctx.answer, ctx.spec.params["metric_id_pattern"])
    if observed is None:
        return _make_result(
            ctx.spec, verdict="PASS", message="no portfolio VaR metric cited; cannot evaluate."
        )
    if observed <= limit:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=observed,
            limit=limit,
            message=f"VaR {observed:.4f} within mandate limit {limit:.4f}.",
        )
    if ctx.answer.decision in {"REDUCE", "HEDGE", "NO_ACTION"}:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=observed,
            limit=limit,
            message="VaR over limit but decision is risk-reducing.",
        )
    return _make_result(
        ctx.spec,
        verdict="FAIL",
        observed=observed,
        limit=limit,
        message=(
            f"VaR {observed:.4f} exceeds mandate limit {limit:.4f}; "
            f"decision is {ctx.answer.decision!r}."
        ),
    )


def _predicate_r005(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    return _make_result(
        ctx.spec,
        verdict="NOT_APPLICABLE",
        message="theme taxonomy / get_theme_exposure does not exist yet; structurally cannot fire.",
    )


def _predicate_r006(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    required_tiers = set(ctx.spec.params.get("required_tiers", ["T1", "T2"]))
    evidence_by_id = {e.evidence_id: e for e in ctx.answer.evidence}
    violating = [
        claim.claim_id
        for claim in ctx.answer.claims
        if claim.claim_type == "causal"
        and not (
            {evidence_by_id[eid].source_tier for eid in claim.evidence_ids if eid in evidence_by_id}
            & required_tiers
        )
    ]
    if not violating:
        return _make_result(
            ctx.spec, verdict="PASS", message="every causal claim has >=1 T1/T2 evidence."
        )
    return _make_result(
        ctx.spec,
        verdict="FAIL",
        message=(
            f"causal claims without T1/T2 evidence (should be downgraded to hedged): "
            f"{violating}."
        ),
    )


def _predicate_r007(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    if ctx.answer.decision == "NO_ACTION":
        return _make_result(
            ctx.spec, verdict="PASS", message="decision is NO_ACTION; rule not triggered."
        )
    max_days = int(ctx.spec.params["max_staleness_trading_days"])
    fresh = [
        metric_id
        for metric_id, mv in ctx.answer.quant_metrics.items()
        if _trading_days_between(mv.provenance.as_of, mv.provenance.computed_at.date()) <= max_days
    ]
    if fresh:
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=float(len(fresh)),
            limit=float(max_days),
            message=f"fresh metrics within {max_days} trading days: {fresh}.",
        )
    return _make_result(
        ctx.spec,
        verdict="FAIL",
        limit=float(max_days),
        message=(
            f"no cited metric is fresh within {max_days} trading days; "
            f"decision={ctx.answer.decision!r}."
        ),
    )


def _predicate_r008(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    terms: list[str] = ctx.spec.params.get("prohibited_terms", [])
    summary_lower = ctx.answer.summary.lower()
    hits = [t for t in terms if t.lower() in summary_lower]
    if not hits:
        return _make_result(
            ctx.spec, verdict="PASS", message="no prohibited certainty language found."
        )
    return _make_result(
        ctx.spec, verdict="FAIL", message=f"prohibited certainty language found: {hits}."
    )


def _predicate_r009(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    return _make_result(
        ctx.spec,
        verdict="NOT_APPLICABLE",
        message=(
            "simulate_trade_impact (M7 portfolio-optimizer scope) does not exist yet; "
            "cannot fire."
        ),
    )


def _predicate_r010(ctx: RuleContext) -> tuple[CheckResult, ConstraintCheck]:
    threshold = float(ctx.spec.params["confidence_threshold"])
    if ctx.answer.confidence >= threshold:
        return _make_result(ctx.spec, verdict="PASS", message="confidence at or above threshold.")
    if ctx.answer.decision == "INSUFFICIENT_EVIDENCE":
        return _make_result(
            ctx.spec,
            verdict="PASS",
            observed=ctx.answer.confidence,
            limit=threshold,
            message="low confidence correctly rewritten to INSUFFICIENT_EVIDENCE.",
        )
    return _make_result(
        ctx.spec,
        verdict="FAIL",
        observed=ctx.answer.confidence,
        limit=threshold,
        message=(
            f"confidence {ctx.answer.confidence:.2f} < {threshold} but decision is "
            f"{ctx.answer.decision!r}, not INSUFFICIENT_EVIDENCE."
        ),
    )


_PREDICATES: dict[str, RulePredicate] = {
    "R-001": _predicate_r001,
    "R-002": _predicate_r002,
    "R-003": _predicate_r003,
    "R-004": _predicate_r004,
    "R-005": _predicate_r005,
    "R-006": _predicate_r006,
    "R-007": _predicate_r007,
    "R-008": _predicate_r008,
    "R-009": _predicate_r009,
    "R-010": _predicate_r010,
}


# ---------------------------------------------------------------- entrypoint --


def run_v4_checks(
    answer: AgentAnswer,
    ledger: Ledger,
    mandate_constraints: dict[str, Any] | None = None,
    *,
    engine: RulesEngine | None = None,
) -> tuple[list[CheckResult], list[ConstraintCheck]]:
    """architecture.md §7.5. One `(CheckResult, ConstraintCheck)` pair per
    registered rule (10 total: 8 real + R-005/R-009 always-NOT_APPLICABLE
    stubs), `check_id`/`rule_id` == the rule's own id.

    `mandate_constraints`: neither `AgentAnswer` nor `Ledger` carries
    mandate data today; `None`/`{}` when no portfolio is in scope. Every
    rule treats a missing key as "no such constraint configured" -> PASS,
    never UNKNOWN/FAIL (a mandate silent on VaR isn't a VaR breach).
    Convention this rule set introduces since `PortfolioOutput.
    mandate_constraints` had no formalized shape before now:
    `{"max_var_pct": float, "max_concentration_pct": float}`.
    """
    resolved_engine = engine or _get_default_engine()
    resolved_mandate = mandate_constraints or {}
    pairs = [
        _PREDICATES[rule_id](RuleContext(answer, ledger, resolved_mandate, spec))
        for rule_id, spec in resolved_engine.specs().items()
    ]
    pairs.sort(key=lambda pair: pair[0].check_id)  # stable, deterministic order
    check_results = [pair[0] for pair in pairs]
    constraint_checks = [pair[1] for pair in pairs]
    return check_results, constraint_checks
