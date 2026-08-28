"""verify/types.py -- the shared per-check result type every verifier layer
(V1-V5) and the verdict-aggregation orchestrator (verify/verdict.py) build
against (architecture.md §7).

`VerificationReport` (contracts/verification.py) stays the coarse, frozen,
`AgentAnswer`-facing summary -- its own docstring says per-check detail
"lives in verify/ starting M4"; this is that type. Deliberately NOT added to
contracts/ (that module is frozen, same discipline already applied to
Ledger/Evidence/etc. in M2/M3).

`verdict` is a 3-value Literal ("PASS"/"WARN"/"FAIL"), not each layer's own
richer vocabulary (ConstraintStatus has 4 values, the V5 critic has 4
verdict values) -- every layer maps its own vocabulary down to this one at
construction time, so aggregation never has to know V4's rule actions or
V5's entailment vocabulary; it only ever reduces a flat list of
PASS/WARN/FAIL.

Every layer emits a PASS `CheckResult` for each check it ran that did not
fail, not just failures. Required for `hallucinated_number_rate` and
precision/recall to have a true denominator (checks attempted, not just
checks flagged) -- a layer emitting only non-PASS entries would make "0
checks run" indistinguishable from "everything passed," unacceptable in a
verifier whose own reliability is being measured.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

CheckVerdict = Literal["PASS", "WARN", "FAIL"]
Layer = Literal["V1", "V2", "V3", "V4", "V5"]


class CheckResult(BaseModel):
    """One outcome of one check, from one layer, uniform across V1-V5.

    `check_id` is layer-scoped and stable (e.g. "v1.evidence_resolution",
    "v1.metric_ref_resolution", "v1.decision_scope", "v2.numeric_grounding",
    a V3 `f"{evidence_id}:{subcheck}"`, a V4 `rule_id` like "R-001",
    "v5.entailment"/"v5.contradiction") -- used both by the eval harness's
    "caught by the correct layer with the correct check" assertions and for
    reporting.
    """

    layer: Layer
    check_id: str
    verdict: CheckVerdict
    message: str

    # Optional cross-references back into the AgentAnswer being checked.
    claim_id: str | None = None
    evidence_id: str | None = None
    rule_id: str | None = None

    # Where in the answer's free text this check's subject was found, e.g.
    # "summary", "claims[c3].text". `span` is a character offset into THAT
    # field's own string, not a global offset.
    source_field: str | None = None
    offending_text: str | None = None
    span: tuple[int, int] | None = None

    # V2's "nearest ledger value" reporting (architecture.md §7.3 point 6),
    # left generic enough for other layers to reuse if useful.
    nearest_ledger_key: str | None = None
    nearest_ledger_value: float | None = None
