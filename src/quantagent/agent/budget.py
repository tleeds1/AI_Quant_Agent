"""agent/budget.py -- the request budget controller (architecture.md §4.2:
"RequestBudget(max_tool_calls, max_wall_ms, max_usd), checked before each
step and before repair; exhaustion short-circuits to synthesis with a
budget_exhausted limitation.").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from quantagent.config import settings
from quantagent.contracts.ledger import ToolCallRecord

BUDGET_EXHAUSTED_LIMITATION = (
    "The analysis stopped early because it reached its tool-call, time, or cost "
    "budget; the answer is based on partial results only."
)


@dataclass(slots=True)
class RequestBudget:
    """Constructed once per request and threaded through EXECUTE and (if a
    repair cycle needs to re-check it) the loop's REPAIR gate.

    Mutated only from `execute_plan`'s own single-threaded, post-wave
    bookkeeping (`record_call`) -- never from inside a concurrently
    scheduled step coroutine -- so no lock is needed regardless of how many
    steps ran concurrently in the wave that just completed.
    """

    max_tool_calls: int
    max_wall_ms: int
    max_usd: float
    _start_monotonic: float = field(default_factory=time.monotonic, init=False)
    _calls_made: int = field(default=0, init=False)
    _cost_spent_usd: float = field(default=0.0, init=False)

    @classmethod
    def from_settings(cls) -> RequestBudget:
        return cls(
            max_tool_calls=settings.max_tool_calls,
            max_wall_ms=settings.max_wall_ms,
            max_usd=settings.max_usd_per_request,
        )

    @property
    def calls_made(self) -> int:
        return self._calls_made

    @property
    def cost_spent_usd(self) -> float:
        return self._cost_spent_usd

    @property
    def elapsed_ms(self) -> int:
        return int((time.monotonic() - self._start_monotonic) * 1000)

    def has_remaining_capacity(self) -> bool:
        """Checked before dispatching another wave, and the method the
        loop's REPAIR transition calls before allowing a repair attempt.
        """
        return (
            self._calls_made < self.max_tool_calls
            and self.elapsed_ms < self.max_wall_ms
            and self._cost_spent_usd < self.max_usd
        )

    def record_call(self, record: ToolCallRecord) -> None:
        """Folds one completed step's actual cost into the running total.
        Every dispatched step counts against `max_tool_calls`, regardless of
        its status -- a DEGRADED short-circuit still consumed a DAG-step
        slot even though no network call happened.
        """
        self._calls_made += 1
        self._cost_spent_usd += record.cost_usd

    @staticmethod
    def limitation_text() -> str:
        return BUDGET_EXHAUSTED_LIMITATION
