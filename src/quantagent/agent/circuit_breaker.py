"""agent/circuit_breaker.py -- one breaker per data provider (architecture.md
§4.2: "circuit breaker per provider").
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

# Fixed, deliberately not settings-driven: a circuit breaker's tuning is an
# operational/reliability concern, not a per-request knob (unlike
# RequestBudget's ceilings, which genuinely vary per tenant plan). 3
# consecutive step-level failures against one provider is "this is not a
# blip" without being trigger-happy on a single bad request; 30s is long
# enough that a flaky-but-recovering provider isn't hammered every request,
# short enough that a short outage doesn't stay open across many requests.
CIRCUIT_FAILURE_THRESHOLD = 3
CIRCUIT_COOLDOWN_S = 30.0


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass(slots=True)
class ProviderCircuitBreaker:
    """CLOSED: calls proceed. OPEN: calls fail fast until `cooldown_s`
    elapses. HALF_OPEN: exactly one probe call is let through; success ->
    CLOSED, failure -> OPEN again with a fresh cooldown.

    `try_reserve_call` is safe to call from multiple steps dispatched in the
    same wave: asyncio is single-threaded and cooperative, and this method
    contains no `await`, so each caller's read-then-mutate runs to
    completion before the next caller's call begins.
    """

    failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD
    cooldown_s: float = CIRCUIT_COOLDOWN_S
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _consecutive_failures: int = field(default=0, init=False)
    _opened_at: float | None = field(default=None, init=False)
    _half_open_probe_in_flight: bool = field(default=False, init=False)

    @property
    def state(self) -> CircuitState:
        return self._state

    def try_reserve_call(self) -> bool:
        """Returns True and reserves the slot if a call may proceed now."""
        if self._state == CircuitState.OPEN:
            if self._opened_at is None or time.monotonic() - self._opened_at < self.cooldown_s:
                return False
            self._state = CircuitState.HALF_OPEN
            self._half_open_probe_in_flight = False

        if self._state == CircuitState.HALF_OPEN:
            if self._half_open_probe_in_flight:
                return False
            self._half_open_probe_in_flight = True
            return True

        return True  # CLOSED

    def record_success(self) -> None:
        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at = None
        self._half_open_probe_in_flight = False

    def record_failure(self) -> None:
        self._half_open_probe_in_flight = False
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            return
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.failure_threshold:
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()


class CircuitBreakerRegistry:
    """One `ProviderCircuitBreaker` per provider name, created lazily.
    Process-wide by default (`provider_circuit_breakers`), mirroring
    `tools.registry.registry`'s own singleton pattern -- state must outlive
    a single ~12s request for "the breaker has been open for N minutes" to
    mean anything. Tests construct their own instance for isolation.
    """

    def __init__(self) -> None:
        self._breakers: dict[str, ProviderCircuitBreaker] = {}

    def get(self, provider: str) -> ProviderCircuitBreaker:
        return self._breakers.setdefault(provider, ProviderCircuitBreaker())


provider_circuit_breakers = CircuitBreakerRegistry()
