from __future__ import annotations

from quantagent.agent.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
    ProviderCircuitBreaker,
)


def test_starts_closed_and_allows_calls() -> None:
    breaker = ProviderCircuitBreaker()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.try_reserve_call() is True


def test_opens_after_threshold_consecutive_failures() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=3, cooldown_s=30.0)
    for _ in range(3):
        assert breaker.try_reserve_call() is True
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.try_reserve_call() is False


def test_success_resets_failure_count() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=3, cooldown_s=30.0)
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.CLOSED  # only 2 consecutive since the reset


def test_half_open_after_cooldown_allows_one_probe() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=1, cooldown_s=0.0)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    assert breaker.try_reserve_call() is True  # cooldown elapsed (0.0s) -> half-open probe
    assert breaker.state == CircuitState.HALF_OPEN
    assert breaker.try_reserve_call() is False  # second concurrent probe blocked


def test_half_open_success_closes_circuit() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=1, cooldown_s=0.0)
    breaker.record_failure()
    assert breaker.try_reserve_call() is True
    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED
    assert breaker.try_reserve_call() is True


def test_half_open_failure_reopens_circuit() -> None:
    breaker = ProviderCircuitBreaker(failure_threshold=1, cooldown_s=0.0)
    breaker.record_failure()
    assert breaker.try_reserve_call() is True
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_registry_creates_breakers_lazily_and_reuses_them() -> None:
    registry = CircuitBreakerRegistry()
    first = registry.get("prices")
    second = registry.get("prices")
    other = registry.get("fundamentals")
    assert first is second
    assert first is not other
