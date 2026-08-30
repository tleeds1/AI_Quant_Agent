from __future__ import annotations


class QuantAgentError(Exception):
    """Base class for every domain exception in this codebase (guideline.md §4.4)."""


class DataError(QuantAgentError):
    """Raised by data/ when a provider or repository cannot satisfy a request."""


class ProviderUnavailableError(DataError):
    """A market-data or filings provider is unreachable or circuit-broken."""


class StaleDataError(DataError):
    """Available data is older than the caller's freshness tolerance."""


class InsufficientDataError(DataError):
    """Fewer observations are available than a quant estimator requires."""


class UnknownTickerError(DataError):
    """The requested ticker does not resolve in any configured provider."""


class ToolError(QuantAgentError):
    """Raised by tools/ adapters around execution failures."""


class ToolTimeoutError(ToolError):
    """A tool call exceeded its declared p95-latency-derived timeout."""


class ToolValidationError(ToolError):
    """Tool input failed Pydantic validation before execution."""


class BudgetExhaustedError(ToolError):
    """The request's RequestBudget was exhausted before the tool could run."""


class VerificationError(QuantAgentError):
    """Raised by verify/ when a draft answer fails a deterministic check."""


class UngroundedNumberError(VerificationError):
    """A numeric token in free text does not match the ledger (architecture.md §7.3)."""


class InvalidCitationError(VerificationError):
    """An Evidence reference fails citation validity (architecture.md §7.4)."""


class ConstraintBreachError(VerificationError):
    """A blocking rule in the constraint-consistency layer fired (architecture.md §7.5)."""


class LLMError(QuantAgentError):
    """Raised by llm/ around a model call that cannot be turned into a usable result."""


class StructuredOutputError(LLMError):
    """The model's forced tool-use output never validated against the target
    schema, even after the one permitted retry (guideline.md §7).
    """


class GuardrailError(QuantAgentError):
    """Raised by guardrails/ on inbound or outbound policy failures."""


class OutOfScopeError(GuardrailError):
    """The request falls outside the agent's financial-analysis mandate."""


class ProhibitedRequestError(GuardrailError):
    """The request matches a disallowed category (architecture.md §8.1)."""


class InjectionDetectedError(GuardrailError):
    """The injection classifier flagged user input or a retrieved chunk (architecture.md §11.3)."""


class PolicyViolationError(GuardrailError):
    """An outbound answer failed prohibited-language, advice-framing,
    PII-egress or leakage screening (architecture.md §8.2). One leaf covers
    all four categories: the existing hierarchy isn't 1:1 exhaustive per
    inbound category either (rate-limit/jurisdiction share GuardrailError
    directly), so four near-identical outbound leaves would be
    over-engineering relative to how the inbound side was built.
    """
