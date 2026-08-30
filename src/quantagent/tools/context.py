from __future__ import annotations

from datetime import date, datetime
from uuid import uuid4

from quantagent.contracts.metrics import MetricUnit, MetricValue
from quantagent.contracts.provenance import Provenance
from quantagent.data.cache import CacheClient
from quantagent.data.providers.factors import FactorDataProvider
from quantagent.data.providers.fundamentals import FundamentalsProvider
from quantagent.data.providers.prices import PriceProvider
from quantagent.data.repositories.portfolio_repository import PortfolioRepository
from quantagent.rag.retrieval import HybridRetriever


class ToolContext:
    """Shared, request-scoped resource holder + per-call provenance factory.

    Split into a long-lived resource holder (`tenant_id`, `portfolios`,
    `prices`, `fundamentals`, `factors`, `cache` -- constructed once per
    request and safely shared across concurrent DAG branches, since nothing
    on it mutates after construction) and `for_call(...)`, which returns a
    new, cheap, call-scoped `ToolContext` carrying its own frozen
    `tool_name`/`inputs_hash`/`tool_call_id`. This matters because M3's
    orchestrator will run independent tool calls concurrently over one
    shared `ToolContext` instance -- per-call state must never race.
    `build_provenance`/`wrap_metric` are the only places `Provenance` is
    constructed in `tools/` (guideline.md §5: hand-building it "is how
    fields get forgotten").
    """

    def __init__(
        self,
        *,
        tenant_id: str,
        portfolios: PortfolioRepository,
        prices: PriceProvider,
        fundamentals: FundamentalsProvider,
        factors: FactorDataProvider,
        cache: CacheClient,
        retrieval: HybridRetriever | None = None,
        tool_name: str | None = None,
        tool_call_id: str | None = None,
        inputs_hash: str | None = None,
    ) -> None:
        self.tenant_id = tenant_id
        self.portfolios = portfolios
        self.prices = prices
        self.fundamentals = fundamentals
        self.factors = factors
        self.cache = cache
        # Optional, unlike every other resource above: RAG isn't wired in
        # every deployment/test the way prices/fundamentals are.
        # `tools/research.py`'s adapters degrade gracefully to an empty
        # result with a documented limitation when this is None.
        self.retrieval = retrieval
        self._tool_name = tool_name
        self._tool_call_id = tool_call_id
        self._inputs_hash = inputs_hash

    def for_call(
        self, *, tool_name: str, inputs_hash: str, tool_call_id: str | None = None
    ) -> ToolContext:
        """Return a new context bound to one tool invocation. Cheap: shares
        every resource reference, only the three call-scoped fields differ.
        """
        return ToolContext(
            tenant_id=self.tenant_id,
            portfolios=self.portfolios,
            prices=self.prices,
            fundamentals=self.fundamentals,
            factors=self.factors,
            cache=self.cache,
            retrieval=self.retrieval,
            tool_name=tool_name,
            tool_call_id=tool_call_id or f"tc_{uuid4().hex[:10]}",
            inputs_hash=inputs_hash,
        )

    def build_provenance(
        self,
        *,
        as_of: date | None = None,
        data_sources: list[str] | None = None,
        estimator: str | None = None,
        sample_size: int | None = None,
        seed: int | None = None,
        warnings: list[str] | None = None,
    ) -> Provenance:
        """The single place `Provenance` is constructed in tools/. Requires
        the context to be call-bound (`for_call` must have run first);
        raises `RuntimeError` otherwise -- a programming error, not a
        user-facing one, so it isn't a `ToolError` subtype.
        """
        if self._tool_name is None or self._inputs_hash is None or self._tool_call_id is None:
            raise RuntimeError("build_provenance requires a call-bound ToolContext (see for_call)")
        return Provenance(
            tool_call_id=self._tool_call_id,
            tool_name=self._tool_name,
            as_of=as_of if as_of is not None else date.today(),
            computed_at=datetime.now(),
            inputs_hash=self._inputs_hash,
            data_sources=data_sources or [],
            estimator=estimator,
            sample_size=sample_size,
            seed=seed,
            warnings=warnings or [],
        )

    def wrap_metric(
        self,
        metric_id: str,
        value: float,
        unit: MetricUnit,
        method: str,
        *,
        as_of: date | None = None,
        window: str | None = None,
        ci_95: tuple[float, float] | None = None,
        sample_size: int | None = None,
        seed: int | None = None,
        warnings: list[str] | None = None,
        data_sources: list[str] | None = None,
        estimator: str | None = None,
    ) -> MetricValue:
        return MetricValue(
            metric_id=metric_id,
            value=value,
            unit=unit,
            method=method,
            window=window,
            ci_95=ci_95,
            provenance=self.build_provenance(
                as_of=as_of,
                data_sources=data_sources,
                estimator=estimator,
                sample_size=sample_size,
                seed=seed,
                warnings=warnings,
            ),
        )
