from __future__ import annotations

from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from quantagent.contracts.evidence import SourceTier
from quantagent.contracts.metrics import MetricUnit, MetricValue
from quantagent.contracts.provenance import Provenance

# ---------------------------------------------------------------- portfolio --


class GetPortfolioInput(BaseModel):
    portfolio_id: str


class PortfolioOutput(BaseModel):
    """Raw entity lookup -- no computation, so no nested MetricValue; the
    top-level `provenance` still records the call for cache/audit purposes.
    """

    portfolio_id: str
    name: str
    base_currency: str
    benchmark_ticker: str
    mandate_constraints: dict[str, Any]
    provenance: Provenance


class GetHoldingsInput(BaseModel):
    portfolio_id: str
    as_of: date | None = None


class HoldingRecord(BaseModel):
    """`quantity`/`cost_basis_usd` are raw ledger facts (passthrough from the
    repository, not computed) -- left as plain floats. `market_value`/`weight`
    require a price fetch + division, so they are `MetricValue`s.
    """

    ticker: str
    quantity: float
    cost_basis_usd: float
    market_value: MetricValue
    weight: MetricValue
    as_of: date


class GetHoldingsOutput(BaseModel):
    portfolio_id: str
    as_of: date
    holdings: list[HoldingRecord]
    provenance: Provenance


class GetTransactionsInput(BaseModel):
    portfolio_id: str
    start: date
    end: date


class TransactionRecord(BaseModel):
    ticker: str
    trade_date: date
    side: Literal["buy", "sell"]
    quantity: float
    price_usd: float


class GetTransactionsOutput(BaseModel):
    portfolio_id: str
    start: date
    end: date
    transactions: list[TransactionRecord]
    provenance: Provenance


# -------------------------------------------------------------------- market --


class GetPricesInput(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=50)
    start: date
    end: date
    adjusted: bool = True


class PriceObservation(BaseModel):
    as_of: date
    prices: dict[str, float]  # ticker -> adjusted close; raw passthrough, not computed


class GetPricesOutput(BaseModel):
    tickers: list[str]
    unresolved_tickers: list[str]
    observations: list[PriceObservation]
    source: str
    provenance: Provenance


class GetReturnsInput(BaseModel):
    tickers: list[str] = Field(min_length=1, max_length=50)
    lookback_days: int = Field(504, ge=2, le=2520)
    kind: Literal["simple", "log"] = "simple"
    # `freq` in architecture.md's abbreviated signature means resampling
    # (weekly/monthly); quant/returns.py has no resampler in M1/M2, so this
    # is deliberately daily-only for v1 -- documented, not silently ignored.


class ReturnObservation(BaseModel):
    as_of: date
    returns: dict[str, float]  # ticker -> return; computed but a structured row, not a scalar


class GetReturnsOutput(BaseModel):
    tickers: list[str]
    kind: Literal["simple", "log"]
    n_obs: int
    observations: list[ReturnObservation]
    warnings: list[str]
    provenance: Provenance


class GetFundamentalsInput(BaseModel):
    ticker: str


class FundamentalsOutput(BaseModel):
    ticker: str
    as_of: date
    sector: str | None
    industry: str | None
    revenue_ttm: MetricValue | None
    net_margin: MetricValue | None
    pe_ratio: MetricValue | None
    source: str
    provenance: Provenance


# ----------------------------------------------------------------- exposure --


class GetSectorExposureInput(BaseModel):
    portfolio_id: str
    scheme: Literal["gics"] = "gics"


class ExposureBucket(BaseModel):
    label: str
    weight: MetricValue
    tickers: list[str]


class GetSectorExposureOutput(BaseModel):
    portfolio_id: str
    scheme: Literal["gics"]
    buckets: list[ExposureBucket]
    unresolved_tickers: list[str]
    provenance: Provenance


class GetFactorExposureInput(BaseModel):
    portfolio_id: str
    model: Literal["ff5_mom"] = "ff5_mom"
    lookback_days: int = Field(756, ge=252, le=2520)


class FactorLoading(BaseModel):
    """`t_stat`/`significant` are regression diagnostics *about* `beta`, not
    a second business metric -- left as plain float/bool inside the record
    that carries `beta`'s own full `MetricValue` (with `estimator="ols_hac"`).
    """

    factor: str
    beta: MetricValue
    t_stat: float
    significant: bool


class GetFactorExposureOutput(BaseModel):
    portfolio_id: str
    model: Literal["ff5_mom"]
    window: str
    loadings: list[FactorLoading]
    r_squared: MetricValue
    idiosyncratic_variance_share: MetricValue
    hac_lags: int
    provenance: Provenance


class GetCorrelationMatrixInput(BaseModel):
    tickers: list[str] = Field(min_length=2, max_length=50)
    lookback_days: int = Field(504, ge=20, le=2520)
    shrinkage: Literal["ledoit_wolf"] = "ledoit_wolf"


class CorrelationRow(BaseModel):
    ticker: str
    correlations: dict[str, float]  # ticker -> corr; a typed matrix row, not a scalar metric


class GetCorrelationMatrixOutput(BaseModel):
    tickers: list[str]
    method: str
    sample_size: int
    rows: list[CorrelationRow]
    provenance: Provenance


class GetConcentrationMetricsInput(BaseModel):
    portfolio_id: str
    top_n: int = Field(5, ge=1, le=50)


class TopHolding(BaseModel):
    ticker: str
    weight: float  # passthrough of the weight already reported by get_holdings


class GetConcentrationMetricsOutput(BaseModel):
    portfolio_id: str
    hhi: MetricValue
    effective_holdings: MetricValue
    top_n_weight: MetricValue
    top_holdings: list[TopHolding]
    provenance: Provenance


# ---------------------------------------------------------------------- risk --


class CalculatePortfolioVarInput(BaseModel):
    portfolio_id: str
    alpha: float = Field(0.95, ge=0.90, le=0.999)
    horizon_days: int = Field(1, ge=1, le=20)
    method: Literal["historical", "parametric", "monte_carlo"] = "historical"
    lookback_days: int = Field(504, ge=250, le=2520)


# calculate_portfolio_var returns MetricValue directly (per the guideline §5
# worked example) -- no wrapper XOutput needed for a single-scalar tool.


class CalculateCvarInput(BaseModel):
    portfolio_id: str
    alpha: float = Field(0.95, ge=0.90, le=0.999)
    lookback_days: int = Field(504, ge=250, le=2520)
    # No `horizon_days`: quant.cvar.portfolio_cvar has no horizon-scaling
    # primitive (unlike var.portfolio_var). Adding VaR_h = VaR_1*sqrt(h)-style
    # scaling here would put a formula in the tool, which the recipe forbids.


class CalculateComponentVarInput(BaseModel):
    portfolio_id: str
    alpha: float = Field(0.95, ge=0.90, le=0.999)
    method: Literal["historical", "parametric"] = "parametric"
    lookback_days: int = Field(504, ge=250, le=2520)
    group_by: Literal["ticker"] = "ticker"
    # group_by=["theme","sector"] needs rules/theme_map.yaml (M3) / is covered
    # by get_sector_exposure already -- ticker-level only for M2.


class ComponentVarEntry(BaseModel):
    ticker: str
    contribution: MetricValue
    share_of_portfolio_var: float
    # left as plain float: it's arithmetic on two already-provenanced
    # MetricValues, not a new estimate -- wrapping it would imply a distinct
    # estimator/sample_size it doesn't have.


class CalculateComponentVarOutput(BaseModel):
    portfolio_id: str
    alpha: float
    method: Literal["historical", "parametric"]
    portfolio_var: MetricValue
    components: list[ComponentVarEntry]
    provenance: Provenance


class CalculateMaxDrawdownInput(BaseModel):
    portfolio_id: str
    lookback_days: int = Field(756, ge=250, le=2520)


class CalculateMaxDrawdownOutput(BaseModel):
    portfolio_id: str
    drawdown: MetricValue
    peak_date: date
    trough_date: date
    recovery_date: date | None
    recovery_duration_days: int | None
    provenance: Provenance


class GetPortfolioBetaInput(BaseModel):
    portfolio_id: str
    benchmark_ticker: str | None = None  # defaults to the portfolio's own benchmark
    lookback_days: int = Field(504, ge=60, le=2520)


class GetPortfolioBetaOutput(BaseModel):
    portfolio_id: str
    benchmark_ticker: str
    beta: MetricValue
    downside_beta: MetricValue
    provenance: Provenance


class CalculateTrackingErrorInput(BaseModel):
    portfolio_id: str
    benchmark_ticker: str | None = None
    lookback_days: int = Field(504, ge=60, le=2520)


# calculate_tracking_error returns MetricValue directly.


# ------------------------------------------------------------------- utility --


class ComputeExpressionInput(BaseModel):
    expr: str = Field(min_length=1, max_length=200)
    refs: dict[str, float] = Field(default_factory=dict)
    unit: MetricUnit = "ratio"
    # `unit` is an addition to architecture.md §3.1's abbreviated signature:
    # the result's unit genuinely varies per call (a ratio, a pct, a usd
    # delta) and the tool has no way to infer it from `expr` alone, so the
    # caller (planner) declares it explicitly rather than the tool guessing.


# compute_expression returns MetricValue directly.


ReportSectionId = Literal[
    "var", "cvar", "component_var", "drawdown", "beta", "tracking_error", "concentration"
]
_ALL_REPORT_SECTIONS: list[ReportSectionId] = [
    "var",
    "cvar",
    "component_var",
    "drawdown",
    "beta",
    "tracking_error",
    "concentration",
]


def _all_report_sections() -> list[ReportSectionId]:
    return list(_ALL_REPORT_SECTIONS)


class GenerateRiskReportInput(BaseModel):
    portfolio_id: str
    sections: list[ReportSectionId] = Field(default_factory=_all_report_sections)
    alpha: float = Field(0.95, ge=0.90, le=0.999)
    horizon_days: int = Field(1, ge=1, le=20)
    lookback_days: int = Field(504, ge=250, le=2520)
    method: Literal["historical", "parametric", "monte_carlo"] = "historical"


class ReportSection(BaseModel):
    section_id: str
    metrics: list[MetricValue]


class ReportArtifact(BaseModel):
    """Deterministic assembly, no LLM prose (architecture.md §4.2 catalogue)."""

    portfolio_id: str
    as_of: date
    generated_at: datetime
    sections: list[ReportSection]
    warnings: list[str]
    provenance: Provenance


# ---------------------------------------------------------------- research (RAG) --

# Shared name constants: `tools/research.py`'s `@registry.tool(name=...)`
# registration and `agent/document_index.py`'s ledger scan must agree on the
# exact same string. Every other tool in this codebase just repeats its own
# literal name once (no shared constant) -- these two are the first tool
# names two independent modules both need to match exactly, which is what
# earns them a named constant here instead (guideline.md §4.3: no magic
# strings in logic).
RETRIEVE_COMPANY_FILINGS = "retrieve_company_filings"
RETRIEVE_FILING_SECTION = "retrieve_filing_section"
SEARCH_RECENT_NEWS = "search_recent_news"
GET_EARNINGS_TRANSCRIPT_SNIPPETS = "get_earnings_transcript_snippets"


class RetrievedFilingChunk(BaseModel):
    """A single retrieved-and-reranked filing passage, citable as `Evidence`
    (architecture.md §4.7). `excerpt`/`char_span` are the ONLY chunk text
    that ever reaches the synthesiser -- deliberately excludes the full
    underlying chunk text (`rag.retrieval.RetrievedChunk.text`), so an
    injected instruction sitting outside the excerpt window is never sent
    to the model at all (a security benefit beyond the injection classifier
    alone).
    """

    chunk_id: str
    ticker: str
    cik: str
    form_type: str
    filed_at: date
    item: str
    section_path: str
    excerpt: str = Field(max_length=300)
    char_span: tuple[int, int]
    source_url: str
    source_tier: SourceTier
    retrieval_score: float


class RetrieveCompanyFilingsInput(BaseModel):
    ticker: str
    form_types: list[str] = Field(default_factory=lambda: ["10-K", "10-Q", "8-K"])
    since: date | None = None
    query: str = Field(min_length=1, description="The research question driving retrieval")


class RetrieveCompanyFilingsOutput(BaseModel):
    ticker: str
    chunks: list[RetrievedFilingChunk]
    provenance: Provenance


class RetrieveFilingSectionInput(BaseModel):
    ticker: str
    form: str
    section: str  # e.g. "item_1a", "item_7"
    query: str = Field(min_length=1, description="The research question driving retrieval")


class RetrieveFilingSectionOutput(BaseModel):
    ticker: str
    form: str
    section: str
    chunks: list[RetrievedFilingChunk]
    provenance: Provenance


class SearchRecentNewsInput(BaseModel):
    tickers: list[str]
    days: int = Field(14, ge=1, le=365)
    min_source_tier: SourceTier | None = None


class SearchRecentNewsOutput(BaseModel):
    """No live news data source is configured in this milestone (see
    `tools/research.py`'s module docstring) -- `chunks` is always empty,
    with the limitation recorded in `provenance.warnings`, mirroring how
    `verify/constraint_rules.py` registers R-005/R-009 as real,
    always-`NOT_APPLICABLE` rules rather than omitting them: the planner's
    tool catalogue and any DAG referencing this tool by name resolve
    against a real, known tool rather than failing plan validation.
    """

    tickers: list[str]
    chunks: list[RetrievedFilingChunk] = Field(default_factory=list)
    provenance: Provenance


class GetEarningsTranscriptSnippetsInput(BaseModel):
    ticker: str
    quarters: list[str]
    query: str = Field(min_length=1)


class GetEarningsTranscriptSnippetsOutput(BaseModel):
    """Permanently degraded stub this milestone -- see `SearchRecentNewsOutput`."""

    ticker: str
    chunks: list[RetrievedFilingChunk] = Field(default_factory=list)
    provenance: Provenance


# --------------------------------------------------------------- optimization (M7) --


class OptimizePortfolioInput(BaseModel):
    portfolio_id: str
    objective: Literal["min_variance", "max_utility", "risk_parity"]
    max_turnover: float | None = Field(None, ge=0.0, le=2.0)
    max_concentration: float | None = Field(None, ge=0.0, le=1.0)
    risk_aversion: float = Field(2.0, ge=0.1, le=100.0)
    target_return: float | None = Field(None, ge=-1.0, le=1.0)
    as_of: date | None = None


class RebalanceTrade(BaseModel):
    ticker: str
    current_weight: float
    target_weight: float
    weight_delta: float
    action: Literal["BUY", "SELL", "HOLD"]
    trade_value_usd: float
    trade_quantity: float


class OptimizePortfolioOutput(BaseModel):
    portfolio_id: str
    objective: str
    as_of: date
    current_expected_return: MetricValue
    current_volatility: MetricValue
    current_sharpe: MetricValue
    target_expected_return: MetricValue
    target_volatility: MetricValue
    target_sharpe: MetricValue
    ex_ante_delta_risk: MetricValue
    trades: list[RebalanceTrade]
    provenance: Provenance


class SimulateTradeImpactInput(BaseModel):
    portfolio_id: str
    target_weights: dict[str, float]
    as_of: date | None = None


class SimulateTradeImpactOutput(BaseModel):
    portfolio_id: str
    as_of: date
    total_trade_value_usd: MetricValue
    estimated_cost_usd: MetricValue
    turnover_pct: MetricValue
    provenance: Provenance
