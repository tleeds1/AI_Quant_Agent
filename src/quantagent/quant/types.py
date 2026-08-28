from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd


@dataclass(frozen=True, slots=True, kw_only=True)
class QuantResult:
    """Provenance-relevant fields every quant/ result carries.

    `tools/` (M2) reads these to assemble a `contracts.Provenance`; quant/
    itself never constructs one (it has no `contracts` dependency on that
    model). `warnings` is content-mutable despite `frozen=True` -- only
    attribute reassignment is blocked -- but quant/ never mutates it after
    construction.
    """

    method: str
    sample_size: int
    seed: int | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True, kw_only=True)
class ScalarResult(QuantResult):
    """A single computed number: beta, downside_beta, apply_stress_scenario."""

    value: float


@dataclass(frozen=True, slots=True, kw_only=True)
class TailRiskResult(QuantResult):
    """historical_var, parametric_var, monte_carlo_var, portfolio_var,
    historical_cvar, portfolio_cvar. `value` is a positive loss fraction.
    """

    value: float
    alpha: float
    horizon_days: int = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class ComponentResult(QuantResult):
    """parametric_component_var, historical_component_var."""

    components: dict[str, float]
    portfolio_value: float


@dataclass(frozen=True, slots=True, kw_only=True)
class CovarianceResult(QuantResult):
    """ledoit_wolf_covariance."""

    matrix: pd.DataFrame
    shrinkage_intensity: float
    t_over_n_ratio: float
    n_assets: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DrawdownResult(QuantResult):
    """max_drawdown. `value` <= 0 always."""

    value: float
    peak_date: date
    trough_date: date
    recovery_date: date | None
    recovery_duration_days: int | None


@dataclass(frozen=True, slots=True, kw_only=True)
class FactorExposureResult(QuantResult):
    """factor_exposure."""

    betas: dict[str, float]
    t_stats: dict[str, float]
    significant: dict[str, bool]
    r_squared: float
    idiosyncratic_variance_share: float
    hac_lags: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ConcentrationResult(QuantResult):
    """portfolio_concentration."""

    hhi: float
    effective_holdings: float
    top_n_weight: float
    top_n: int
