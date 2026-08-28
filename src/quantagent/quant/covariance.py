from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from quantagent.contracts.errors import InsufficientDataError
from quantagent.quant.constants import MIN_COVARIANCE_OBSERVATIONS
from quantagent.quant.types import CovarianceResult
from quantagent.quant.validation import as_float64, assert_no_nan


def ledoit_wolf_covariance(asset_returns: pd.DataFrame) -> CovarianceResult:
    """Ledoit-Wolf shrinkage covariance of `asset_returns` (architecture.md §4.4).

    Sample covariance is unstable when T/N < 10, the normal case for a
    retail portfolio with ~2 years of daily data -- the estimator and the
    T/N ratio are recorded on the result for provenance, not hidden. Raises
    `InsufficientDataError` below `MIN_COVARIANCE_OBSERVATIONS`.
    """
    assert_no_nan(asset_returns, context="ledoit_wolf_covariance")
    n_obs, n_assets = asset_returns.shape
    if n_obs < MIN_COVARIANCE_OBSERVATIONS:
        raise InsufficientDataError(
            f"need at least {MIN_COVARIANCE_OBSERVATIONS} observations for covariance "
            f"estimation, got {n_obs}"
        )
    estimator = LedoitWolf().fit(as_float64(asset_returns))
    matrix = pd.DataFrame(
        estimator.covariance_.astype(np.float64),
        index=asset_returns.columns,
        columns=asset_returns.columns,
    )
    return CovarianceResult(
        method="ledoit_wolf",
        sample_size=n_obs,
        matrix=matrix,
        shrinkage_intensity=float(estimator.shrinkage_),
        t_over_n_ratio=n_obs / n_assets,
        n_assets=n_assets,
    )


def correlation_from_covariance(cov: CovarianceResult) -> pd.DataFrame:
    """Correlation matrix D^-1/2 . Sigma . D^-1/2 derived from an already-shrunk
    covariance matrix. PSD by construction (property-tested in
    tests/property/test_covariance_properties.py).
    """
    std = np.sqrt(np.diag(cov.matrix.to_numpy()))
    outer_std = np.outer(std, std)
    correlation = cov.matrix.to_numpy() / outer_std
    return pd.DataFrame(correlation, index=cov.matrix.index, columns=cov.matrix.columns)


def portfolio_variance(weights: pd.Series, cov_matrix: pd.DataFrame) -> float:
    """sigma_p^2 = w^T . Sigma . w.

    Deliberately separated from `ledoit_wolf_covariance` so algebraic
    properties (e.g. adding an uncorrelated asset can't increase variance
    beyond its weighted share) can be tested against a hand-built Sigma
    without estimation noise.
    """
    aligned_weights = weights.reindex(cov_matrix.index).to_numpy(dtype=np.float64)
    sigma = cov_matrix.to_numpy(dtype=np.float64)
    return float(aligned_weights @ sigma @ aligned_weights)
