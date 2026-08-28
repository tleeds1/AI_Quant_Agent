from __future__ import annotations

import numpy as np
import pandas as pd
from hypothesis import strategies as st

_TICKER_ALPHABET = [f"T{i}" for i in range(20)]


def _tickers(n: int) -> list[str]:
    return _TICKER_ALPHABET[:n]


@st.composite
def long_only_weights(draw: st.DrawFn, min_n: int = 2, max_n: int = 15) -> pd.Series:
    """N assets in [min_n, max_n], each weight in (0, 1], normalized to sum to 1.

    Required for the HHI in [1/N, 1] property, which only holds long-only.
    """
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    raw = draw(
        st.lists(st.floats(min_value=0.001, max_value=1.0, allow_nan=False), min_size=n, max_size=n)
    )
    total = sum(raw)
    return pd.Series([v / total for v in raw], index=_tickers(n))


@st.composite
def bounded_return_matrix(
    draw: st.DrawFn, n_assets: int, min_t: int = 300, max_t: int = 500
) -> pd.DataFrame:
    """T x n_assets matrix, each cell in [-0.10, 0.10], float64, business-day index.

    Bounded daily returns avoid pathological outliers dominating quantile-based
    assertions.
    """
    t = draw(st.integers(min_value=min_t, max_value=max_t))
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    data = rng.uniform(-0.10, 0.10, size=(t, n_assets))
    index = pd.bdate_range("2015-01-01", periods=t)
    return pd.DataFrame(data, index=index, columns=_tickers(n_assets))


@st.composite
def portfolio(
    draw: st.DrawFn, min_n: int = 2, max_n: int = 10, min_t: int = 300, max_t: int = 500
) -> tuple[pd.Series, pd.DataFrame]:
    """A (weights, asset_returns) pair with matching tickers and a shared index --
    the base generator most property tests compose with.
    """
    weights = draw(long_only_weights(min_n=min_n, max_n=max_n))
    returns = draw(bounded_return_matrix(n_assets=len(weights), min_t=min_t, max_t=max_t))
    return weights, returns


@st.composite
def psd_covariance(draw: st.DrawFn, n_assets: int) -> pd.DataFrame:
    """A random symmetric positive-semi-definite n x n covariance matrix,
    constructed as `A @ A.T / n + eps * I` so it is well-conditioned and PSD
    by construction -- used to test algebraic identities without estimation
    noise (e.g. component VaR summing exactly to portfolio VaR).
    """
    seed = draw(st.integers(min_value=0, max_value=2**31 - 1))
    rng = np.random.default_rng(seed)
    a = rng.uniform(-0.05, 0.05, size=(n_assets, n_assets))
    sigma = a @ a.T / n_assets + 1e-4 * np.eye(n_assets)
    tickers = _tickers(n_assets)
    return pd.DataFrame(sigma, index=tickers, columns=tickers)
