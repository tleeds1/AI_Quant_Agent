from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ["AAA", "BBB", "CCC"]


def build_date_index(n: int, *, start: str = "2020-01-01") -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, periods=n)


def build_weights(**overrides: float) -> pd.Series:
    defaults: dict[str, float] = {"AAA": 0.5, "BBB": 0.3, "CCC": 0.2}
    defaults.update(overrides)
    return pd.Series(defaults)


def build_return_matrix(
    *,
    n_obs: int = 300,
    tickers: list[str] | None = None,
    seed: int = 7,
    daily_vol: float = 0.01,
) -> pd.DataFrame:
    """Deterministic synthetic daily return matrix (fixed seed) -- large enough
    for the sample-size guards without touching real market data.
    """
    resolved_tickers = tickers if tickers is not None else DEFAULT_TICKERS
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0003, scale=daily_vol, size=(n_obs, len(resolved_tickers)))
    return pd.DataFrame(data, index=build_date_index(n_obs), columns=resolved_tickers)


def build_market_returns(*, n_obs: int = 300, seed: int = 11, daily_vol: float = 0.01) -> pd.Series:
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0002, scale=daily_vol, size=n_obs)
    return pd.Series(data, index=build_date_index(n_obs), name="market")


def build_factor_returns(
    *, n_obs: int = 300, seed: int = 13, factor_names: list[str] | None = None
) -> pd.DataFrame:
    resolved_names = (
        factor_names if factor_names is not None else ["mkt", "smb", "hml", "rmw", "cma", "mom"]
    )
    rng = np.random.default_rng(seed)
    data = rng.normal(loc=0.0, scale=0.01, size=(n_obs, len(resolved_names)))
    return pd.DataFrame(data, index=build_date_index(n_obs), columns=resolved_names)


def build_price_panel(
    *,
    n_obs: int = 300,
    tickers: list[str] | None = None,
    seed: int = 7,
    start_price: float = 100.0,
) -> pd.DataFrame:
    """Synthetic adjusted-close price panel with `n_obs` return periods
    (i.e. `n_obs + 1` price rows).
    """
    resolved_tickers = tickers if tickers is not None else DEFAULT_TICKERS
    returns = build_return_matrix(n_obs=n_obs, tickers=resolved_tickers, seed=seed)
    prices = start_price * (1.0 + returns).cumprod()
    first_date = prices.index[0] - pd.tseries.offsets.BDay(1)
    initial_row = pd.DataFrame(
        [[start_price] * len(resolved_tickers)], index=[first_date], columns=resolved_tickers
    )
    return pd.concat([initial_row, prices])
