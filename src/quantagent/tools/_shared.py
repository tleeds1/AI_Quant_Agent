from __future__ import annotations

from datetime import timedelta

import pandas as pd

from quantagent.contracts.errors import ToolValidationError
from quantagent.data.providers.prices import PricePanel, PriceProvider
from quantagent.data.repositories.portfolio_repository import Holding


async def fetch_priced_weights(
    holdings: list[Holding], prices: PriceProvider, *, lookback_days: int = 7
) -> tuple[pd.Series, PricePanel]:
    """Current market-value weights for `holdings`, priced off the latest
    close on/before `holdings[0].as_of`. Raises `ToolValidationError` if
    `holdings` is empty (callers should check before calling; this is a
    defensive floor).
    """
    if not holdings:
        raise ToolValidationError("cannot compute weights for an empty holdings list")
    as_of = holdings[0].as_of
    tickers = [h.ticker for h in holdings]
    panel = await prices.get_prices(tickers, start=as_of - timedelta(days=lookback_days), end=as_of)
    latest = panel.prices.iloc[-1]
    market_values = pd.Series({h.ticker: h.quantity * float(latest[h.ticker]) for h in holdings})
    return market_values / market_values.sum(), panel
