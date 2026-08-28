from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

from quantagent.data.providers.factors import FactorDataProvider, KenFrenchFactorDataProvider
from quantagent.data.providers.fundamentals import (
    Fundamentals,
    FundamentalsPanel,
    FundamentalsProvider,
    YFinanceFundamentalsProvider,
)
from quantagent.data.providers.prices import PriceProvider, YFinancePriceProvider


def test_yfinance_price_provider_satisfies_price_provider_protocol() -> None:
    assert isinstance(YFinancePriceProvider(), PriceProvider)


def test_dummy_fundamentals_provider_satisfies_protocol() -> None:
    @dataclass
    class DummyFundamentalsProvider:
        async def get_fundamentals(self, ticker: str) -> Fundamentals:
            return Fundamentals(
                ticker=ticker,
                as_of=date(2026, 8, 22),
                sector=None,
                industry=None,
                revenue_ttm=None,
                net_margin=None,
                pe_ratio=None,
                source="dummy",
            )

        async def get_fundamentals_batch(self, tickers: Sequence[str]) -> FundamentalsPanel:
            return FundamentalsPanel(
                fundamentals={},
                tickers=[],
                unresolved_tickers=list(tickers),
                as_of=date(2026, 8, 22),
                source="dummy",
                warnings=[],
            )

    assert isinstance(DummyFundamentalsProvider(), FundamentalsProvider)


def test_yfinance_fundamentals_provider_satisfies_protocol() -> None:
    assert isinstance(YFinanceFundamentalsProvider(), FundamentalsProvider)


def test_ken_french_factor_provider_satisfies_protocol() -> None:
    assert isinstance(KenFrenchFactorDataProvider(), FactorDataProvider)
