from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from unittest.mock import AsyncMock

import pandas as pd

from quantagent.data.providers.factors import FACTOR_COLUMNS, FactorReturnPanel
from quantagent.data.providers.fundamentals import Fundamentals, FundamentalsPanel
from quantagent.data.providers.prices import PricePanel
from quantagent.data.repositories.portfolio_repository import Holding, PortfolioMeta, Transaction
from quantagent.tools.context import ToolContext
from tests.unit.quant.builders import build_date_index, build_factor_returns, build_price_panel

DEFAULT_PORTFOLIO_ID = "pf_1"
DEFAULT_TENANT_ID = "tenant_1"

# FakePriceProvider/FakeFactorDataProvider both ignore the requested start/end
# and always return the same deterministic date range (tests/unit/quant/builders'
# build_date_index("2020-01-01", n)) -- this is what lets asset-return and
# benchmark/factor-return panels align cleanly in beta/tracking-error/factor
# tool tests without a real calendar-negotiation dance.
# 500, not 300: at alpha=0.95 the CVaR tail needs >= MIN_CVAR_TAIL_OBSERVATIONS=20
# (300*0.05=15 is too few), matching the same fix applied to the M1 CLI script.
_DEFAULT_N_OBS = 500


def build_portfolio_meta(**overrides: object) -> PortfolioMeta:
    defaults: dict[str, object] = {
        "id": DEFAULT_PORTFOLIO_ID,
        "tenant_id": DEFAULT_TENANT_ID,
        "name": "Test Portfolio",
        "base_currency": "USD",
        "benchmark_ticker": "SPY",
        "mandate_constraints": {},
    }
    defaults.update(overrides)
    return PortfolioMeta(**defaults)  # type: ignore[arg-type]


def build_holding(**overrides: object) -> Holding:
    defaults: dict[str, object] = {
        "portfolio_id": DEFAULT_PORTFOLIO_ID,
        "ticker": "AAA",
        "quantity": 10.0,
        "cost_basis": 100.0,
        "as_of": date(2026, 8, 22),
    }
    defaults.update(overrides)
    return Holding(**defaults)  # type: ignore[arg-type]


def build_transaction(**overrides: object) -> Transaction:
    defaults: dict[str, object] = {
        "portfolio_id": DEFAULT_PORTFOLIO_ID,
        "ticker": "AAA",
        "side": "buy",
        "quantity": 10.0,
        "price": 100.0,
        "trade_date": date(2026, 1, 5),
    }
    defaults.update(overrides)
    return Transaction(**defaults)  # type: ignore[arg-type]


class FakePortfolioRepository:
    def __init__(
        self,
        portfolio: PortfolioMeta | None = None,
        holdings: list[Holding] | None = None,
        transactions: list[Transaction] | None = None,
    ) -> None:
        self._portfolio = portfolio if portfolio is not None else build_portfolio_meta()
        self._holdings = holdings if holdings is not None else [build_holding()]
        self._transactions = transactions if transactions is not None else []

    async def get_portfolio(self, portfolio_id: str, tenant_id: str) -> PortfolioMeta | None:
        if portfolio_id != self._portfolio.id or tenant_id != self._portfolio.tenant_id:
            return None
        return self._portfolio

    async def get_holdings(
        self, portfolio_id: str, tenant_id: str, as_of: date | None = None
    ) -> list[Holding]:
        if portfolio_id != self._portfolio.id or tenant_id != self._portfolio.tenant_id:
            return []
        return sorted(self._holdings, key=lambda h: h.ticker)

    async def get_transactions(
        self, portfolio_id: str, tenant_id: str, start: date, end: date
    ) -> list[Transaction]:
        if portfolio_id != self._portfolio.id or tenant_id != self._portfolio.tenant_id:
            return []
        matching = [t for t in self._transactions if start <= t.trade_date <= end]
        return sorted(matching, key=lambda t: (t.trade_date, t.ticker))


class FakePriceProvider:
    def __init__(self, n_obs: int = _DEFAULT_N_OBS, seed: int = 7) -> None:
        self._n_obs = n_obs
        self._seed = seed

    async def get_prices(
        self, tickers: Sequence[str], start: date, end: date, *, adjusted: bool = True
    ) -> PricePanel:
        prices = build_price_panel(n_obs=self._n_obs, tickers=list(tickers), seed=self._seed)
        return PricePanel(
            prices=prices,
            tickers=list(tickers),
            unresolved_tickers=[],
            as_of=prices.index[-1].date(),
            source="fake",
            warnings=[],
            n_obs=len(prices),
        )


class FakeFundamentalsProvider:
    def __init__(self, sector_by_ticker: dict[str, str] | None = None) -> None:
        self._sector_by_ticker = sector_by_ticker or {}

    async def get_fundamentals(self, ticker: str) -> Fundamentals:
        return Fundamentals(
            ticker=ticker,
            as_of=date(2026, 8, 22),
            sector=self._sector_by_ticker.get(ticker, "Technology"),
            industry="Software",
            revenue_ttm=1_000_000.0,
            net_margin=0.2,
            pe_ratio=25.0,
            source="fake",
        )

    async def get_fundamentals_batch(self, tickers: Sequence[str]) -> FundamentalsPanel:
        resolved = {t: await self.get_fundamentals(t) for t in tickers}
        return FundamentalsPanel(
            fundamentals=resolved,
            tickers=list(resolved),
            unresolved_tickers=[],
            as_of=date(2026, 8, 22),
            source="fake",
            warnings=[],
        )


class FakeFactorDataProvider:
    def __init__(self, n_obs: int = _DEFAULT_N_OBS, seed: int = 13) -> None:
        self._n_obs = n_obs
        self._seed = seed

    async def get_factor_returns(self, start: date, end: date) -> FactorReturnPanel:
        returns = build_factor_returns(
            n_obs=self._n_obs, seed=self._seed, factor_names=FACTOR_COLUMNS
        )
        risk_free = pd.Series(0.0001, index=build_date_index(self._n_obs))
        return FactorReturnPanel(
            returns=returns,
            risk_free=risk_free,
            factors=list(FACTOR_COLUMNS),
            as_of=returns.index[-1].date(),
            source="fake",
            warnings=[],
            n_obs=len(returns),
        )


def build_tool_context(
    *,
    portfolio: PortfolioMeta | None = None,
    holdings: list[Holding] | None = None,
    transactions: list[Transaction] | None = None,
    sector_by_ticker: dict[str, str] | None = None,
    price_n_obs: int = _DEFAULT_N_OBS,
    factor_n_obs: int = _DEFAULT_N_OBS,
    tenant_id: str | None = None,
) -> ToolContext:
    resolved_tenant = tenant_id or (portfolio.tenant_id if portfolio else DEFAULT_TENANT_ID)
    return ToolContext(
        tenant_id=resolved_tenant,
        portfolios=FakePortfolioRepository(portfolio, holdings, transactions),  # type: ignore[arg-type]
        prices=FakePriceProvider(n_obs=price_n_obs),  # type: ignore[arg-type]
        fundamentals=FakeFundamentalsProvider(sector_by_ticker),  # type: ignore[arg-type]
        factors=FakeFactorDataProvider(n_obs=factor_n_obs),  # type: ignore[arg-type]
        cache=AsyncMock(),
    )
