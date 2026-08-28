from __future__ import annotations

import io
import zipfile
from datetime import date
from unittest.mock import AsyncMock

import httpx
import pytest
import respx

from quantagent.contracts.errors import InsufficientDataError, ProviderUnavailableError
from quantagent.data.providers.factors import (
    FACTOR_CACHE_TTL_S,
    KEN_FRENCH_BASE_URL,
    KenFrenchFactorDataProvider,
)

_FF5_URL = f"{KEN_FRENCH_BASE_URL}/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOM_URL = f"{KEN_FRENCH_BASE_URL}/F-F_Momentum_Factor_daily_CSV.zip"


def _zip_bytes(member_name: str, content: str) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr(member_name, content)
    return buffer.getvalue()


def _ff5_csv(rows: list[str], *, preamble_lines: int = 4) -> str:
    preamble = "\n".join(f"preamble line {i}" for i in range(preamble_lines))
    header = ",Mkt-RF,SMB,HML,RMW,CMA,RF"
    footer = "\nCopyright 2026 Eugene F. Fama and Kenneth R. French"
    return "\n".join([preamble, header, *rows]) + footer


def _mom_csv(rows: list[str], *, preamble_lines: int = 14) -> str:
    preamble = "\n".join(f"preamble line {i}" for i in range(preamble_lines))
    header = ",Mom"
    footer = "\nCopyright 2026 Eugene F. Fama and Kenneth R. French"
    return "\n".join([preamble, header, *rows]) + footer


def _mock_routes(ff5_csv: str, mom_csv: str) -> None:
    respx.get(_FF5_URL).mock(
        return_value=httpx.Response(
            200, content=_zip_bytes("F-F_Research_Data_5_Factors_2x3_daily.csv", ff5_csv)
        )
    )
    respx.get(_MOM_URL).mock(
        return_value=httpx.Response(
            200, content=_zip_bytes("F-F_Momentum_Factor_daily.csv", mom_csv)
        )
    )


@respx.mock
async def test_percent_values_are_converted_to_decimal() -> None:
    _mock_routes(
        _ff5_csv(["20260630,    0.73,   -0.10,   -0.62,   -1.10,   -0.49,    0.01"]),
        _mom_csv(["20260630,    1.06"]),
    )
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2026, 6, 30), date(2026, 6, 30))

    assert panel.returns.loc["2026-06-30", "mkt_rf"] == pytest.approx(0.0073)
    assert panel.risk_free.loc["2026-06-30"] == pytest.approx(0.0001)


@respx.mock
async def test_header_discovery_handles_different_preamble_lengths() -> None:
    _mock_routes(
        _ff5_csv(
            ["20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00"], preamble_lines=4
        ),
        _mom_csv(["20260101,    0.05"], preamble_lines=14),
    )
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))

    assert panel.n_obs == 1


@respx.mock
async def test_missing_sentinel_rows_are_dropped_with_a_warning() -> None:
    _mock_routes(
        _ff5_csv(
            [
                "20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00",
                "20260102,  -99.99,  -99.99,  -99.99,  -99.99,  -99.99,  -99.99",
            ]
        ),
        _mom_csv(["20260101,    0.05", "20260102,    0.05"]),
    )
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 2))

    assert panel.n_obs == 1
    assert any("missing factor values" in w for w in panel.warnings)


@respx.mock
async def test_ff5_and_momentum_inner_join_drops_momentum_only_dates() -> None:
    _mock_routes(
        _ff5_csv(["20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00"]),
        _mom_csv(
            [
                "20260101,    0.05",
                "20260102,    0.05",  # no matching FF5 row -> dropped by inner join
            ]
        ),
    )
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 2))

    assert panel.n_obs == 1


@respx.mock
async def test_end_past_coverage_is_clipped_with_a_warning_not_a_raise() -> None:
    _mock_routes(
        _ff5_csv(["20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00"]),
        _mom_csv(["20260101,    0.05"]),
    )
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2026, 1, 1), date(2026, 12, 31))

    assert panel.n_obs == 1
    assert any("clipped" in w for w in panel.warnings)


@respx.mock
async def test_window_entirely_outside_coverage_raises_insufficient_data() -> None:
    _mock_routes(
        _ff5_csv(["20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00"]),
        _mom_csv(["20260101,    0.05"]),
    )
    provider = KenFrenchFactorDataProvider()

    with pytest.raises(InsufficientDataError):
        await provider.get_factor_returns(date(2020, 1, 1), date(2020, 1, 2))


async def test_start_after_end_raises_insufficient_data() -> None:
    provider = KenFrenchFactorDataProvider()

    with pytest.raises(InsufficientDataError):
        await provider.get_factor_returns(date(2026, 1, 10), date(2026, 1, 1))


@respx.mock
async def test_http_failure_raises_provider_unavailable() -> None:
    respx.get(_FF5_URL).mock(return_value=httpx.Response(500))
    respx.get(_MOM_URL).mock(
        return_value=httpx.Response(
            200, content=_zip_bytes("F-F_Momentum_Factor_daily.csv", _mom_csv(["20260101,0.05"]))
        )
    )
    provider = KenFrenchFactorDataProvider()

    with pytest.raises(ProviderUnavailableError):
        await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))


@respx.mock
async def test_non_zip_bytes_raise_provider_unavailable() -> None:
    respx.get(_FF5_URL).mock(return_value=httpx.Response(200, content=b"not a zip"))
    respx.get(_MOM_URL).mock(
        return_value=httpx.Response(
            200, content=_zip_bytes("F-F_Momentum_Factor_daily.csv", _mom_csv(["20260101,0.05"]))
        )
    )
    provider = KenFrenchFactorDataProvider()

    with pytest.raises(ProviderUnavailableError):
        await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))


@respx.mock
async def test_returns_columns_and_dtype_are_canonical() -> None:
    _mock_routes(
        _ff5_csv(["20260101,    0.10,    0.20,    0.30,    0.40,    0.50,    0.00"]),
        _mom_csv(["20260101,    0.05"]),
    )
    provider = KenFrenchFactorDataProvider()

    panel = await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))

    assert list(panel.returns.columns) == ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
    assert panel.returns.dtypes.eq("float64").all()
    assert panel.risk_free.index.equals(panel.returns.index)


@respx.mock
async def test_cache_hit_issues_zero_http_requests() -> None:
    route_ff5 = respx.get(_FF5_URL).mock(
        return_value=httpx.Response(
            200,
            content=_zip_bytes(
                "F-F_Research_Data_5_Factors_2x3_daily.csv",
                _ff5_csv(["20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00"]),
            ),
        )
    )
    respx.get(_MOM_URL).mock(
        return_value=httpx.Response(
            200, content=_zip_bytes("F-F_Momentum_Factor_daily.csv", _mom_csv(["20260101,0.05"]))
        )
    )
    cache = AsyncMock()
    cache.get.return_value = None
    provider = KenFrenchFactorDataProvider(cache=cache)
    await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))
    assert route_ff5.call_count == 1
    cached_bytes = cache.set.call_args.args[1]

    route_ff5.reset()
    cache.get.return_value = cached_bytes
    await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))

    assert route_ff5.call_count == 0


@respx.mock
async def test_cache_set_uses_configured_ttl() -> None:
    _mock_routes(
        _ff5_csv(["20260101,    0.10,    0.00,    0.00,    0.00,    0.00,    0.00"]),
        _mom_csv(["20260101,0.05"]),
    )
    cache = AsyncMock()
    cache.get.return_value = None
    provider = KenFrenchFactorDataProvider(cache=cache)

    await provider.get_factor_returns(date(2026, 1, 1), date(2026, 1, 1))

    cache.set.assert_awaited_once()
    assert cache.set.call_args.kwargs["ttl_s"] == FACTOR_CACHE_TTL_S
