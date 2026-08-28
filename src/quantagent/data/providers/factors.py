from __future__ import annotations

import asyncio
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import date
from typing import Protocol, runtime_checkable

import httpx
import pandas as pd

from quantagent.contracts.errors import InsufficientDataError, ProviderUnavailableError
from quantagent.data.cache import CacheClient, compute_inputs_hash

# architecture.md §13.2: "factor loadings 1 day"
FACTOR_CACHE_TTL_S = 86_400

FACTOR_COLUMNS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
RISK_FREE_COLUMN = "rf"

KEN_FRENCH_BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp"
_FF5_DAILY_ARCHIVE = "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
_MOMENTUM_DAILY_ARCHIVE = "F-F_Momentum_Factor_daily_CSV.zip"

_RAW_TO_CANONICAL_COLUMN = {
    "Mkt-RF": "mkt_rf",
    "SMB": "smb",
    "HML": "hml",
    "RMW": "rmw",
    "CMA": "cma",
    "RF": RISK_FREE_COLUMN,
    "Mom": "mom",
}

# Ken French quotes returns in percent (a printed -0.67 is -0.0067) and flags
# missing observations with -99.99 / -999, both stated in each file's own
# preamble. Skipping either conversion is silent and catastrophic: factor betas
# would come back ~100x off with no error anywhere.
_PERCENT_TO_DECIMAL = 100.0
_MISSING_SENTINEL_THRESHOLD = -99.0

_DATA_ROW_PATTERN = re.compile(r"^\s*\d{8}\s*,")
_HTTP_TIMEOUT_S = 30.0


@dataclass(frozen=True, slots=True)
class FactorReturnPanel:
    """Daily Fama-French 5 + momentum factor returns as DECIMALS, not percent.

    `risk_free` is the matching daily risk-free rate shipped in the same
    file: `quant.factors.factor_exposure` regresses *excess* portfolio
    returns, so the caller needs `r_f` on exactly this calendar to build
    them.
    """

    returns: pd.DataFrame
    risk_free: pd.Series
    factors: list[str]
    as_of: date
    source: str
    warnings: list[str]
    n_obs: int


@runtime_checkable
class FactorDataProvider(Protocol):
    async def get_factor_returns(self, start: date, end: date) -> FactorReturnPanel: ...


class KenFrenchFactorDataProvider:
    """`FactorDataProvider` backed by the Kenneth French Data Library.

    Public zipped CSVs, no API key. The whole published history is fetched,
    parsed and cached under one key, then sliced per request: the library
    ships one file per factor set, so a per-window cache key would
    re-download the same ~400 KB for every distinct window.
    """

    def __init__(
        self, cache: CacheClient | None = None, base_url: str = KEN_FRENCH_BASE_URL
    ) -> None:
        self._cache = cache
        self._base_url = base_url.rstrip("/")

    async def get_factor_returns(self, start: date, end: date) -> FactorReturnPanel:
        if start > end:
            raise InsufficientDataError(f"start {start} must not be after end {end}")
        history = await self._load_history()
        return _slice_to_panel(history, start, end)

    async def _load_history(self) -> pd.DataFrame:
        key = _history_cache_key()
        if self._cache is not None:
            cached = await self._cache.get(key)
            if cached is not None:
                return _history_from_cache_bytes(cached)

        history = await self._download_history()

        if self._cache is not None:
            await self._cache.set(key, _history_to_cache_bytes(history), ttl_s=FACTOR_CACHE_TTL_S)
        return history

    async def _download_history(self) -> pd.DataFrame:
        async with httpx.AsyncClient(timeout=_HTTP_TIMEOUT_S, follow_redirects=True) as client:
            ff5_archive, momentum_archive = await asyncio.gather(
                _fetch_archive(client, f"{self._base_url}/{_FF5_DAILY_ARCHIVE}"),
                _fetch_archive(client, f"{self._base_url}/{_MOMENTUM_DAILY_ARCHIVE}"),
            )
        ff5 = _parse_ken_french_csv(_read_only_member(ff5_archive))
        momentum = _parse_ken_french_csv(_read_only_member(momentum_archive))
        return ff5.join(momentum, how="inner")


async def _fetch_archive(client: httpx.AsyncClient, url: str) -> bytes:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise ProviderUnavailableError(f"Ken French request failed for {url}: {exc}") from exc
    return response.content


def _read_only_member(archive: bytes) -> str:
    """Each Ken French zip holds exactly one CSV, but its name tracks the
    file title and has changed before, so read whichever member is present.
    """
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
            return bundle.read(bundle.namelist()[0]).decode("utf-8-sig", errors="replace")
    except (zipfile.BadZipFile, IndexError) as exc:
        raise ProviderUnavailableError(f"Ken French archive is unreadable: {exc}") from exc


def _parse_ken_french_csv(text: str) -> pd.DataFrame:
    """Parse one Ken French daily factor CSV into a decimal-valued,
    DatetimeIndex-ed frame with canonical lowercase column names.

    The data sits inside free prose: a variable-length preamble (4 lines in
    the FF5 file, 14 in the momentum file -- so the count is discovered,
    never hard-coded), a `,Mkt-RF,SMB,...` header row, `YYYYMMDD,...` data
    rows, a blank line, and a copyright footer that will crash date parsing
    if kept.
    """
    lines = text.splitlines()
    header = next((line for line in lines if line.lstrip().startswith(",")), None)
    data_rows = [line for line in lines if _DATA_ROW_PATTERN.match(line)]
    if header is None or not data_rows:
        raise ProviderUnavailableError("Ken French CSV has no recognisable header or data rows")

    frame = pd.read_csv(
        io.StringIO("\n".join([header, *data_rows])),
        index_col=0,
        skipinitialspace=True,
        dtype="float64",
    )
    frame.index = pd.DatetimeIndex(
        pd.to_datetime(frame.index.astype("int64").astype(str), format="%Y%m%d"), name=None
    )
    frame = frame.rename(columns=_RAW_TO_CANONICAL_COLUMN)
    return frame.mask(frame <= _MISSING_SENTINEL_THRESHOLD) / _PERCENT_TO_DECIMAL


def _slice_to_panel(history: pd.DataFrame, start: date, end: date) -> FactorReturnPanel:
    in_window = (history.index >= pd.Timestamp(start)) & (history.index <= pd.Timestamp(end))
    window = history.loc[in_window]
    if window.empty:
        raise InsufficientDataError(
            f"Ken French factor data covers "
            f"{history.index.min().date()}..{history.index.max().date()}, which does not "
            f"overlap the requested window {start}..{end}"
        )

    warnings = _window_warnings(history, start, end)
    complete = window.dropna(how="any")
    if len(complete) < len(window):
        warnings.append(f"dropped {len(window) - len(complete)} days with missing factor values")

    return FactorReturnPanel(
        returns=complete[FACTOR_COLUMNS].astype("float64"),
        risk_free=complete[RISK_FREE_COLUMN].astype("float64"),
        factors=list(FACTOR_COLUMNS),
        as_of=complete.index.max().date(),
        source="ken_french",
        warnings=warnings,
        n_obs=len(complete),
    )


def _window_warnings(history: pd.DataFrame, start: date, end: date) -> list[str]:
    """Ken French publishes with a multi-week lag, so a request ending
    "today" is normal and must be clipped-and-warned, not rejected.
    """
    warnings: list[str] = []
    available_start = history.index.min().date()
    available_end = history.index.max().date()
    if start < available_start:
        warnings.append(f"factor history starts {available_start}; requested start {start}")
    if end > available_end:
        warnings.append(
            f"factor data is published with a lag; requested end {end}, "
            f"clipped to {available_end}"
        )
    return warnings


def _history_cache_key() -> str:
    inputs_hash = compute_inputs_hash(source="ken_french", model="ff5_mom", frequency="daily")
    return f"quantagent:v1:factors:{inputs_hash}"


def _history_to_cache_bytes(history: pd.DataFrame) -> bytes:
    payload = {"history": history.to_json(orient="split", date_format="iso")}
    return json.dumps(payload).encode()


def _history_from_cache_bytes(data: bytes) -> pd.DataFrame:
    payload = json.loads(data.decode())
    frame = pd.read_json(io.StringIO(payload["history"]), orient="split")
    return frame.astype("float64")
