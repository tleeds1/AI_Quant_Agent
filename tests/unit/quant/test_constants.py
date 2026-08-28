from __future__ import annotations

from scipy import stats

from quantagent.quant.constants import Z_SCORE_95


def test_z_score_95_matches_scipy_norm_ppf() -> None:
    reference = -stats.norm.ppf(0.05)

    assert abs(Z_SCORE_95 - reference) < 1e-4
