from __future__ import annotations

# Annualisation (architecture.md §4.4, "as named constants, never inline")
TRADING_DAYS_PER_YEAR = 252
MONTHS_PER_YEAR = 12
QUARTERS_PER_YEAR = 4

# Reference value only: parametric_var() computes z generically via
# scipy.stats.norm.ppf(1 - alpha) for any alpha; this is the closed-form
# comparison value the alpha=0.95 reference test asserts against.
Z_SCORE_95 = 1.6449

# Sample-size guards (guideline.md §6 rule 4 -- raise, never approximate)
MIN_VAR_OBSERVATIONS = 250
MIN_CVAR_TAIL_OBSERVATIONS = 20
MIN_COVARIANCE_OBSERVATIONS = 20
MIN_BETA_OBSERVATIONS = 60
MIN_FACTOR_REGRESSION_OBSERVATIONS = 60
# Same value as MIN_BETA_OBSERVATIONS today, kept independent: beta and tracking
# error are distinct estimators, and sharing the symbol would let a future
# revision to one silently move the other's guard.
MIN_TRACKING_ERROR_OBSERVATIONS = 60

# Monte Carlo (architecture.md §4.4: "n_sims >= 10_000, fixed seed")
MONTE_CARLO_MIN_SIMULATIONS = 10_000
MONTE_CARLO_DEFAULT_SIMULATIONS = 10_000
MONTE_CARLO_T_DIST_DEFAULT_DOF = 5.0

# Horizon scaling (architecture.md §4.4: "always surfaced as a limitation for h > 10")
HORIZON_SCALING_MAX_RELIABLE_DAYS = 10

# Factor exposure (architecture.md §4.4: "|t| < 2 ... flagged")
FACTOR_TSTAT_SIGNIFICANCE_THRESHOLD = 2.0

# Concentration (architecture.md §4.4: "top-5 weight")
TOP_N_HOLDINGS_CONCENTRATION = 5

# Calendar alignment fill rule (~1 trading week grace for exchange-specific holidays)
CALENDAR_MAX_FORWARD_FILL_DAYS = 5
