"""X-only feature coverage rules for future research; no alpha screening."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_groups import OUTCOME_COLUMNS, feature_family

LAGGED_OUTCOME_COLUMNS = (
    "lagged_forward_returns", "lagged_market_forward_excess_returns", "lagged_risk_free_rate",
)


def research_candidate_columns(columns, allow_lagged: bool = True) -> list[str]:
    """94 anonymous inputs plus 3 competition-available lagged outcomes, if present."""
    return [c for c in columns if feature_family(c) or (allow_lagged and c in LAGGED_OUTCOME_COLUMNS)]


def available_lagged_outcomes(history: pd.DataFrame) -> pd.DataFrame:
    """Recreate API-available one-row lags from an ordered historical stream.

    Only software tests call this helper now. A future caller must shift before
    selecting a validation block, using earlier observed history, never test.csv.
    """
    if not OUTCOME_COLUMNS.issubset(history.columns) or "date_id" not in history:
        raise ValueError("Ordered dates and all original outcomes are required.")
    dates = history["date_id"]
    if dates.isna().any() or dates.duplicated().any() or not dates.diff().dropna().eq(1).all():
        raise ValueError("Lag generation requires consecutive ordered trading IDs.")
    return pd.DataFrame({f"lagged_{c}": history[c].shift(1) for c in sorted(OUTCOME_COLUMNS)}, index=history.index)


def feature_availability(training_x: pd.DataFrame, policy: dict) -> pd.DataFrame:
    """Assess only the supplied training window, without Y or later observations.

    Coverage since first observation distinguishes unavailable pre-launch
    history from sporadic missing data. Recent coverage prevents stale inputs.
    No field is dropped from source data by this diagnostic tool.
    """
    if (training_x.empty or list(training_x.columns) != research_candidate_columns(training_x.columns)
            or training_x.columns.duplicated().any()):
        raise ValueError("Availability accepts nonempty candidate X only, never current outcomes or metadata.")
    if not training_x.index.is_unique or not training_x.index.is_monotonic_increasing:
        raise ValueError("Training X must have ordered unique row indices.")
    recent_window = policy["recent_window"]
    if (policy["min_nonmissing_observations"] < 1 or recent_window < 1
            or not 0 <= policy["min_historical_coverage_since_first_observation"] <= 1
            or not 0 <= policy["max_recent_missing_ratio"] <= 1):
        raise ValueError("Invalid availability thresholds.")
    rows = []
    for column in training_x:
        series = training_x[column]
        if not pd.api.types.is_numeric_dtype(series) or np.isinf(series.to_numpy(dtype=float)).any():
            raise ValueError("Candidate X must be numeric and finite when observed.")
        observed = series.notna().to_numpy()
        count = int(observed.sum())
        first = int(np.flatnonzero(observed)[0]) if count else None
        coverage = count / (len(series) - first) if count else 0.0
        recent_missing = float(series.iloc[-recent_window:].isna().mean())
        distinct = int(series.nunique(dropna=True))
        reasons = []
        if count < policy["min_nonmissing_observations"]:
            reasons.append("insufficient_observations")
        if coverage < policy["min_historical_coverage_since_first_observation"]:
            reasons.append("insufficient_post_launch_coverage")
        if recent_missing > policy["max_recent_missing_ratio"]:
            reasons.append("excess_recent_missingness")
        if distinct < policy["min_distinct_values"]:
            reasons.append("insufficient_variation")
        rows.append({"feature": column, "observed": count, "first_observed_position": first,
            "post_launch_coverage": coverage, "recent_missing_ratio": recent_missing,
            "distinct_observed": distinct, "eligible": not reasons, "reasons": ";".join(reasons)})
    return pd.DataFrame(rows)
