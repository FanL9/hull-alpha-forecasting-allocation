"""Conservative name filter and optional past-only missing-value candidates."""
from collections.abc import Iterable

import pandas as pd

from src.features.feature_groups import feature_family


def eligible_features(columns: Iterable[str], excluded: Iterable[str] = ()) -> list[str]:
    """A name-based safety filter, not predictive feature selection."""
    denied = set(excluded)
    return [column for column in columns if feature_family(column) and column not in denied]


def retain_missing(frame: pd.DataFrame) -> pd.DataFrame:
    """Default: preserve every row, column, outcome, encoding and NaN."""
    return frame.copy(deep=True)


def _features_only(frame: pd.DataFrame) -> None:
    if not len(frame.columns) or any(feature_family(c) is None for c in frame):
        raise ValueError("Candidate preprocessing accepts anonymous X columns only.")
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Candidate preprocessing requires ordered unique row indices.")


def expanding_prior_median(frame: pd.DataFrame) -> pd.DataFrame:
    """Optional candidate, not used by default; unavailable history stays NaN.

    At row t the fallback uses observations strictly before t. No fitted
    full-sample value, backward fill, target column or initial fallback exists.
    """
    _features_only(frame)
    return frame.fillna(frame.expanding(min_periods=1).median().shift(1))


def missing_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Optional contemporaneous X-only flags; not appended to default output."""
    _features_only(frame)
    return frame.isna().astype("int8").add_suffix("_missing")
