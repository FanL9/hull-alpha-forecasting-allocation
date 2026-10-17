"""Predictor eligibility only; train-window imputation belongs to Phase 3."""

from collections.abc import Iterable

from src.features.feature_groups import feature_family


def eligible_features(columns: Iterable[str], excluded: Iterable[str] = ()) -> list[str]:
    """Allow only recognized anonymous features, excluding configured fields.

    This is a name-based safety filter, not a claim that feature availability
    has been verified. No imputer, scaler or selector is fitted in Phase 1.
    """
    denied = set(excluded)
    return [column for column in columns if feature_family(column) and column not in denied]
