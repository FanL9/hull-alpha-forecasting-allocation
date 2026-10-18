"""Pre-registered M/E/I/P/V/S/D composites; pure logic, tested only synthetically.

All fitted quantities (members, directions, percentile maps) come from one
training window. Validation rows are only mapped, never used for fitting.
Lagged outcome fields belong to no family and never enter a composite.
"""
from __future__ import annotations

from math import ceil

import numpy as np
import pandas as pd

from src.features.feature_groups import OUTCOME_COLUMNS, feature_family

COMPOSITE_FAMILIES = ("M", "E", "I", "P", "V", "S", "D")


def training_direction(training_feature: pd.Series, training_target: pd.Series) -> int:
    """Sign of the training-window Spearman correlation; 0 means exclude."""
    pairs = pd.DataFrame({"x": training_feature, "y": training_target}).dropna()
    if len(pairs) < 3 or pairs["x"].nunique() < 2 or pairs["y"].nunique() < 2:
        return 0
    rho = pairs["x"].rank().corr(pairs["y"].rank())
    return 0 if not np.isfinite(rho) or rho == 0 else int(np.sign(rho))


def training_percentile(training_feature: pd.Series, values: pd.Series) -> pd.Series:
    """Mid-rank empirical CDF of the training values, applied to any values.

    p(v) = (#train < v + 0.5 × #train == v) / n_train. On the training values
    this equals (average rank − 0.5) / n. NaNs remain NaN.
    """
    reference = np.sort(training_feature.dropna().to_numpy(dtype=float))
    if not len(reference):
        raise ValueError("A percentile map needs observed training values.")
    v = values.to_numpy(dtype=float)
    below = np.searchsorted(reference, v, side="left")
    at_or_below = np.searchsorted(reference, v, side="right")
    result = (below + .5 * (at_or_below - below)) / len(reference)
    return pd.Series(np.where(np.isnan(v), np.nan, result), index=values.index)


def family_composite(family: str, training_x: pd.DataFrame, training_target: pd.Series,
                     apply_x: pd.DataFrame, availability: pd.DataFrame,
                     min_members: int = 2, min_fraction: float = .5) -> tuple[pd.Series, dict]:
    """Equal-weight mean of oriented training-percentile members for one fold.

    Qualified members pass the fold's training availability rule and have a
    nonzero training Spearman sign. A row needs at least
    max(min_members, ceil(min_fraction × qualified)) observed members.
    """
    if family not in COMPOSITE_FAMILIES:
        raise ValueError("Unknown composite family.")
    if OUTCOME_COLUMNS & (set(training_x) | set(apply_x)) or not training_x.index.equals(training_target.index):
        raise ValueError("Use X-only frames and a target aligned to training rows only.")
    eligible = set(availability.loc[availability["eligible"], "feature"])
    candidates = [c for c in training_x if feature_family(c) == family and c in eligible]
    directions = {c: training_direction(training_x[c], training_target) for c in candidates}
    members = [c for c in candidates if directions[c] != 0]
    required = max(min_members, ceil(min_fraction * len(members)))
    metadata = {"family": family, "qualified_members": members, "required_per_row": required,
                "excluded_unavailable": [c for c in training_x if feature_family(c) == family and c not in eligible],
                "excluded_zero_direction": [c for c in candidates if directions[c] == 0],
                "directions": {c: directions[c] for c in members}}
    if len(members) < min_members:
        return pd.Series(np.nan, index=apply_x.index, name=family), metadata
    oriented = pd.DataFrame({c: directions[c] * training_percentile(training_x[c], apply_x[c]) for c in members})
    counts = oriented.notna().sum(axis=1)
    return oriented.mean(axis=1).where(counts >= required).rename(family), metadata


def family_composites(training_x: pd.DataFrame, training_target: pd.Series, apply_x: pd.DataFrame,
                      availability: pd.DataFrame, min_members: int = 2,
                      min_fraction: float = .5) -> tuple[pd.DataFrame, dict]:
    """All seven pre-registered composites for one fold, in fixed family order."""
    series, metadata = {}, {}
    for family in COMPOSITE_FAMILIES:
        series[family], metadata[family] = family_composite(
            family, training_x, training_target, apply_x, availability, min_members, min_fraction)
    return pd.DataFrame(series, index=apply_x.index), metadata
