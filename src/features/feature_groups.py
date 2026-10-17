"""Recognize anonymous feature families without inventing feature meanings."""

from __future__ import annotations

import re
from collections.abc import Iterable

FAMILY_LABELS = {
    "M": "Market dynamics / technical",
    "E": "Macro economic",
    "I": "Interest rate",
    "P": "Price / valuation",
    "V": "Volatility",
    "S": "Sentiment",
    "MOM": "Momentum",
    "D": "Dummy / binary",
}
OUTCOME_COLUMNS = {"forward_returns", "risk_free_rate", "market_forward_excess_returns"}
METADATA_COLUMNS = {"date_id", "is_scored"}
_FEATURE_PATTERN = re.compile(r"^(MOM|M|E|I|P|V|S|D)\d+$")


def feature_family(column: str) -> str | None:
    """Match MOM before M; metadata and lagged outcomes are not features."""
    match = _FEATURE_PATTERN.fullmatch(column)
    return match.group(1) if match else None


def group_features(columns: Iterable[str]) -> dict[str, list[str]]:
    """Return actual columns in each family, preserving file column order."""
    groups: dict[str, list[str]] = {family: [] for family in FAMILY_LABELS}
    for column in columns:
        family = feature_family(column)
        if family is not None:
            groups[family].append(column)
    return groups


def column_role(column: str) -> str:
    """Conservatively distinguish outcomes, metadata, features and unknowns."""
    if column in OUTCOME_COLUMNS:
        return "outcome / unavailable as contemporaneous predictor"
    if column in METADATA_COLUMNS:
        return "time index" if column == "date_id" else "evaluation metadata"
    if column.startswith("lagged_"):
        return "lagged field; availability requires official API verification"
    if feature_family(column):
        return "anonymous feature"
    return "unclassified; excluded until documented"
