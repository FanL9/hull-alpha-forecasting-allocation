"""Load real CSV files in original order; never synthesize missing data."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


class DataUnavailableError(RuntimeError):
    """Raised when required real inputs cannot be read."""


def load_csv(path: Path) -> pd.DataFrame:
    """Load a nonempty CSV, rejecting duplicate column names before pandas."""
    if not path.is_file():
        raise DataUnavailableError(f"Required official file is missing: {path.name}")
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            columns = next(csv.reader(handle), [])
        if not columns:
            raise DataUnavailableError(f"CSV has no header: {path.name}")
        if len(set(columns)) != len(columns):
            raise DataUnavailableError(f"Duplicate CSV headers: {path.name}")
        frame = pd.read_csv(path, low_memory=False)
    except (OSError, ValueError, pd.errors.ParserError) as exc:
        raise DataUnavailableError(f"Cannot read {path.name}: {type(exc).__name__}") from exc
    if frame.empty:
        raise DataUnavailableError(f"CSV has no data rows: {path.name}")
    return frame


def load_official_pair(raw_dir: Path, train_file: str, test_file: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read both files without sorting, shuffling, filling or transforming."""
    return load_csv(raw_dir / train_file), load_csv(raw_dir / test_file)
