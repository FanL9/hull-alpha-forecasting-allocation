"""Fixed identifier-only reservation; never materialize simulation values."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.download import sha256
from src.features.feature_groups import OUTCOME_COLUMNS, feature_family
from src.utils.paths import utc_timestamp, write_json

DEVELOPMENT_ROWS = 8868
FULL_ROWS = 9048
SIMULATION_ROWS = 180
SPLIT_PATH = "artifacts/split_definition.json"


def read_header(path: Path) -> list[str]:
    """Read the header only, without inferring dtypes from any data row."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        columns = next(csv.reader(handle), [])
    if not columns or len(columns) != len(set(columns)):
        raise ValueError("CSV header is empty or contains duplicate fields.")
    return columns


def inspect_identifiers(path: Path) -> tuple[list[str], pd.Series]:
    columns = read_header(path)
    if "date_id" not in columns:
        raise ValueError("date_id is required.")
    dates = pd.read_csv(path, usecols=["date_id"])["date_id"]
    if (dates.isna().any() or not pd.api.types.is_integer_dtype(dates)
            or dates.duplicated().any() or not dates.is_monotonic_increasing):
        raise ValueError("Identifiers must be ordered, unique, nonmissing integers.")
    return columns, dates


def prepare_split(root: Path, train_path: Path, test_path: Path) -> dict:
    """Validate IDs/headers and persist the immutable, hash-bound reservation."""
    columns, dates = inspect_identifiers(train_path)
    if not np.array_equal(dates.to_numpy(), np.arange(FULL_ROWS)):
        raise ValueError("Expected exactly date_id 0..9047 in original file order.")
    features = [c for c in columns if feature_family(c)]
    if len(features) != 94 or set(columns) != set(features) | OUTCOME_COLUMNS | {"date_id"}:
        raise ValueError("Expected 94 features, three outcomes and date_id.")
    test_columns, test_dates = inspect_identifiers(test_path)
    if {c for c in test_columns if feature_family(c)} != set(features):
        raise ValueError("Official inference feature schema is incompatible.")
    plan = {
        "schema_version": 2,
        "full_train_start": 0, "full_train_end": 9047, "full_train_rows": FULL_ROWS,
        "development_start": 0, "development_end": 8867, "development_rows": DEVELOPMENT_ROWS,
        "competition_simulation_start": 8868, "competition_simulation_end": 9047,
        "competition_simulation_rows": SIMULATION_ROWS, "competition_simulation_locked": True,
        "split_rule": "Fixed last 180 ordered labelled rows; no gap, shuffle or data-driven boundary.",
        "created_at_utc": utc_timestamp(), "source_train_sha256": sha256(train_path),
        "source_test_sha256": sha256(test_path),
        "target_leakage_policy": "Development-only target EDA, separate from X; no X-Y analysis or fitting.",
        "competition_simulation_data_policy": "IDs, row count and header compatibility only; feature/target values are never parsed for analysis.",
        "simulation_dtype_validation": "Header-compatible with the same train CSV; value-derived dtypes not inspected.",
        "train_columns": columns, "feature_columns": features,
        "outcome_columns": [c for c in columns if c in OUTCOME_COLUMNS],
        "official_test": {
            "role": "inference_schema_only", "used_as_model_test_set": False,
            "rows": len(test_dates), "columns": test_columns,
            "date_id_min": int(test_dates.min()), "date_id_max": int(test_dates.max()),
            "overlap_with_training_ids": int(test_dates.isin(dates).sum()),
            "train_only_columns": sorted(set(columns) - set(test_columns)),
            "test_only_columns": sorted(set(test_columns) - set(columns)),
            "feature_schema_compatible": True, "value_statistics_computed": False,
        },
        "future_validation": {"method": "expanding_window", "status": "design_only",
                              "fold_count": None, "internal_holdout_rows": None, "selected_model": None},
    }
    policy_path = root / "configs/split_policy.json"
    if policy_path.exists():
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        for key in ["development_start", "development_end", "development_rows",
                    "competition_simulation_start", "competition_simulation_end",
                    "competition_simulation_rows", "competition_simulation_locked", "source_train_sha256"]:
            if policy.get(key) != plan[key]:
                raise ValueError(f"Tracked split policy differs: {key}")
    destination = root / SPLIT_PATH
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        plan["created_at_utc"] = saved.get("created_at_utc")
        if saved != plan:
            raise ValueError("Locked split/source/schema changed; refusing to overwrite reservation.")
        return saved
    write_json(destination, plan)
    return plan


def validate_development(frame: pd.DataFrame, plan: dict) -> None:
    if len(frame) != DEVELOPMENT_ROWS or list(frame.columns) != plan["train_columns"]:
        raise ValueError("Development shape/schema differs from the locked plan.")
    if not np.array_equal(frame["date_id"].to_numpy(), np.arange(DEVELOPMENT_ROWS)):
        raise ValueError("Development must contain only ordered date_id 0..8867.")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in frame):
        raise ValueError("Development columns must be numeric; values are not coerced.")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Development contains infinity; investigate without replacing values.")


def load_development(train_path: Path, plan: dict) -> pd.DataFrame:
    """Only the 8,868-row prefix enters a value-bearing dataframe."""
    if (plan["development_end"] != 8867 or plan["development_rows"] != DEVELOPMENT_ROWS
            or plan["competition_simulation_locked"] is not True
            or plan["source_train_sha256"] != sha256(train_path)):
        raise ValueError("Invalid or stale development reservation.")
    frame = pd.read_csv(train_path, nrows=DEVELOPMENT_ROWS)
    validate_development(frame, plan)
    return frame
