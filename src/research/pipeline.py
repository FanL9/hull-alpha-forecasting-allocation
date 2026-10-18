"""Hash verification → fixed reservation → development-only processing and EDA."""
from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.download import sha256
from src.data.preprocess import retain_missing
from src.data.validate_data import run_locked_integrity
from src.features.feature_groups import group_features
from src.research.eda import feature_eda, target_eda
from src.research.figures import make_figures
from src.research.reporting import records, render_report, strict_value
from src.utils.paths import inside_root, utc_timestamp, write_json
from src.validation.time_split import SPLIT_PATH, load_development


def load_eda_settings(path: Path) -> dict:
    settings = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (settings["phase"] != "data_processing_eda" or settings["analysis_start"] != 0
            or settings["analysis_end"] != 8867 or settings["default_processing"] != "retain_nan"):
        raise ValueError("Only development-only descriptive EDA with retain_nan is allowed.")
    future = settings["future_validation"]
    if future != {"method": "expanding_window", "status": "protocol_frozen_not_executed",
                  "config_path": "configs/validation.yaml", "fold_count": 11,
                  "internal_holdout_rows": 1008, "selected_model": None}:
        raise ValueError("Validation parameters are frozen; the model remains undecided.")
    for key in ["time_block_rows", "rolling_window_rows", "rolling_min_observations", "rolling_output_stride", "correlation_min_pairs", "target_autocorrelation_max_lag"]:
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    if settings["rolling_min_observations"] > settings["rolling_window_rows"]:
        raise ValueError("Rolling minimum cannot exceed the trailing window.")
    if not 0 <= settings["high_xx_correlation_threshold"] <= 1:
        raise ValueError("X-X correlation threshold must be in [0,1].")
    return settings


def run_phase2(root: Path, config: dict, settings_path: Path) -> dict:
    """No downloader, alpha, model, regime, signal or portfolio call exists here."""
    root = root.resolve()
    settings = load_eda_settings(settings_path)
    integrity = run_locked_integrity(root, config)
    plan = json.loads((root / SPLIT_PATH).read_text(encoding="utf-8"))
    train = inside_root(root, config["data"]["raw_dir"]) / config["data"]["train_file"]
    development = load_development(train, plan)
    features, dates = plan["feature_columns"], development["date_id"]
    X = development[features]
    tables = feature_eda(X, dates, settings)
    target_result = target_eda(development["market_forward_excess_returns"], dates, settings)
    tables.update(target_result["tables"])
    artifacts = root / "artifacts"
    for name, table in tables.items():
        table.to_csv(artifacts / f"{name}.csv", index=name in {"xx_correlation", "xx_pair_counts"})
    processed = retain_missing(development)
    processed_path = root / "data/processed/development.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_csv(processed_path, index=False)
    quality = {"rows": len(development), "columns": len(development.columns), "feature_columns": len(features),
        "outcome_columns": plan["outcome_columns"], "date_id_min": int(dates.min()), "date_id_max": int(dates.max()),
        "duplicate_rows": int(development.duplicated().sum()), "duplicate_date_ids": int(dates.duplicated().sum()),
        "chronological_order": bool(dates.is_monotonic_increasing), "unit_steps": bool(dates.diff().dropna().eq(1).all()),
        "infinite_cells": int(np.isinf(development.to_numpy(dtype=float)).sum()),
        "dtypes": {c: str(development[c].dtype) for c in development},
        "outcome_missing_counts": {c: int(development[c].isna().sum()) for c in plan["outcome_columns"]}}
    summary = {"schema_version": 2, "status": "complete", "phase": "data_processing_eda",
        "generated_at_utc": utc_timestamp(), "settings": settings,
        "analysis_scope": {"area": "development_only", "start_date_id": 0, "end_date_id": 8867, "rows": 8868,
            "simulation_values_parsed": False, "official_test_values_parsed": False, "feature_target_analysis_performed": False},
        "split": {k: v for k, v in plan.items() if k not in {"train_columns", "feature_columns", "outcome_columns"}},
        "development_quality": quality,
        "feature_families": {f: len(names) for f, names in group_features(features).items() if names},
        "absent_mom_family": True,
        "feature_missingness": {"features_with_missing": int(X.isna().any().sum()),
            "missing_cells": int(X.isna().sum().sum()), "missing_fraction": float(X.isna().to_numpy().mean()),
            "all_missing_features": [c for c in X if X[c].isna().all()],
            "constant_features": [c for c in X if X[c].nunique(dropna=True) == 1]},
        "family_statistics": records(tables["family_statistics"]),
        "xx_structure": {"high_correlation_pairs": len(tables["high_xx_pairs"]), "feature_target_columns_in_matrix": False},
        "target_only": strict_value(target_result["summary"]),
        "processing": {"default": "retain_nan", "features_preserved": len(features), "outcomes_preserved": len(plan["outcome_columns"]),
            "values_changed": False, "rows_dropped": 0, "columns_dropped": 0, "optional_candidates_applied": False,
            "processed_path": processed_path.relative_to(root).as_posix(), "processed_sha256": sha256(processed_path)},
        "provenance": {"source": integrity["provenance"]["source"], "official_file_count": integrity["provenance"]["official_file_count"],
            "source_train_sha256": plan["source_train_sha256"], "source_test_sha256": plan["source_test_sha256"],
            "settings_sha256": sha256(settings_path), "phase1_outputs_preserved": True, "data_downloaded": False},
        "execution": {"alpha_research": False, "models_fitted": 0, "folds_executed": 0, "oos_predictions": 0,
            "regime_research": False, "signals": False, "allocation": False, "backtest": False},
        "dependencies": {p: importlib.metadata.version(p) for p in ["pandas", "numpy", "matplotlib", "PyYAML"]},
        "table_rows": {name: len(table) for name, table in tables.items()},
        "limitations": ["Anonymous fields: no economic field-level interpretation.",
            "Past Phase 1 and withdrawn research inspection cannot be undone; lock is a prospective-use policy.",
            "No calendar mapping; windows refer to ordered rows, not verified calendar days.",
            "Feature release timing, label construction and official scoring require future verification."]}
    figure_names = make_figures(tables, root / "reports/figures")
    summary["figures"] = figure_names
    summary["source_code_sha256"] = {p.relative_to(root).as_posix(): sha256(p) for p in sorted((root / "src").rglob("*.py"))}
    write_json(artifacts / "phase2_data_summary.json", strict_value(summary))
    report = root / "reports/phase2_data_processing_eda.md"
    report.write_text(render_report(summary, tables, figure_names), encoding="utf-8")
    return strict_value(summary)
