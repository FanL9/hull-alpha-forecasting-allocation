"""Hash-bind protocol rules without reading market values or running research."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from src.data.download import sha256
from src.utils.paths import inside_root, utc_timestamp, write_json
from src.validation.walk_forward import load_protocol, make_folds

LOCK_PATH = "configs/protocol_lock.json"
PROTOCOL_VERSION = "phase2.5-v2"
ARCHIVED_LOCKS = {"phase2.5-v1": "configs/protocol_lock_phase2.5-v1.json"}
NORMATIVE_FILES = (
    "configs/validation.yaml", "configs/research.yaml", "configs/base.yaml", "configs/official_metric.yaml",
    "configs/portfolio.yaml", "configs/split_policy.json", "docs/research_protocol.md", "docs/official_metric_notes.md",
    "docs/official_api_notes.md", "src/validation/walk_forward.py", "src/validation/protocol_freeze.py",
    "src/evaluation/kaggle_metric.py", "src/features/availability.py", "src/features/feature_groups.py",
    "src/statistics/multiple_testing.py", "src/statistics/robust_inference.py",
    "src/features/family_composites.py", "configs/protocol_lock_phase2.5-v1.json",
    "tests/fixtures/kaggle_metric_synthetic_cases.json",
)


def verify_protocol_lock(root: Path) -> dict:
    """Fail closed on any change to this frozen version's normative files."""
    saved = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    if saved.get("protocol_version") != PROTOCOL_VERSION or set(saved["sha256"]) != set(NORMATIVE_FILES):
        raise ValueError("Unknown protocol version or incomplete lock inventory.")
    for relative, expected in saved["sha256"].items():
        path = inside_root(root, relative)
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Frozen protocol changed: {relative}; explicitly version changes before research.")
    return saved


def freeze_protocol(root: Path) -> dict:
    """Create metadata only; reruns verify the existing lock, never overwrite it."""
    root = root.resolve()
    protocol = load_protocol(root / "configs/validation.yaml")
    research = yaml.safe_load((root / "configs/research.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    for field in ["initial_train_size", "validation_window", "step_size", "purge_size", "embargo_size", "fold_count",
                  "internal_holdout_start", "internal_holdout_end", "final_competition_start", "final_competition_end"]:
        if base["validation"][field] != getattr(protocol, field):
            raise ValueError(f"Duplicated validation configuration differs: {field}")
    if (research["future_validation"]["fold_count"] != protocol.fold_count
            or research["future_validation"]["internal_holdout_rows"] != protocol.internal_holdout_end - protocol.internal_holdout_start + 1):
        raise ValueError("Research settings differ from the authoritative validation protocol.")
    validation = yaml.safe_load((root / "configs/validation.yaml").read_text(encoding="utf-8"))
    if research["protocol"]["version"] != PROTOCOL_VERSION or validation["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Configured protocol version differs from the current frozen version.")
    if (root / LOCK_PATH).exists():
        return verify_protocol_lock(root)
    folds = make_folds(np.arange(protocol.development_start, protocol.development_end + 1), protocol)
    saved = {"protocol_version": PROTOCOL_VERSION, "created_at_utc": utc_timestamp(),
        "status": "frozen_not_research_executed", "defined_before_real_xy_research": True,
        "supersedes": {"protocol_version": "phase2.5-v1", "archived_lock": ARCHIVED_LOCKS["phase2.5-v1"]},
        "sha256": {p: sha256(root / p) for p in NORMATIVE_FILES},
        "fold_metadata": [fold.metadata() for fold in folds],
        "internal_holdout": {"start": 7860, "end": 8867, "rows": 1008, "locked": True,
                             "historically_untouched": False, "reservation": "prospective"},
        "competition_simulation": {"start": 8868, "end": 9047, "rows": 180, "locked": True},
        "execution": {"real_data_eda_repeated": False, "real_xy_tests": 0, "models_fitted": 0,
                      "real_portfolio_evaluations": 0, "competition_values_used": False},
        "primary_target": research["protocol"]["primary_target"],
        "primary_metric": research["protocol"]["primary_metric"]["name"],
        "secondary_metric": research["protocol"]["secondary_metric"]["name"],
        "hypothesis_families": {"single_features": research["protocol"]["primary_hypothesis_family"]["size"],
                                "family_composites": research["protocol"]["secondary_hypothesis_family"]["size"],
                                "global_sensitivity": research["protocol"]["global_sensitivity"]["combined_hypotheses"]},
        "baselines": research["protocol"]["baselines"]}
    write_json(root / LOCK_PATH, saved)
    return saved
