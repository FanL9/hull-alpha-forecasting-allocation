"""Lock executable research rules and data bindings, independently of prose."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from src.data.download import sha256
from src.utils.paths import inside_root, utc_timestamp, write_json
from src.validation.walk_forward import load_protocol, make_folds

LOCK_PATH = "configs/protocol_lock.json"
LOCK_VERSION = 2
RULE_VERSION = "phase2.5-v1"
NORMATIVE_FILES = (
    "configs/validation.yaml", "configs/research.yaml", "configs/base.yaml", "configs/official_metric.yaml",
    "configs/portfolio.yaml", "configs/split_policy.json",
    "src/validation/walk_forward.py", "src/validation/protocol_freeze.py", "src/validation/time_split.py",
    "src/evaluation/kaggle_metric.py", "src/features/availability.py", "src/features/feature_groups.py",
    "src/statistics/multiple_testing.py", "src/statistics/robust_inference.py",
    "src/data/download.py", "src/data/load_data.py", "src/data/preprocess.py", "src/data/validate_data.py",
    "src/research/eda.py", "src/research/pipeline.py", "src/utils/paths.py", "src/utils/seed.py",
    "tests/fixtures/kaggle_metric_synthetic_cases.json",
)
# A lock-format upgrade must preserve all scientific inputs protected by v1.
# Split-policy scope and lock infrastructure are the intentional format changes.
UPGRADE_RULE_FILES = (
    "configs/validation.yaml", "configs/research.yaml", "configs/base.yaml", "configs/official_metric.yaml",
    "configs/portfolio.yaml", "src/validation/walk_forward.py", "src/evaluation/kaggle_metric.py",
    "src/features/availability.py", "src/features/feature_groups.py", "src/statistics/multiple_testing.py",
    "src/statistics/robust_inference.py", "tests/fixtures/kaggle_metric_synthetic_cases.json",
)


def _text_sha256(path: Path) -> str:
    """Match Git text bytes on all platforms; data files never use this rule."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _data_bindings(root: Path) -> dict[str, str]:
    policy = json.loads((root / "configs/split_policy.json").read_text(encoding="utf-8"))
    bindings = policy["data_sha256"]
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError("Split policy must bind raw-data hashes.")
    if bindings.get("data/raw/train.csv") != policy["source_train_sha256"]:
        raise ValueError("Raw train hash differs between split-policy fields.")
    for relative, expected in bindings.items():
        path = inside_root(root, relative)
        if (not path.is_relative_to((root / "data/raw").resolve()) or not isinstance(expected, str)
                or len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise ValueError("Only explicit SHA-256 raw-data bindings are allowed.")
    return dict(bindings)


def _rule_metadata(root: Path) -> dict:
    protocol = load_protocol(root / "configs/validation.yaml")
    research = yaml.safe_load((root / "configs/research.yaml").read_text(encoding="utf-8"))
    base = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    for field in ["initial_train_size", "validation_window", "step_size", "purge_size", "embargo_size", "fold_count",
                  "internal_holdout_start", "internal_holdout_end", "final_competition_start", "final_competition_end"]:
        if base["validation"][field] != getattr(protocol, field):
            raise ValueError(f"Duplicated validation configuration differs: {field}")
    holdout_rows = protocol.internal_holdout_end - protocol.internal_holdout_start + 1
    if (research["future_validation"]["fold_count"] != protocol.fold_count
            or research["future_validation"]["internal_holdout_rows"] != holdout_rows):
        raise ValueError("Research settings differ from authoritative validation configuration.")
    folds = make_folds(np.arange(protocol.development_start, protocol.development_end + 1), protocol)
    return {"fold_metadata": [fold.metadata() for fold in folds],
        "internal_holdout": {"start": protocol.internal_holdout_start, "end": protocol.internal_holdout_end,
            "rows": holdout_rows, "locked": True, "historically_untouched": False, "reservation": "prospective"},
        "competition_simulation": {"start": protocol.final_competition_start, "end": protocol.final_competition_end,
            "rows": protocol.final_competition_end - protocol.final_competition_start + 1, "locked": True},
        "primary_target": research["protocol"]["primary_target"], "baselines": research["protocol"]["baselines"]}


def verify_protocol_lock(root: Path, *, verify_data: bool = False) -> dict:
    """Check config/code and derived rules; optionally hash actual raw files.

    Docs, notebooks and narrative reports are outside the lock. Metadata-only
    verification needs no licensed data; real research must request data checks.
    """
    root = root.resolve()
    saved = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
    if (saved.get("lock_version") != LOCK_VERSION or saved.get("protocol_version") != RULE_VERSION
            or set(saved["sha256"]) != set(NORMATIVE_FILES)):
        raise ValueError("Unknown lock version or incomplete executable inventory.")
    for relative, expected in saved["sha256"].items():
        path = inside_root(root, relative)
        if not path.is_file() or _text_sha256(path) != expected:
            raise ValueError(f"Frozen protocol changed: {relative}; explicitly version scientific changes.")
    if saved["data_sha256"] != _data_bindings(root):
        raise ValueError("Frozen data bindings differ from split policy.")
    for key, expected in _rule_metadata(root).items():
        if saved[key] != expected:
            raise ValueError(f"Frozen metadata differs from executable rules: {key}")
    if verify_data:
        for relative, expected in saved["data_sha256"].items():
            path = inside_root(root, relative)
            if not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Frozen raw data differs or is missing: {relative}")
    return saved


def freeze_protocol(root: Path, *, upgrade_lock: bool = False) -> dict:
    """Create a v2 lock, or explicitly migrate v1 without changing research rules."""
    root = root.resolve()
    metadata = _rule_metadata(root)
    bindings = _data_bindings(root)
    previous = None
    if (root / LOCK_PATH).exists():
        previous = json.loads((root / LOCK_PATH).read_text(encoding="utf-8"))
        if previous.get("lock_version") == LOCK_VERSION:
            return verify_protocol_lock(root)
        if not upgrade_lock:
            raise ValueError("An explicit lock-format upgrade is required for a v1 lock.")
        if previous.get("protocol_version") != RULE_VERSION:
            raise ValueError("Cannot migrate an unknown research-rule version.")
        for relative in UPGRADE_RULE_FILES:
            if _text_sha256(root / relative) != previous["sha256"][relative]:
                raise ValueError(f"Scientific rules changed during lock migration: {relative}")
        for key, expected in metadata.items():
            if previous[key] != expected:
                raise ValueError(f"Research metadata changed during lock migration: {key}")
    saved = {"lock_version": LOCK_VERSION, "protocol_version": RULE_VERSION,
        "created_at_utc": previous["created_at_utc"] if previous else utc_timestamp(),
        "lock_created_at_utc": utc_timestamp(), "status": "frozen_not_research_executed",
        "text_hash_representation": "UTF-8 text bytes with LF line endings",
        "data_hash_representation": "original file bytes",
        "sha256": {p: _text_sha256(root / p) for p in NORMATIVE_FILES}, "data_sha256": bindings,
        **metadata,
        "execution": {"real_data_eda_repeated": False, "real_xy_tests": 0, "models_fitted": 0,
                      "real_portfolio_evaluations": 0, "competition_values_used": False}}
    write_json(root / LOCK_PATH, saved)
    return saved
