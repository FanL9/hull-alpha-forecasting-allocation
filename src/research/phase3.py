"""Explicit real-data gate and a separate, marked synthetic Phase 3 entry."""
from __future__ import annotations

import json
from pathlib import Path

import yaml

from src.research.alpha_pipeline import build_alpha_tables, load_alpha_design, validate_research_history
from src.research.alpha_results import Phase3Result, apply_multiple_testing, build_manifest, write_phase3_outputs
from src.utils.paths import inside_root
from src.validation.protocol_freeze import verify_protocol_lock
from src.validation.walk_forward import load_research_prefix
from src.validation.time_split import read_header


def _output_destination(root, output_dir):
    if output_dir is None:
        return None
    destination = inside_root(root, output_dir)
    if not destination.is_relative_to((root / "artifacts").resolve()):
        raise ValueError("Phase 3 run outputs must stay under artifacts/.")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("Phase 3 output directory must be empty; preserve prior run evidence.")
    return destination


def _execute(root, history, lock, *, synthetic, output_dir):
    destination = _output_destination(root, output_dir)
    design = load_alpha_design(root)
    validate_research_history(history, design)
    tables = build_alpha_tables(history, design)
    result = Phase3Result(tables, apply_multiple_testing(tables, design),
                          build_manifest(root, history, design, lock, synthetic=synthetic))
    if destination is not None:
        write_phase3_outputs(result, destination)
    return result


def run_synthetic_phase3(root: Path, history, *, output_dir: str | None = None) -> Phase3Result:
    """Accept only explicitly marked artificial inputs; never load market data."""
    if history.attrs.get("phase3_data_kind") != "synthetic":
        raise ValueError("Synthetic mode requires an explicitly marked synthetic frame.")
    root = root.resolve()
    lock = verify_protocol_lock(root, verify_data=False)
    return _execute(root, history, lock, synthetic=True, output_dir=output_dir)


def run_real_phase3(root: Path, *, run_real_phase3: bool = False,
                    output_dir: str = "artifacts/phase3") -> Phase3Result:
    """Future authorized entry: verify data before the only safe prefix loader.

    Never invoked during synthetic development. Hash verification can stream
    original file bytes; only load_research_prefix parses research observations.
    """
    if run_real_phase3 is not True:
        raise PermissionError("Real research requires explicit --run-real-phase3 authorization.")
    root = root.resolve()
    lock = verify_protocol_lock(root, verify_data=True)
    _output_destination(root, output_dir)
    design = load_alpha_design(root)
    policy = json.loads((root / "configs/split_policy.json").read_text(encoding="utf-8"))
    base = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
    train = inside_root(root, f"{base['data']['raw_dir']}/{base['data']['train_file']}")
    # Header-only access is allowed after verification; the existing safe loader
    # compares it and parses at most 7860 rows. Never materialize full train.csv.
    policy = {**policy, "train_columns": read_header(train)}
    history = load_research_prefix(train, policy, design.validation)
    return _execute(root, history, lock, synthetic=False, output_dir=output_dir)
