"""Deterministic synthetic null/power checks of the unchanged full pipeline.

Every case uses 97 singles, seven composites, and the frozen 2000 bootstrap
draws. Multiprocessing runs independent artificial cases; it never loads CSVs.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
from math import pi, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.availability import LAGGED_OUTCOME_COLUMNS
from src.research.alpha_pipeline import load_alpha_design
from src.research.phase3 import run_synthetic_phase3
from src.research.phase3_execution import pipeline_code_identity
from src.utils.paths import utc_timestamp, write_json
from src.validation.protocol_freeze import verify_protocol_lock

NOISE_SEEDS = (11, 29, 47, 71, 101)
POWER_SEEDS = (137, 271, 409)
POWER_STRENGTHS = {"M1": 0., "M2": .03, "M3": .05, "M4": .08,
                   "E1": -.03, "E2": -.05, "E3": -.08}


@dataclass
class CalibrationResult:
    noise: pd.DataFrame
    power: pd.DataFrame
    power_summary: pd.DataFrame
    manifest: dict


@contextmanager
def artificial_data_only(root: Path):
    """Workers also deny access to actual data, independent of pytest fixtures."""
    import builtins
    import io
    import os
    from unittest.mock import patch

    protected = (root / "data").resolve()
    def guarded(original):
        def open_file(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).resolve().is_relative_to(protected):
                raise AssertionError("Calibration cannot access project market data.")
            return original(file, *args, **kwargs)
        return open_file
    with patch("builtins.open", guarded(builtins.open)), patch("io.open", guarded(io.open)):
        yield


def make_calibration_history(root: Path, seed: int, *, power: bool = False) -> pd.DataFrame:
    """IID null, or Gaussian-copula signals with specified population Rank IC.

    Lagged target is reconstructed by the pipeline. For an IID target it has
    zero population predictive IC; no lag values are injected or substituted.
    D members remain independent Bernoulli variables under the null.
    """
    design = load_alpha_design(root)
    n = design.validation.research_end + 1
    rng = np.random.default_rng(seed)
    y = rng.normal(size=n)
    values = {"date_id": np.arange(n), "market_forward_excess_returns": y,
              "forward_returns": rng.normal(size=n), "risk_free_rate": rng.normal(size=n)}
    for name in design.single_features:
        if name not in LAGGED_OUTCOME_COLUMNS:
            values[name] = rng.integers(0, 2, n).astype(float) if name.startswith("D") else rng.normal(size=n)
    if power:
        # Common random numbers make strength comparisons less noisy without
        # selecting a seed or requiring significance from any weak signal.
        noise = rng.normal(size=n)
        for name, rank_ic in POWER_STRENGTHS.items():
            rho = 2 * sin(pi * rank_ic / 6)
            values[name] = rho * y + sqrt(1 - rho ** 2) * noise
    frame = pd.DataFrame(values, columns=design.source_columns)
    frame.attrs["phase3_data_kind"] = "synthetic"
    return frame


def _calibration_case(root_string: str, kind: str, seed: int) -> dict:
    root = Path(root_string)
    with artificial_data_only(root):
        frame = make_calibration_history(root, seed, power=kind == "power")
        result = run_synthetic_phase3(root, frame)
    table = result.tables.single_features
    common = {"seed": seed, "single_hypothesis_count": len(table),
              "family_hypothesis_count": len(result.tables.family_composites),
              "global_hypothesis_count": result.manifest["global_hypothesis_count"],
              "testable_count": int(table["status"].eq("testable").sum()),
              "bootstrap_valid_repetitions_min": int(table["bootstrap_valid_repetitions"].min())}
    if kind == "noise":
        rows = [{**common, "mean_signed_ic": float(table["oos_rank_ic"].mean()),
                 "mean_abs_ic": float(table["oos_rank_ic"].abs().mean()),
                 "bh_pass_count": int(table["bh_pass"].sum()), "by_pass_count": int(table["by_pass"].sum())}]
    else:
        indexed = table.set_index("feature")
        rows = [{**common, "hypothesis": name, "injected_strength": strength,
                 "estimated_rank_ic": float(indexed.loc[name, "oos_rank_ic"]),
                 "nw_p_value": float(indexed.loc[name, "raw_p_value"]),
                 "bh_pass": bool(indexed.loc[name, "bh_pass"]),
                 "by_pass": bool(indexed.loc[name, "by_pass"])} for name, strength in POWER_STRENGTHS.items()]
    return {"kind": kind, "seed": seed, "rows": rows,
            "phase3_pipeline_code_hash": result.manifest["phase3_pipeline_code_hash"]}


def run_calibrations(root: Path, output_dir: Path, *, workers: int = 1) -> CalibrationResult:
    """Run a fixed panel once and save only artificial aggregate summaries."""
    root = root.resolve()
    if type(workers) is not int or not 1 <= workers <= 4:
        raise ValueError("Use 1–4 workers for the fixed synthetic calibration panel.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Preserve prior calibration evidence; use an empty output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    lock = verify_protocol_lock(root, verify_data=False)
    jobs = [(str(root), kind, seed) for kind, seeds in [("noise", NOISE_SEEDS), ("power", POWER_SEEDS)] for seed in seeds]
    completed = []
    def record(case):
        completed.append(case)
        write_json(output_dir / "progress.json", {"completed": [{"kind": c["kind"], "seed": c["seed"]} for c in completed],
                                                  "total_cases": len(jobs)})
    if workers == 1:
        for job in jobs:
            record(_calibration_case(*job))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for future in as_completed([executor.submit(_calibration_case, *job) for job in jobs]):
                record(future.result())
    code = pipeline_code_identity()
    if any(c["phase3_pipeline_code_hash"] != code["phase3_pipeline_code_hash"] for c in completed):
        raise RuntimeError("Calibration implementation changed while cases were running.")
    noise = pd.DataFrame([r for c in completed if c["kind"] == "noise" for r in c["rows"]]).sort_values("seed").reset_index(drop=True)
    power = pd.DataFrame([r for c in completed if c["kind"] == "power" for r in c["rows"]]).sort_values(["injected_strength", "seed"]).reset_index(drop=True)
    summary = power.groupby("injected_strength", sort=True).agg(
        estimated_rank_ic=("estimated_rank_ic", "mean"), nw_p_value=("nw_p_value", "median"),
        bh_pass_count=("bh_pass", "sum"), by_pass_count=("by_pass", "sum"), seeds=("seed", "count")).reset_index()
    for name, table in [("pure_noise_calibration.csv", noise), ("weak_signal_calibration.csv", power),
                        ("weak_signal_summary.csv", summary)]:
        table.to_csv(output_dir / name, index=False, lineterminator="\n")
    manifest = {"run_type": "synthetic", "purpose": "software calibration only; no real Alpha evidence",
                "run_timestamp": utc_timestamp(), "protocol_version": lock["protocol_version"], **code,
                "noise_seeds": list(NOISE_SEEDS), "power_seeds": list(POWER_SEEDS),
                "strengths": list(POWER_STRENGTHS.values()), "fold_count": 11,
                "single_hypothesis_count": 97, "family_hypothesis_count": 7, "global_hypothesis_count": 104,
                "bootstrap_repetitions": 2000, "bootstrap_role": "confidence_interval",
                "market_data_read": False, "csv_line_ending": "LF",
                "output_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output_dir.glob("*.csv"))}}
    write_json(output_dir / "calibration_manifest.json", manifest)
    return CalibrationResult(noise, power, summary, manifest)
