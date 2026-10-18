"""Family-specific decisions and reproducible Phase 3 output artifacts."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from src.features.family_composites import COMPOSITE_FAMILIES
from src.research.alpha_pipeline import AlphaDesign, AlphaTables
from src.statistics.multiple_testing import protocol_multiplicity
from src.utils.paths import utc_timestamp, write_json


@dataclass
class Phase3Result:
    tables: AlphaTables
    multiple_testing_summary: pd.DataFrame
    manifest: dict


def apply_multiple_testing(tables: AlphaTables, design: AlphaDesign) -> pd.DataFrame:
    """BH alone sets each family's formal decision; BY/global stay sensitivity."""
    single, composite = tables.single_features, tables.family_composites
    assert tuple(single["feature"]) == design.single_features
    assert tuple(composite["feature"]) == COMPOSITE_FAMILIES
    corrections = protocol_multiplicity(
        dict(zip(single["feature"], single["raw_p_value"])), design.single_features,
        dict(zip(composite["feature"], composite["raw_p_value"])), COMPOSITE_FAMILIES,
        alpha=design.rules["primary_hypothesis_family"]["fdr_alpha"])
    count = len(single)
    for table, prefix, subset in [(single, "single_feature", slice(0, count)),
                                   (composite, "family_composite", slice(count, None))]:
        bh, by = corrections[prefix + "_bh"], corrections[prefix + "_by_sensitivity"]
        table["bh_q_value"], table["bh_pass"] = bh.adjusted_pvalues, bh.rejected
        table["by_q_value"], table["by_pass"] = by.adjusted_pvalues, by.rejected
        for method in ["bh", "by"]:
            global_result = corrections[f"global_{method}_sensitivity"]
            table[f"global_{method}_q_value"] = global_result.adjusted_pvalues[subset]
            table[f"global_{method}_pass_sensitivity"] = global_result.rejected[subset]
        table["formal_conclusion"] = np.where(table["status"].eq("not_testable"), "NOT_TESTABLE",
                                              np.where(table["bh_pass"], "PASS", "FAIL"))
        table["descriptive_note"] = np.where(table["bh_pass"] & ~table["by_pass"], "dependence-sensitive", "")
    return pd.DataFrame([{"analysis": name, "method": result.method, "alpha": result.alpha,
                          "hypotheses": len(result.adjusted_pvalues), "rejected": int(result.rejected.sum()),
                          "role": "sensitivity" if "sensitivity" in name else "family_primary"}
                         for name, result in corrections.items()])


def build_manifest(root: Path, history: pd.DataFrame, design: AlphaDesign, lock: dict, *, synthetic: bool) -> dict:
    """Bind results to rules, executing implementation, dataset identity and Git."""
    def git(*args):
        process = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return process.stdout.strip() if process.returncode == 0 else None
    implementation = Path(__file__).resolve().parent
    code = {f"src/research/{name}": hashlib.sha256((implementation / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
            for name in ["alpha_pipeline.py", "alpha_results.py", "phase3.py", "synthetic_alpha.py"]}
    code["src/cli.py"] = hashlib.sha256((implementation.parent / "cli.py").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    fingerprint = hashlib.sha256(json.dumps(list(history.columns)).encode() +
                                 pd.util.hash_pandas_object(history, index=False).to_numpy().tobytes()).hexdigest()
    inference = design.rules["inference"]
    return {"protocol_version": design.rules["version"], "git_commit": git("rev-parse", "HEAD"),
            "git_worktree_dirty": bool(git("status", "--porcelain")), "implementation_sha256": code,
            "protocol_lock_hash": hashlib.sha256((root / "configs/protocol_lock.json").read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "verified_dataset_identity": {"kind": "synthetic" if synthetic else "official_hull",
                "raw_data_hashes_verified": not synthetic,
                "official_data_sha256": {} if synthetic else lock["data_sha256"],
                "research_frame_sha256": fingerprint},
            "run_timestamp": utc_timestamp(), "run_mode": "synthetic" if synthetic else "real",
            "real_alpha_research_executed": not synthetic,
            "internal_holdout_used": False, "final_competition_used": False,
            "research_date_id_start": design.validation.research_start,
            "research_date_id_end": design.validation.research_end, "research_rows": len(history),
            "fold_count": design.validation.fold_count, "folds": lock["fold_metadata"],
            "target": design.rules["primary_target"], "primary_metric": design.rules["primary_metric"],
            "secondary_metric": design.rules["secondary_metric"], "hac_lag": inference["newey_west_lags"],
            "bootstrap_config": {k: inference[k] for k in ["bootstrap_role", "bootstrap_input", "bootstrap_method",
                "bootstrap_block_length", "bootstrap_repetitions", "confidence_level", "random_seed"]},
            "single_hypothesis_count": len(design.single_features),
            "family_hypothesis_count": len(COMPOSITE_FAMILIES),
            "global_hypothesis_count": len(design.single_features) + len(COMPOSITE_FAMILIES),
            "formal_decision_rule": "family-specific BH-FDR; BY/global/bootstrap/fold diagnostics do not change decisions",
            "single_hypotheses": list(design.single_features), "family_hypotheses": list(COMPOSITE_FAMILIES)}


def write_phase3_outputs(result: Phase3Result, directory: Path) -> None:
    """Publish a completed run without overwriting a previous run's evidence."""
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Phase 3 output directory must be empty; preserve prior run evidence.")
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {"single_feature_results.csv": result.tables.single_features,
               "family_composite_results.csv": result.tables.family_composites,
               "fold_level_ic.csv": result.tables.fold_level_ic,
               "feature_availability.csv": result.tables.feature_availability,
               "multiple_testing_summary.csv": result.multiple_testing_summary}
    for name, table in outputs.items():
        table.to_csv(directory / name, index=False)
    write_json(directory / "fold_transform_audit.json", result.tables.fold_transforms)
    result.manifest["output_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(directory.iterdir()) if p.is_file()}
    write_json(directory / "phase3_run_manifest.json", result.manifest)
