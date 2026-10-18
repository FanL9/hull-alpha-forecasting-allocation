"""Separate post-hoc diagnostics of sealed Phase 3 PASS signals; never a rerun.

The only real-data reader is the existing bounded research-prefix loader.
Official decisions, masks, winsor bounds and composite recipes are read-only.
No NW, bootstrap, FDR, prediction model or portfolio function is called.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import yaml

from src.features.availability import available_lagged_outcomes
from src.features.family_composites import training_percentile
from src.research.alpha_pipeline import load_alpha_design, validate_research_history
from src.research.posthoc_statistics import (DIAGNOSTIC_TYPE, average_linkage, correlation_structure,
    decile_analysis, divergence_summary, lag_decay, rank_ic, reverse_return_ic)
from src.utils.paths import inside_root, project_root, utc_timestamp, write_json
from src.validation.protocol_freeze import verify_protocol_lock
from src.validation.time_split import read_header
from src.validation.walk_forward import load_research_prefix

RESULTS_COMMIT = "53402c59692955c258bbd7c0c7875e9ffd29757a"
EXPECTED_SINGLES = ("M1", "M3", "M9", "M11", "E2", "E3", "E19", "P2", "P3", "P4", "P5", "P8",
    "P10", "P11", "V7", "V9", "V10", "V13", "S2", "S3", "S5", "S8",
    "lagged_forward_returns", "lagged_market_forward_excess_returns")
EXPECTED_COMPOSITES = ("M", "P", "V", "S")


@dataclass
class SealedPhase3:
    manifest: dict
    singles: pd.DataFrame
    composites: pd.DataFrame
    folds: pd.DataFrame
    availability: pd.DataFrame
    transforms: list[dict]
    protected_hashes: dict[str, str]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def protected_hashes(root: Path) -> dict[str, str]:
    paths = [*sorted((root / "reports/phase3_real_run").glob("*")),
             root / "reports/phase3_alpha_research_results.md",
             *sorted((root / "artifacts/phase3").glob("*"))]
    state = root / "artifacts/phase3_execution_state.json"
    if state.exists():
        paths.append(state)
    return {p.relative_to(root).as_posix(): sha256(p) for p in paths if p.is_file()}


def assert_preserved(root: Path, expected: dict[str, str]) -> None:
    if protected_hashes(root) != expected:
        raise ValueError("Sealed Phase 3 results changed; diagnostics must never alter them.")


def load_sealed_phase3(root: Path) -> SealedPhase3:
    directory = root / "reports/phase3_real_run"
    before = protected_hashes(root)
    seal = json.loads((directory / "snapshot_seal.json").read_text(encoding="utf-8"))
    for name, expected in seal["sha256"].items():
        if sha256(directory / name) != expected:
            raise ValueError("Sealed Phase 3 hash mismatch.")
    for relative, expected in before.items():
        if relative.startswith("reports/"):
            committed = git(root, "show", f"{RESULTS_COMMIT}:{relative}")
            if hashlib.sha256(committed).hexdigest() != expected:
                raise ValueError("Formal report differs from the sealed results commit.")
    manifest = json.loads((directory / "phase3_run_manifest.json").read_text(encoding="utf-8"))
    singles = pd.read_csv(directory / "single_feature_results.csv")
    composites = pd.read_csv(directory / "family_composite_results.csv")
    if (tuple(singles.loc[singles["formal_conclusion"].eq("PASS"), "feature"]) != EXPECTED_SINGLES
            or tuple(composites.loc[composites["formal_conclusion"].eq("PASS"), "feature"]) != EXPECTED_COMPOSITES
            or manifest["run_type"] != "real" or manifest["git_tag"] != "phase3-pipeline-v1"):
        raise ValueError("Only the sealed 24/4 PASS objects can enter these diagnostics.")
    return SealedPhase3(manifest, singles, composites, pd.read_csv(directory / "fold_level_ic.csv"),
        pd.read_csv(directory / "feature_availability.csv"),
        json.loads((directory / "fold_transform_audit.json").read_text(encoding="utf-8")), before)


def frozen_oos_grid(sealed: SealedPhase3) -> tuple[pd.Series, pd.DataFrame]:
    metadata = sealed.manifest["folds"]
    dates = np.concatenate([np.arange(f["validation_start"], f["validation_end"] + 1) for f in metadata])
    labels = np.concatenate([np.full(f["validation_rows"], f["fold"]) for f in metadata])
    blocks = pd.Series(labels, index=pd.Index(dates, name="date_id"), name="fold")
    names = EXPECTED_SINGLES + EXPECTED_COMPOSITES
    eligible = pd.DataFrame(False, index=blocks.index, columns=names)
    for name in names:
        rows = sealed.folds.loc[sealed.folds["hypothesis"].eq(name)]
        accepted = rows.loc[rows["eligible"], "fold"].tolist()
        eligible[name] = blocks.isin(accepted)
    return blocks, eligible


def reconstruct_composites(x: pd.DataFrame, sealed: SealedPhase3) -> pd.DataFrame:
    """Replay saved recipes and training maps, never refit signs or members.

    C_t exists only on the original validation grid. Its time shift retains the
    previous fold's actual C value at boundaries; no earlier C is fabricated.
    """
    result = pd.DataFrame(np.nan, index=x.index, columns=EXPECTED_COMPOSITES)
    for info, fold in zip(sealed.transforms, sealed.manifest["folds"], strict=True):
        if info["fold"] != fold["fold"] or (info["train_start"], info["train_end"]) != (fold["train_start"], fold["train_end"]):
            raise ValueError("Frozen composite training metadata differs.")
        train = x.loc[fold["train_start"]:fold["train_end"]]
        validation = x.loc[fold["validation_start"]:fold["validation_end"]]
        for family in EXPECTED_COMPOSITES:
            recipe = info["family_composites"][family]
            members = recipe["qualified_members"]
            if len(members) < 2:
                continue
            oriented = {}
            for name in members:
                reference_hash = hashlib.sha256(np.sort(train[name].dropna().to_numpy(dtype="<f8")).tobytes()).hexdigest()
                if reference_hash != recipe["percentile_reference_sha256"][name]:
                    raise ValueError("Training percentile reference differs from the sealed recipe.")
                oriented[name] = recipe["directions"][name] * training_percentile(train[name], validation[name])
            values = pd.DataFrame(oriented, index=validation.index)
            result.loc[validation.index, family] = values.mean(axis=1).where(values.notna().sum(axis=1) >= recipe["required_per_row"])
    return result


def diagnostic_tables(history: pd.DataFrame, root: Path, sealed: SealedPhase3) -> dict[str, pd.DataFrame]:
    design = load_alpha_design(root)
    validate_research_history(history, design)
    frame = history.set_index("date_id")
    candidates = pd.concat([frame[[c for c in design.single_features if c in frame]],
                            available_lagged_outcomes(history).set_axis(frame.index)], axis=1)
    x = candidates.loc[:, list(EXPECTED_SINGLES)]
    target = frame[design.rules["primary_target"]]
    blocks, masks = frozen_oos_grid(sealed)
    eligible_single = masks.loc[:, list(EXPECTED_SINGLES)]
    eligible_composite = masks.loc[:, list(EXPECTED_COMPOSITES)]
    lag, wide = lag_decay(x, target, blocks, eligible_single)
    reverse = reverse_return_ic(x, target, blocks, eligible_single)
    composite = reconstruct_composites(candidates, sealed)
    composite_lag, composite_wide = lag_decay(composite, target, blocks, eligible_composite)
    reconciliation = []
    for table, original in [(wide, sealed.singles), (composite_wide, sealed.composites)]:
        indexed = original.set_index("feature")
        for row in table.to_dict("records"):
            name = row["feature"]
            saved = indexed.loc[name]
            if not np.isclose(row["IC_k0"], saved["oos_rank_ic"], atol=1e-10, rtol=0) or row["n_obs_k0"] != saved["oos_pairs"]:
                raise ValueError(f"k=0 does not reconcile to sealed Phase 3: {name}")
            reconciliation.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name,
                "sealed_rank_ic": saved["oos_rank_ic"], "diagnostic_rank_ic_k0": row["IC_k0"],
                "difference": row["IC_k0"] - saved["oos_rank_ic"], "n_obs": row["n_obs_k0"]})
    oos_x = x.loc[blocks.index].where(eligible_single)
    clipped = pd.DataFrame(np.nan, index=blocks.index, columns=x.columns)
    for row in sealed.availability.to_dict("records"):
        name = row["feature"]
        if name not in clipped or not row["eligible"]:
            continue
        selected = blocks.eq(row["fold"]) & eligible_single[name]
        clipped.loc[selected, name] = oos_x.loc[selected, name].clip(row["winsor_lower"], row["winsor_upper"])
    divergence = divergence_summary(oos_x, clipped, target.loc[blocks.index], blocks, sealed.singles)
    if not np.allclose(divergence["frozen_winsorized_pearson_ic"], divergence["reconstructed_winsorized_pearson_ic"], atol=1e-10, rtol=0):
        raise ValueError("Frozen winsorized Pearson values did not reconcile.")
    deciles = pd.concat([decile_analysis(x, target, "research_prefix_0_7859"),
                         decile_analysis(oos_x, target.loc[blocks.index], "phase3_eligible_oos")], ignore_index=True)
    pc, sc, bc, pairs, long = correlation_structure(oos_x, blocks)
    links, clusters = average_linkage(sc, cut_distance=.2)
    sensitivity = []
    for threshold in [.1, .2, .3]:
        _, grouped = average_linkage(sc, threshold)
        sensitivity.append({"diagnostic_type": DIAGNOSTIC_TYPE, "cut_distance": threshold,
                            "cluster_count": len(grouped), "exploratory_choice": True})
    lag_audit = []
    for name in ["lagged_forward_returns", "lagged_market_forward_excess_returns"]:
        source = name.removeprefix("lagged_")
        expected = frame[source].shift(1)
        exact = np.allclose(x[name], expected, atol=0, rtol=0, equal_nan=True)
        if not exact:
            raise ValueError("Lagged outcome field does not equal the one-row past source.")
        lag_audit.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name,
                          "source_outcome": source, "source_date_offset": -1,
                          "exact_previous_row_match": bool(exact), "first_row_missing": bool(pd.isna(x[name].iloc[0]))})
    tables = {"lag_decay": lag, "lag_decay_summary": wide, "reverse_return_ic": reverse,
        "decile_analysis": deciles, "rank_pearson_divergence": divergence,
        "pass_feature_correlations": long, "high_correlation_pairs": long[long["spearman_corr"].abs().ge(.8)].copy(),
        "cluster_summary": clusters, "hierarchical_linkage": links,
        "cluster_cut_sensitivity": pd.DataFrame(sensitivity), "composite_lag_decay": composite_lag,
        "composite_lag_decay_summary": composite_wide, "k0_reconciliation": pd.DataFrame(reconciliation),
        "lagged_field_timing_audit": pd.DataFrame(lag_audit),
        "fold_level_diagnostics": sealed.folds.loc[sealed.folds["hypothesis"].isin(EXPECTED_SINGLES + EXPECTED_COMPOSITES)].assign(diagnostic_type=DIAGNOSTIC_TYPE)}
    for name, matrix in [("pearson", pc), ("spearman", sc), ("blockwise_spearman", bc), ("pair_counts", pairs)]:
        tables[f"{name}_correlation_matrix"] = matrix.rename_axis("feature").reset_index().assign(diagnostic_type=DIAGNOSTIC_TYPE)
    return tables


def run_posthoc(root: Path) -> dict:
    root = root.resolve()
    output = root / "reports/phase3_posthoc_diagnostics"
    report = root / "reports/phase3_posthoc_diagnostics.md"
    if output.exists() or report.exists():
        raise FileExistsError("Do not overwrite an existing post-hoc diagnostic snapshot.")
    run_commit = git(root, "rev-parse", "HEAD").decode().strip()
    source_dirty = bool(git(root, "status", "--porcelain").strip())
    started = utc_timestamp()
    verify_protocol_lock(root, verify_data=False)
    sealed = load_sealed_phase3(root)
    try:
        lock = verify_protocol_lock(root, verify_data=True)
        design = load_alpha_design(root)
        config = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
        path = inside_root(root, f"{config['data']['raw_dir']}/{config['data']['train_file']}")
        policy = json.loads((root / "configs/split_policy.json").read_text(encoding="utf-8"))
        policy = {**policy, "train_columns": read_header(path)}
        history = load_research_prefix(path, policy, design.validation)
        tables = diagnostic_tables(history, root, sealed)
        from src.research.posthoc_reporting import write_diagnostic_report
        output.mkdir(parents=True)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
        report.write_text(write_diagnostic_report(tables, sealed.manifest, output, report), encoding="utf-8", newline="\n")
        code_names = [f"src/research/{name}.py" for name in ["posthoc_statistics", "posthoc_diagnostics", "posthoc_reporting"]]
        code = {name: hashlib.sha256((root / name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() for name in code_names}
        manifest = {"diagnostic_type": "posthoc", "analysis_label": DIAGNOSTIC_TYPE,
            "git_commit": run_commit,
            "git_branch": git(root, "branch", "--show-current").decode().strip(),
            "git_worktree_dirty": source_dirty,
            "run_timestamp": started, "completed_at_utc": utc_timestamp(), "data_start": 0, "data_end": 7859,
            "data_rows": len(history), "data_identity_sha256": lock["data_sha256"],
            "source_phase3_commit": sealed.manifest["git_commit"], "source_phase3_tag": sealed.manifest["git_tag"],
            "source_phase3_results_commit": RESULTS_COMMIT,
            "single_feature_count": 24, "composite_count": 4,
            "internal_holdout_used": False, "final_180_used": False, "phase3_results_modified": False,
            "phase3_confirmatory_rerun": False, "new_pvalues_or_fdr": False,
            "protocol_version": lock["protocol_version"], "protocol_modified": False,
            "frozen_phase3_pipeline_modified": False,
            "lag_scope": "original eligible OOS folds; shifts on the consecutive full research grid",
            "composite_lag_scope": "shift reconstructed historical OOS C_t; first k OOS rows remain missing",
            "decile_scopes": ["full research prefix", "original eligible OOS"],
            "drop_definition": "abs(IC_k0)-abs(IC_k1); relative divides by abs(IC_k0)",
            "clustering": {"method": "average_linkage", "distance": "1-abs(pooled_OOS_Spearman)",
                "cut_distance": .2, "cut_sensitivity": [.1, .2, .3], "exploratory_choice": True,
                "high_pair_display_threshold": .8, "independence_not_established": True},
            "diagnostic_code_sha256": code, "protected_phase3_sha256_before": sealed.protected_hashes,
            "output_sha256": {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob("*")) if p.is_file()},
            "report_sha256": sha256(report)}
        assert_preserved(root, sealed.protected_hashes)
        manifest["protected_phase3_sha256_after"] = protected_hashes(root)
        write_json(output / "diagnostic_manifest.json", manifest)
        return manifest
    finally:
        assert_preserved(root, sealed.protected_hashes)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--run-posthoc", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_posthoc:
        parser.error("An explicit --run-posthoc is required; no formal Phase 3 run is invoked.")
    manifest = run_posthoc(args.root)
    print(json.dumps({key: manifest[key] for key in ["diagnostic_type", "git_commit", "data_start", "data_end",
        "single_feature_count", "composite_count", "phase3_results_modified", "phase3_confirmatory_rerun"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
