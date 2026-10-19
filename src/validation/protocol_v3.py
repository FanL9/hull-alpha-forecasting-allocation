"""Metadata-only Protocol v3 contracts and integrity checks; no research runner.

Inputs are configuration, date IDs, code bytes and archived report bytes only.
No market loader, transformation, correlation estimator or model is executed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess

import yaml

from src.utils.paths import project_root, utc_timestamp
from src.validation.protocol_freeze import verify_protocol_lock
from src.validation.walk_forward import load_protocol

CONFIG = "configs/protocol_v3.yaml"
DOCUMENT = "docs/protocol_v3_engineered_and_deep_alpha.md"
LOCK = "configs/protocol_v3_lock.json"
TAG = "protocol-v3"
NORMATIVE = (CONFIG, DOCUMENT, "src/validation/protocol_v3.py", "tests/test_protocol_v3.py")
FAMILIES = ("M", "E", "I", "P", "V", "S", "D")
FIT_ROLES = ("selector", "regime_threshold", "scaler", "model", "score_normalizer", "residualizer")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path, *, text: bool = True) -> str:
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def raw_names(config: dict) -> tuple[str, ...]:
    counts = config["input_universe"]["anonymous_counts"]
    return tuple(f"{f}{i}" for f in FAMILIES for i in range(1, counts[f] + 1)) + tuple(config["input_universe"]["lagged_fields"])


def candidate_ids(config: dict) -> tuple[str, ...]:
    return (tuple(f"raw/{n}/{t}" for n in raw_names(config) for t in config["transformations"]["names"])
            + tuple(f"family/{f}/Level" for f in FAMILIES)
            + tuple(f"interaction/{f}/HighV" for f in FAMILIES))


@dataclass(frozen=True)
class TemporalSplit:
    training_start: int
    training_end: int
    purge_id: int
    validation_start: int
    validation_end: int


def inner_splits(reference_end: int, config: dict) -> tuple[TemporalSplit, ...]:
    require(type(reference_end) is int and 0 <= reference_end <= config["research"]["end"], "Reference must remain inside research.")
    rows = config["inner"]["block_rows"]
    first = reference_end - config["inner"]["blocks"] * rows + 1
    require(first >= 2, "Insufficient reference history; no adaptive split shortening.")
    return tuple(TemporalSplit(0, start - 2, start - 1, start, start + rows - 1)
                 for start in range(first, reference_end + 1, rows))


def stopping_split(reference_end: int, config: dict) -> TemporalSplit:
    require(type(reference_end) is int and 0 <= reference_end <= config["research"]["end"], "Stopping reference outside research.")
    start = reference_end - config["training"]["early_stopping"]["rows"] + 1
    require(start >= 2, "Insufficient history for the fixed stopping block.")
    return TemporalSplit(0, start - 2, start - 1, start, reference_end)


def assert_fit_scope(role: str, fit_start: int, fit_end: int, split: TemporalSplit, config: dict) -> None:
    """Validate metadata of a future fit; this function cannot fit anything."""
    require(role in FIT_ROLES, "Unknown fit role.")
    require(all(type(v) is int for v in (fit_start, fit_end)), "Integer fit boundaries required.")
    require(0 <= fit_start <= fit_end <= split.training_end < split.purge_id < split.validation_start,
            f"{role} may fit only the purged training prefix, never validation.")
    require(split.validation_end <= config["research"]["end"], "Phase 4/5 cannot access holdout/final.")


def assert_source_date(source_date: int, decision_date: int, config: dict) -> None:
    require(type(source_date) is int and type(decision_date) is int, "Integer source dates required.")
    require(0 <= source_date <= decision_date <= config["research"]["end"], "Future, pre-history or held-out source forbidden.")


def parameter_counts(config: dict) -> dict[str, int]:
    p, h = config["deep"]["input_channels"], config["models"]["gru"]["hidden_size"]
    c, k = config["models"]["tcn"]["channels"], config["models"]["tcn"]["kernel_size"]
    gru = 3 * h * p + 3 * h * h + 6 * h + h + 1
    tcn = c * (p * k + 1) + c * (c * k + 1) + c * (p + 1) + 2 * c * (c * k + 1) + c + 1
    return {"ridge": p * config["deep"]["lookback"] + 1, "gru": gru, "tcn": tcn,
            "nonlinear_ensemble": len(config["deep"]["seeds"]) * (gru + tcn)}


def validate_config(c: dict) -> None:
    require(c["protocol_version"] == "v3" and c["parent_protocol"] == "phase2.5-v2", "Protocol ancestry differs.")
    require(c["status"] == "frozen_before_real_phase4_phase5_results" and c["freeze_tag"] == TAG, "Freeze identity differs.")
    require(c["scope"] == "protocol_only_no_research_execution", "This module authorizes no experiments.")
    require(c["target"] == "market_forward_excess_returns", "Target changed.")
    r = c["research"]
    require((r["start"], r["end"]) == (0, 7859) and r["loader"] == "src.validation.walk_forward.load_research_prefix", "Research boundary/loader changed.")
    require(not any(r[k] for k in ["real_xy_access_this_task", "phase4_executed", "phase5_executed"]), "Protocol task cannot execute research.")
    require(r["prior_phase3_and_posthoc_results_observed"] and r["prior_delta5_delta21_diagnostics_observed"], "Prior observation disclosure required.")
    require(c["outer"] == {"initial_train_size": 5087, "validation_window": 252, "step_size": 252,
        "purge": 1, "embargo": 0, "folds": 11, "shuffle": False, "refit": "frozen_within_validation_block"}, "Outer v2 windows changed.")
    require(c["inner"]["blocks"] == 2 and c["inner"]["block_rows"] == 252 and c["inner"]["purge"] == 1
            and c["inner"]["expanding"] and not c["inner"]["outer_validation_used"], "Inner temporal contract violated.")
    u = c["input_universe"]
    require(u["anonymous_counts"] == dict(zip(FAMILIES, [18, 20, 9, 13, 13, 12, 9], strict=True)), "Anonymous schema differs.")
    require(u["lagged_fields"] == ["lagged_forward_returns", "lagged_market_forward_excess_returns", "lagged_risk_free_rate"]
            and u["lagged_source_offset"] == -1 and not u["phase3_pass_global_prefilter"], "Lagged fields or global prefilter invalid.")
    require(len(raw_names(c)) == u["raw_count"] == 97, "Raw input count differs.")
    t = c["transformations"]
    require(t["names"] == ["Level", "Delta5", "Delta21", "Z252", "Rank63"] and t["max_per_raw"] == 5
            and t["delta_windows"] == [5, 21], "Transformation universe is closed.")
    require(not t["centered"] and t["allowed_max_source_offset"] == 0 and not t["impute_before_transform"], "Future-looking/imputed rolling forbidden.")
    require(t["zscore"] == {"window": 252, "window_unit": "trading_rows", "min_periods": 126,
        "ddof": 0, "include_current": True, "zero_std": "nan"}, "Z252 definition differs.")
    require(t["rank"] == {"window": 63, "window_unit": "observed_values", "min_periods": 32,
        "include_current": True, "ties": "mid_ecdf", "missing_current": "nan"}, "Rank63 definition differs.")
    require(c["family_composites"]["families"] == list(FAMILIES) and c["family_composites"]["fit_on"] == "purged_reference_training_only"
            and not c["family_composites"]["phase3_pass_filter"] and c["family_composites"]["refit_inside_each_inner_block"], "Family construction scope differs.")
    v = c["regime"]
    require(v["source"] == "training_rebuilt_V_family_composite" and v["fit_on"] == "purged_reference_training_only"
            and not v["validation_y_used"] and v["states"] == 2 and v["threshold"] == "training_median"
            and v["quantile"] == .5 and not v["physical_volatility_order_identified"], "Regime scope or definition differs.")
    require(c["interactions"]["families"] == list(FAMILIES) and c["interactions"]["count"] == 7
            and c["interactions"]["forms"] == ["family_composite_times_high_indicator"]
            and not c["interactions"]["low_interactions"] and not c["interactions"]["additional_pairwise"], "Interactions must be finite and enumerated.")
    require(c["candidates"] == {"raw_representations": 485, "family_levels": 7, "regime_interactions": 7,
        "maximum_total": 499, "adaptive_additions": False} and len(set(candidate_ids(c))) == 499, "Candidate inventory differs.")
    a = c["availability"]
    require(a == {"min_observed": 252, "min_post_launch_coverage": .8, "recent_window": 504, "max_recent_missing": .25,
        "min_distinct": 2, "on_transformed_values": True, "reference_training_only": True}, "Availability differs from v2 principles.")
    s = c["selection"]
    require(s["fit_on"] == "reference_training_only" and s["method"] == "top_k" and s["top_k"] == 16
            and s["fixed_ic_threshold"] is None and s["direction"] == "both_inner_ics_strictly_same_nonzero_sign", "Nested Top-K rule differs.")
    require(s["redundancy"]["fit_on"] == "reference_training_only" and s["redundancy"]["absolute_threshold"] == .9
            and s["redundancy"]["keep_at_most"] == 16, "Redundancy may not use outer validation.")
    d = c["deep"]
    require(d["representative_fit_on"] == "each_purged_reference_training_only" and not d["phase3_posthoc_clusters_reused"]
            and d["representatives_max"] == d["input_channels"] == 16 and d["cluster_cut_distance"] == .1, "Deep inputs require nested representatives.")
    require(d["lookback"] == 63 and d["max_parameters"] == 10000 and d["seeds"] == [11, 22, 33, 44, 55], "Deep budget/window/seeds differ.")
    require(d["scaling"]["fit_on"] == "purged_reference_training_only" and d["scaling"]["method"] == "observed_training_mean_std"
            and not d["scaling"]["clip_target"] and d["fit_on"] == "purged_reference_training_only", "Scaler/model fit scope differs.")
    require(set(c["models"]) == {"ridge", "gru", "tcn"} and c["models"]["ridge"]["alphas"] == [.1, 1., 10.]
            and c["models"]["gru"]["settings"] == c["models"]["tcn"]["settings"] == 1, "Model candidate universe differs.")
    g, tcn = c["models"]["gru"], c["models"]["tcn"]
    require((g["layers"], g["hidden_size"], g["input_channels"], g["bias_vectors_per_gate"]) == (1, 8, 16, 2)
            and (tcn["blocks"], tcn["convs_per_block"], tcn["channels"], tcn["kernel_size"]) == (2, 2, 8, 3)
            and tcn["dilations"] == [1, 2] and tcn["padding"] == "left_only_d_times_kernel_minus_one"
            and tcn["pooling"] == "mean_all_63_causal_positions" and not tcn["batch_norm"], "Temporal architecture/causality differs.")
    counts = parameter_counts(c)
    require(counts == {"ridge": 1009, "gru": 633, "tcn": 1137, "nonlinear_ensemble": 8850}
            and max(counts.values()) <= d["max_parameters"], "Model parameter budget differs.")
    require(g["trainable_parameters"] == counts["gru"] and tcn["trainable_parameters"] == counts["tcn"]
            and c["models"]["ridge"]["maximum_fitted_coefficients"] == counts["ridge"]
            and c["nonlinear_pipeline"]["total_ensemble_parameters"] == counts["nonlinear_ensemble"], "Declared parameter count differs from architecture.")
    es = c["training"]["early_stopping"]
    require(es["fit_on"] == "last_252_inside_reference_training_only" and es["rows"] == 252 and es["purge"] == 1
            and not es["outer_validation_used"] and es["selector_scaler_fit_on"] == "strictly_earlier_weight_training_prefix"
            and (es["minimum_checkpoint_epoch"], es["max_epochs"], es["patience"], es["min_delta"]) == (10, 100, 10, .0001), "Early stopping cannot use outer validation.")
    require(c["nonlinear_pipeline"]["weights"] == {"gru": .5, "tcn": .5}, "C weights changed.")
    require(c["incremental"]["fit_on"] == "two_inner_oos_predictions_inside_reference_training_only"
            and not c["incremental"]["outer_y_used_to_fit"], "Residualization must be training-only OOF.")
    require(c["combined"]["weights"] == {k: 1 / 3 for k in ["A", "B", "C"]}
            and c["combined"]["normalization_fit_on"] == "two_inner_oos_scores_inside_reference_training_only", "Combined weights/normalization differ.")
    p = c["primary_selection"]
    require(p["candidates"] == p["complexity_order"] == ["A", "B", "C", "D"] and p["minimum_finite_folds"] == 11
            and p["minimum_positive_folds"] == 7 and p["simplicity_margin"] == .01 and not p["exact_margin_in_near_best"], "Primary selection rule differs.")
    require(not c["evaluation"]["new_hac_inference"] and not c["evaluation"]["new_fdr_tests"], "v3 permits no new confirmatory significance exercise.")
    h, f = c["holdout"], c["final_simulation"]
    require((h["start"], h["end"], h["rows"], h["refit_end"], h["purge_id"]) == (7860, 8867, 1008, 7858, 7859)
            and (f["start"], f["end"], f["rows"]) == (8868, 9047, 180)
            and h["locked"] and f["locked"] and h["max_runs"] == f["max_runs"] == 1, "Held-out reservations must stay locked.")
    require("separate_explicit_authorization" in h["prerequisites"] and c["allocation"]["bounds"] == [0., 2.]
            and not c["allocation"]["optimization_in_v3"], "v3 does not authorize holdout/allocation execution.")


def load_config(root: Path) -> dict:
    c = yaml.safe_load((root / CONFIG).read_text(encoding="utf-8"))
    validate_config(c)
    parent = load_protocol(root / "configs/validation.yaml")
    require((parent.research_start, parent.research_end, parent.fold_count) == (0, 7859, 11), "Parent chronology changed.")
    return c


def protected_report_paths(root: Path) -> tuple[str, ...]:
    directories = ("reports/phase3_real_run", "reports/phase3_posthoc_diagnostics", "reports/phase3_posthoc_extended")
    names = [p.relative_to(root).as_posix() for d in directories for p in sorted((root / d).rglob("*")) if p.is_file()]
    names += ["reports/phase3_alpha_research_results.md", "reports/phase3_posthoc_diagnostics.md", "reports/phase3_posthoc_extended.md"]
    require(len(names) == 43 and len(set(names)) == 43, "All 43 tracked Phase 3 result files must be present.")
    return tuple(sorted(names))


def inherited_identity(root: Path) -> dict:
    verify_protocol_lock(root, verify_data=False)
    m = json.loads((root / "reports/phase3_real_run/phase3_run_manifest.json").read_text(encoding="utf-8"))
    implementation = {name: digest(root / name) for name in m["implementation_sha256"]}
    combined = hashlib.sha256(json.dumps(implementation, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(implementation == m["implementation_sha256"] and combined == m["phase3_pipeline_code_hash"], "Frozen Phase 3 implementation changed.")
    return {"phase2_5_lock_sha256": digest(root / "configs/protocol_lock.json"),
            "phase3_pipeline_code_hash": combined, "phase3_implementation_sha256": implementation,
            "protected_report_sha256": {name: digest(root / name, text=False) for name in protected_report_paths(root)}}


def build_lock(root: Path) -> dict:
    load_config(root)
    return {"protocol_version": "v3", "freeze_tag": TAG, "status": "frozen_before_real_phase4_phase5_results",
            "created_at_utc": utc_timestamp(), "text_hash_representation": "UTF-8 LF",
            "report_hash_representation": "original file bytes", "phase4_phase5_results_seen": False,
            "prior_posthoc_results_seen": True, "raw_data_opened_this_task": False,
            "normative_sha256": {name: digest(root / name) for name in NORMATIVE}, **inherited_identity(root)}


def freeze(root: Path) -> dict:
    path = root / LOCK
    if path.exists():
        return verify_lock(root)
    lock = build_lock(root)
    # Exclusive creation; an established v3 lock is never silently replaced.
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(lock, handle, indent=2)
        handle.write("\n")
    return lock


def verify_lock(root: Path) -> dict:
    load_config(root)
    lock = json.loads((root / LOCK).read_text(encoding="utf-8"))
    require(lock["protocol_version"] == "v3" and lock["freeze_tag"] == TAG and not lock["phase4_phase5_results_seen"], "Unknown freeze identity.")
    require(set(lock["normative_sha256"]) == set(NORMATIVE), "Incomplete v3 normative inventory.")
    for name, expected in lock["normative_sha256"].items():
        require(digest(root / name) == expected, f"Frozen v3 file changed: {name}")
    for key, value in inherited_identity(root).items():
        require(lock[key] == value, f"Inherited sealed evidence changed: {key}")
    return lock


def git(root: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(root), *args])


def verify_tag(root: Path) -> str:
    lock = verify_lock(root)
    require(git(root, "cat-file", "-t", f"refs/tags/{TAG}").strip() == b"tag", "v3 tag must be annotated.")
    commit = git(root, "rev-parse", f"refs/tags/{TAG}^{{commit}}").decode().strip()
    require(git(root, "merge-base", commit, "HEAD").decode().strip() == commit, "HEAD must descend from the v3 freeze.")
    for name in (*NORMATIVE, LOCK):
        blob = git(root, "show", f"{commit}:{name}").replace(b"\r\n", b"\n")
        expected = lock["normative_sha256"].get(name, digest(root / LOCK))
        require(hashlib.sha256(blob).hexdigest() == expected, "Tag does not bind this v3 freeze.")
    return commit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    mode = parser.add_mutually_exclusive_group(required=True)
    for name in ["check-config", "freeze", "check", "check-tag"]:
        mode.add_argument(f"--{name}", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.check_config:
        c = load_config(root)
        result = {"raw_inputs": len(raw_names(c)), "candidate_count": len(candidate_ids(c)), "parameter_counts": parameter_counts(c)}
    elif args.freeze:
        result = {"lock": LOCK, "protocol_version": freeze(root)["protocol_version"]}
    elif args.check_tag:
        result = {"tag": TAG, "commit": verify_tag(root)}
    else:
        result = {"protocol_version": verify_lock(root)["protocol_version"], "status": "verified"}
    print(json.dumps({**result, "real_data_read": False, "models_trained": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
