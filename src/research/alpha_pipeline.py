"""Phase 3 engine over an explicitly supplied research prefix; no file loading.

All inferential primitives come from the frozen v2 implementation. This module
only assembles folds, records training decisions, and retains unavailable time
positions. Neither holdout nor competition observations are accepted.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.availability import (LAGGED_OUTCOME_COLUMNS, available_lagged_outcomes,
                                       feature_availability)
from src.features.family_composites import COMPOSITE_FAMILIES, family_composites
from src.features.feature_groups import feature_family
from src.statistics.robust_inference import (apply_winsor_bounds, blockwise_spearman_hac,
                                            blockwise_spearman_interval, correlation_hac,
                                            fit_winsor_bounds)
from src.validation.walk_forward import ValidationProtocol, load_protocol, make_folds

# Observed schema, documented in docs/data_dictionary.md; no inferred MOM family.
ANONYMOUS_COUNTS = {"M": 18, "E": 20, "I": 9, "P": 13, "V": 13, "S": 12, "D": 9}


@dataclass(frozen=True)
class AlphaDesign:
    rules: dict
    validation: ValidationProtocol
    source_columns: tuple[str, ...]
    single_features: tuple[str, ...]


@dataclass
class AlphaTables:
    single_features: pd.DataFrame
    family_composites: pd.DataFrame
    fold_level_ic: pd.DataFrame
    feature_availability: pd.DataFrame
    fold_transforms: list[dict]


def load_alpha_design(root: Path) -> AlphaDesign:
    """Read frozen configuration and schema metadata only, never market rows."""
    rules = yaml.safe_load((root / "configs/research.yaml").read_text(encoding="utf-8"))["protocol"]
    validation = load_protocol(root / "configs/validation.yaml")
    anonymous = tuple(f"{family}{i}" for family, count in ANONYMOUS_COUNTS.items() for i in range(1, count + 1))
    columns = ("date_id", *anonymous, "forward_returns", "risk_free_rate", rules["primary_target"])
    candidates = anonymous + LAGGED_OUTCOME_COLUMNS
    if (rules["version"] != "phase2.5-v2" or len(anonymous) != rules["anonymous_feature_count"]
            or len(candidates) != rules["primary_hypothesis_family"]["size"]
            or tuple(rules["secondary_hypothesis_family"]["families"]) != COMPOSITE_FAMILIES
            or rules["secondary_hypothesis_family"]["size"] != len(COMPOSITE_FAMILIES)
            or rules["global_sensitivity"]["combined_hypotheses"] != len(candidates) + len(COMPOSITE_FAMILIES)):
        raise ValueError("The frozen 97 / 7 / 104 hypothesis inventory must be complete.")
    return AlphaDesign(rules, validation, columns, candidates)


def validate_research_history(history: pd.DataFrame, design: AlphaDesign) -> None:
    """Reject a wider, partial, reordered or malformed prefix before using X/Y."""
    expected = np.arange(design.validation.research_start, design.validation.research_end + 1)
    if (history.columns.duplicated().any() or set(history.columns) != set(design.source_columns)
            or "date_id" not in history):
        raise ValueError("Research input must use the exact frozen source schema.")
    dates = history["date_id"].to_numpy()
    if not np.issubdtype(dates.dtype, np.integer) or not np.array_equal(dates, expected):
        raise ValueError("Only consecutive research date_id 0–7859 is accepted; no holdout or final rows.")
    for column in history.columns:
        if not pd.api.types.is_numeric_dtype(history[column]) or np.isinf(history[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Research values must be numeric finite-or-NaN: {column}")


def _fold_diagnostic(name, scope, fold, x, y, clipped, training_eligible, rules) -> dict:
    pairs = np.isfinite(x) & np.isfinite(y)
    count = int(pairs.sum())
    coverage = count / len(y)
    reasons = []
    if not training_eligible:
        reasons.append("training_unavailable")
    if count < rules["minimum_pairs_per_fold"]:
        reasons.append("insufficient_validation_pairs")
    if coverage < rules["minimum_validation_coverage"]:
        reasons.append("insufficient_validation_coverage")
    # Undefined correlations do not create an estimable validation fold.
    if count and (np.unique(x[pairs]).size < 2 or np.unique(y[pairs]).size < 2):
        reasons.append("undefined_validation_correlation")
    valid = not reasons
    rank_ic = correlation_hac(x, y, lags=0, method="spearman").correlation if valid else np.nan
    pearson = correlation_hac(clipped, y, lags=0).correlation if valid else np.nan
    return {"hypothesis": name, "scope": scope, "fold": fold.number,
            "validation_start": fold.validation_ids[0], "validation_end": fold.validation_ids[-1],
            "training_eligible": bool(training_eligible), "eligible": valid,
            "pairs": count, "coverage": coverage, "fold_ic": rank_ic,
            "winsorized_pearson_ic": pearson, "reasons": ";".join(reasons)}


def summarize_hypothesis(name, family, x, y, clipped, blocks, diagnostics, rules) -> dict:
    """One two-sided NW p-value; bootstrap and fold stability are descriptive."""
    inference = rules["inference"]
    eligible = [d for d in diagnostics if d["eligible"]]
    valid_blocks = {d["fold"] for d in eligible}
    keep = np.isin(blocks, list(valid_blocks))
    oos_x = np.where(keep, x, np.nan)
    oos_y = np.where(keep, y, np.nan)
    estimate = blockwise_spearman_hac(oos_x, oos_y, blocks, lags=inference["newey_west_lags"])
    enough_folds = len(eligible) >= inference["minimum_eligible_folds"]
    testable = enough_folds and np.isfinite(estimate.pvalue)
    interval = {"lower": np.nan, "upper": np.nan, "valid_repetitions": 0}
    if enough_folds and np.isfinite(estimate.correlation):
        interval = blockwise_spearman_interval(
            oos_x, oos_y, blocks, block_length=inference["bootstrap_block_length"],
            repetitions=inference["bootstrap_repetitions"], confidence=inference["confidence_level"],
            seed=inference["random_seed"])
    ics = np.array([d["fold_ic"] for d in eligible], dtype=float)
    positive, negative = int((ics > 0).sum()), int((ics < 0).sum())
    pearson = correlation_hac(np.where(keep, clipped, np.nan), oos_y, lags=0).correlation
    se = estimate.standard_error
    reason = "" if testable else ("insufficient_eligible_folds" if not enough_folds else "undefined_hac_uncertainty")
    return {"feature": name, "family": family, "eligible_fold_count": len(eligible),
            "training_eligible_fold_count": sum(d["training_eligible"] for d in diagnostics),
            "status": "testable" if testable else "not_testable", "not_testable_reason": reason,
            "oos_rank_ic": estimate.correlation, "oos_pairs": estimate.pairs,
            "mean_fold_ic": float(ics.mean()) if len(ics) else np.nan,
            "std_fold_ic": float(ics.std(ddof=1)) if len(ics) > 1 else np.nan,
            "positive_fold_count": positive, "negative_fold_count": negative,
            "positive_fold_ratio": positive / len(ics) if len(ics) else np.nan,
            "direction_consistency": max(positive, negative) / len(ics) if len(ics) else np.nan,
            "nw_se": se, "nw_stat": estimate.correlation / se if np.isfinite(se) and se > 1e-15 else np.nan,
            "raw_p_value": float(estimate.pvalue) if testable else 1.0,
            "bootstrap_ci_lower": interval["lower"], "bootstrap_ci_upper": interval["upper"],
            "bootstrap_valid_repetitions": interval["valid_repetitions"],
            "winsorized_pearson_ic": pearson}


def build_alpha_tables(history: pd.DataFrame, design: AlphaDesign) -> AlphaTables:
    """Run both fixed hypothesis families, independently of single-feature BH."""
    validate_research_history(history, design)
    frame = history.set_index("date_id")
    x = pd.concat([frame[[c for c in design.single_features if c not in LAGGED_OUTCOME_COLUMNS]],
                   available_lagged_outcomes(history).set_axis(frame.index)], axis=1)
    x = x.loc[:, list(design.single_features)]
    target = frame[design.rules["primary_target"]]
    folds = make_folds(np.arange(design.validation.development_start, design.validation.development_end + 1),
                       design.validation)
    names = list(design.single_features) + list(COMPOSITE_FAMILIES)
    index = np.concatenate([f.validation_ids for f in folds])
    blocks = np.concatenate([np.full(len(f.validation_ids), f.number) for f in folds])
    signals = pd.DataFrame(np.nan, index=index, columns=names)
    winsorized = signals.copy()
    diagnostics, availability_rows, audits = [], [], []
    settings = design.rules["secondary_metric"]
    composite_rules = design.rules["secondary_hypothesis_family"]
    for fold in folds:
        train_x, validation_x = x.loc[list(fold.train_ids)], x.loc[list(fold.validation_ids)]
        train_y, validation_y = target.loc[train_x.index], target.loc[validation_x.index]
        availability = feature_availability(train_x, design.rules["feature_availability"])
        eligible = availability.loc[availability["eligible"], "feature"].tolist()
        bounds = (fit_winsor_bounds(train_x[eligible], settings["x_winsor_lower"], settings["x_winsor_upper"])
                  if eligible else pd.DataFrame(columns=["lower", "upper"], dtype=float))
        availability = availability.set_index("feature")
        availability["winsor_lower"] = bounds["lower"]
        availability["winsor_upper"] = bounds["upper"]
        availability["fold"] = fold.number
        availability["train_start"], availability["train_end"] = fold.train_ids[0], fold.train_ids[-1]
        availability_rows.append(availability.reset_index())
        if eligible:
            # Both transformations use the same training bounds; y stays unchanged.
            apply_winsor_bounds(train_x[eligible], bounds)
            winsorized.loc[validation_x.index, eligible] = apply_winsor_bounds(validation_x[eligible], bounds)
            signals.loc[validation_x.index, eligible] = validation_x[eligible]
        # Apply one training mapping to training and validation, needed for the
        # descriptive composite's own training-only winsor bounds.
        combined = pd.concat([train_x, validation_x])
        composites, metadata = family_composites(
            train_x, train_y, combined, availability.reset_index(),
            min_members=composite_rules["min_available_members"],
            min_fraction=composite_rules["min_available_fraction"])
        for family in COMPOSITE_FAMILIES:
            info = metadata[family]
            info["percentile_reference_sha256"] = {
                c: hashlib.sha256(np.sort(train_x[c].dropna().to_numpy(dtype="<f8")).tobytes()).hexdigest()
                for c in info["qualified_members"]}
            train_composite = composites.loc[train_x.index, [family]]
            bounds_c = fit_winsor_bounds(train_composite, settings["x_winsor_lower"], settings["x_winsor_upper"])
            defined = bounds_c.notna().all(axis=1).all()
            info["winsor_bounds"] = {k: float(bounds_c.loc[family, k]) if defined else None for k in ["lower", "upper"]}
            signals.loc[validation_x.index, family] = composites.loc[validation_x.index, family]
            if defined:
                apply_winsor_bounds(train_composite, bounds_c)
                winsorized.loc[validation_x.index, family] = apply_winsor_bounds(composites.loc[validation_x.index, [family]], bounds_c)[family]
        audits.append({"fold": fold.number, "train_start": fold.train_ids[0], "train_end": fold.train_ids[-1],
                       "family_composites": metadata})
        for name in names:
            single = name in design.single_features
            training_eligible = name in eligible if single else len(metadata[name]["qualified_members"]) >= composite_rules["min_available_members"]
            diagnostics.append(_fold_diagnostic(
                name, "single_features" if single else "family_composites", fold,
                signals.loc[validation_x.index, name].to_numpy(), validation_y.to_numpy(),
                winsorized.loc[validation_x.index, name].to_numpy(), training_eligible, design.rules["inference"]))
    rows = {name: summarize_hypothesis(
        name, feature_family(name) or ("lagged" if name in design.single_features else name),
        signals[name].to_numpy(), target.loc[index].to_numpy(), winsorized[name].to_numpy(), blocks,
        [d for d in diagnostics if d["hypothesis"] == name], design.rules) for name in names}
    return AlphaTables(pd.DataFrame([rows[n] for n in design.single_features]),
                       pd.DataFrame([rows[n] for n in COMPOSITE_FAMILIES]),
                       pd.DataFrame(diagnostics), pd.concat(availability_rows, ignore_index=True), audits)
