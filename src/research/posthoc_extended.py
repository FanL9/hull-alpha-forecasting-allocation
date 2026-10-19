"""Extended post-hoc timing diagnostics: long-lag decay and ΔX change signals.

The 0–5 day lag test can reveal a one-day timing error but not slow look-ahead
(centered smoothing, revised or backfilled levels), because persistent inputs
barely move over a few rows. This module extends the shift to months and
replaces levels by changes. It reuses the sealed Phase 3 grid, masks and
composite recipes; it computes no p-values, FDR or confirmatory statistics.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.availability import available_lagged_outcomes
from src.research.alpha_pipeline import load_alpha_design, validate_research_history
from src.research.posthoc_diagnostics import (EXPECTED_COMPOSITES, EXPECTED_SINGLES, SealedPhase3,
    assert_preserved, frozen_oos_grid, git, load_sealed_phase3, protected_hashes,
    reconstruct_composites, sha256)
from src.research.posthoc_statistics import DIAGNOSTIC_TYPE, rank_ic
from src.utils.paths import inside_root, project_root, utc_timestamp, write_json
from src.validation.protocol_freeze import verify_protocol_lock
from src.validation.time_split import read_header
from src.validation.walk_forward import load_research_prefix

LONG_LAGS = (0, 1, 5, 21, 63, 126, 252)
CHANGE_HORIZONS = (1, 5, 21)
OUTPUT = "reports/phase3_posthoc_extended"
REPORT = "reports/phase3_posthoc_extended.md"
PRIOR_DIAGNOSTICS = ("reports/phase3_posthoc_diagnostics", "reports/phase3_posthoc_diagnostics.md")


def long_lag_decay(x: pd.DataFrame, y: pd.Series, blocks: pd.Series, eligible: pd.DataFrame,
                   lags=LONG_LAGS) -> pd.DataFrame:
    """IC(X_{t−k}, y_t) on the frozen OOS grid, natural and common-sample.

    Shifts happen on the full consecutive research grid before OOS selection,
    so X_{t−k} may come from earlier training rows but never from t' > t.
    The common sample keeps only rows where every lag is observed, so changes
    across k are not driven by coverage.
    """
    if not x.index.equals(y.index) or not x.index.to_series().diff().dropna().eq(1).all():
        raise ValueError("Use consecutive aligned research dates before shifting.")
    if not eligible.index.equals(blocks.index) or list(eligible.columns) != list(x.columns):
        raise ValueError("Frozen OOS masks and block labels must align.")
    dates, labels = blocks.index, blocks.to_numpy()
    rows = []
    for name in x:
        target = y.loc[dates].where(eligible[name]).to_numpy()
        shifted = {lag: x[name].shift(lag).loc[dates].where(eligible[name]).to_numpy() for lag in lags}
        common = np.isfinite(target) & np.logical_and.reduce([np.isfinite(v) for v in shifted.values()])
        for lag, values in shifted.items():
            natural, n = rank_ic(values, target, labels)
            fixed, n_common = rank_ic(np.where(common, values, np.nan), np.where(common, target, np.nan), labels)
            rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name, "lag_k": lag,
                         "rank_ic": natural, "n_obs": n,
                         "common_sample_rank_ic": fixed, "common_sample_n_obs": n_common})
    return pd.DataFrame(rows)


def change_signal_ic(x: pd.DataFrame, y: pd.Series, blocks: pd.Series, eligible: pd.DataFrame,
                     horizons=CHANGE_HORIZONS) -> pd.DataFrame:
    """IC of ΔX_h = X_t − X_{t−h} beside the level IC on the same rows.

    For a slow level effect, short changes carry little of the level's IC.
    A daily change that dominates the level points to information arriving
    in the latest update, which deserves timing scrutiny.
    """
    if not x.index.equals(y.index) or not x.index.to_series().diff().dropna().eq(1).all():
        raise ValueError("Use consecutive aligned research dates before differencing.")
    dates, labels = blocks.index, blocks.to_numpy()
    rows = []
    for name in x:
        target = y.loc[dates].where(eligible[name]).to_numpy()
        level = x[name].loc[dates].where(eligible[name]).to_numpy()
        for horizon in horizons:
            change = x[name].diff(horizon).loc[dates].where(eligible[name]).to_numpy()
            both = np.isfinite(level) & np.isfinite(change) & np.isfinite(target)
            level_ic, n = rank_ic(np.where(both, level, np.nan), np.where(both, target, np.nan), labels)
            change_ic, _ = rank_ic(np.where(both, change, np.nan), np.where(both, target, np.nan), labels)
            nonzero = float(np.mean(change[both] != 0)) if both.any() else np.nan
            rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name, "horizon_h": horizon,
                         "level_rank_ic": level_ic, "change_rank_ic": change_ic,
                         "change_to_level_ratio": abs(change_ic) / abs(level_ic) if level_ic else np.nan,
                         "nonzero_change_fraction": nonzero, "n_obs": n})
    return pd.DataFrame(rows)


def summarize_long_lag(table: pd.DataFrame) -> pd.DataFrame:
    """Wide common-sample view with the share of k=0 magnitude kept at each lag."""
    wide = table.pivot(index="feature", columns="lag_k", values="common_sample_rank_ic")
    wide = wide.loc[table["feature"].drop_duplicates()]
    result = pd.DataFrame({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": wide.index,
                           "common_sample_n_obs": table.groupby("feature", sort=False)["common_sample_n_obs"].first().to_numpy()})
    for lag in wide.columns:
        result[f"IC_k{lag}"] = wide[lag].to_numpy()
    base = wide[0].abs().replace(0, np.nan)
    for lag in [lag for lag in wide.columns if lag]:
        result[f"retained_k{lag}"] = (np.sign(wide[lag]) * np.sign(wide[0]) * wide[lag].abs() / base).to_numpy()
    return result.reset_index(drop=True)


def extended_tables(history: pd.DataFrame, root: Path, sealed: SealedPhase3) -> dict[str, pd.DataFrame]:
    design = load_alpha_design(root)
    validate_research_history(history, design)
    frame = history.set_index("date_id")
    candidates = pd.concat([frame[[c for c in design.single_features if c in frame]],
                            available_lagged_outcomes(history).set_axis(frame.index)], axis=1)
    x = candidates.loc[:, list(EXPECTED_SINGLES)]
    target = frame[design.rules["primary_target"]]
    blocks, masks = frozen_oos_grid(sealed)
    singles, composites = masks.loc[:, list(EXPECTED_SINGLES)], masks.loc[:, list(EXPECTED_COMPOSITES)]
    composite = reconstruct_composites(candidates, sealed)
    single_lag = long_lag_decay(x, target, blocks, singles)
    composite_lag = long_lag_decay(composite, target, blocks, composites)
    for table, original in [(single_lag, sealed.singles), (composite_lag, sealed.composites)]:
        saved = original.set_index("feature")
        for row in table.loc[table["lag_k"].eq(0)].to_dict("records"):
            if not np.isclose(row["rank_ic"], saved.loc[row["feature"], "oos_rank_ic"], atol=1e-10, rtol=0):
                raise ValueError(f"k=0 does not reconcile to sealed Phase 3: {row['feature']}")
    return {"long_lag_decay": single_lag, "long_lag_summary": summarize_long_lag(single_lag),
            "composite_long_lag_decay": composite_lag,
            "composite_long_lag_summary": summarize_long_lag(composite_lag),
            "change_signal_ic": change_signal_ic(x, target, blocks, singles)}


def _markdown(frame: pd.DataFrame, digits: int = 3) -> list[str]:
    def cell(value):
        return f"{value:+.{digits}f}" if isinstance(value, float) and np.isfinite(value) else ("—" if isinstance(value, float) else str(value))
    lines = ["| " + " | ".join(frame.columns) + " |", "| " + " | ".join("---" for _ in frame.columns) + " |"]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return lines


def write_extended_report(tables: dict[str, pd.DataFrame], output: Path, report: Path) -> str:
    link = output.relative_to(report.parent).as_posix() + "/"
    lag_columns = ["feature", *[f"IC_k{k}" for k in LONG_LAGS]]
    change = tables["change_signal_ic"].pivot(index="feature", columns="horizon_h", values="change_rank_ic")
    change = change.loc[tables["change_signal_ic"]["feature"].drop_duplicates()]
    level = tables["change_signal_ic"].groupby("feature", sort=False)["level_rank_ic"].first()
    change_view = pd.DataFrame({"feature": change.index, "level IC": level.loc[change.index].to_numpy(),
                                **{f"ΔX_{h} IC": change[h].to_numpy() for h in CHANGE_HORIZONS}})
    lines = ["# Phase 3 Extended Post-hoc Timing Diagnostics", "",
        "**This analysis is post-hoc and does not alter the pre-registered Phase 3 conclusions.**",
        "It is exploratory: no p-values, FDR or new pass criteria are computed.", "",
        "## Why this extension", "",
        "The 0–5 day lag test rules out a one-day misalignment for most signals, but it cannot detect slow look-ahead "
        "(for example centered smoothing, later revisions or backfilled levels): a persistent input hardly changes over a few rows.",
        f"This extension shifts X by k ∈ {{{', '.join(map(str, LONG_LAGS))}}} rows and tests changes ΔX_h = X_t − X_(t−h) for h ∈ {{{', '.join(map(str, CHANGE_HORIZONS))}}}.",
        "All shifts use only earlier rows of date_id 0–7859 and the sealed Phase 3 OOS grid, eligibility masks and composite recipes; k=0 reconciles to the sealed results.", "",
        "Reading guide (descriptive, not decision rules):", "",
        "- IC that stays similar out to k=126–252 suggests a slow level/regime association, with few independent observations.",
        "- IC that holds for small k and then collapses near one lag can indicate a smoothing or publication window of about that length.",
        "- A daily change ΔX_1 with IC comparable to the level suggests the information sits in the latest update and needs timing scrutiny.", "",
        "## Single-feature long-lag decay (common sample)", "",
        *_markdown(tables["long_lag_summary"][lag_columns]), "",
        "## Composite long-lag decay (common sample)", "",
        "Composites exist only on the OOS grid, so long shifts leave early rows missing and the common sample is shorter.", "",
        *_markdown(tables["composite_long_lag_summary"][["feature", "common_sample_n_obs", *lag_columns[1:]]]), "",
        "## Level versus change signals", "",
        *_markdown(change_view), "",
        f"Full tables: [long lag]({link}long_lag_decay.csv), [long-lag summary]({link}long_lag_summary.csv), "
        f"[composite long lag]({link}composite_long_lag_decay.csv), [change signals]({link}change_signal_ic.csv).", "",
        "## Limitations", "",
        "Long-lag and change diagnostics cannot certify feature release times. Rows inside a 252-row block are strongly dependent for slow inputs, "
        "so similar IC across lags does not imply many independent confirmations. The Internal Holdout remains the decisive out-of-sample check.", "",
        "This post-hoc diagnostic analysis does not alter the frozen Phase 3 PASS/FAIL/NOT_TESTABLE conclusions.",
        "No Internal Holdout or Final-180 data were used.", ""]
    return "\n".join(lines)


def run_extended(root: Path) -> dict:
    root = root.resolve()
    output, report = root / OUTPUT, root / REPORT
    if output.exists() or report.exists():
        raise FileExistsError("Do not overwrite an existing extended diagnostic snapshot.")
    run_commit = git(root, "rev-parse", "HEAD").decode().strip()
    source_dirty = bool(git(root, "status", "--porcelain").strip())
    started = utc_timestamp()
    verify_protocol_lock(root, verify_data=False)
    sealed = load_sealed_phase3(root)
    prior = {p: sha256(root / p) if (root / p).is_file() else None for p in PRIOR_DIAGNOSTICS}
    prior.update({f.relative_to(root).as_posix(): sha256(f)
                  for f in sorted((root / PRIOR_DIAGNOSTICS[0]).rglob("*")) if f.is_file()})
    try:
        lock = verify_protocol_lock(root, verify_data=True)
        design = load_alpha_design(root)
        config = yaml.safe_load((root / "configs/base.yaml").read_text(encoding="utf-8"))
        path = inside_root(root, f"{config['data']['raw_dir']}/{config['data']['train_file']}")
        policy = json.loads((root / "configs/split_policy.json").read_text(encoding="utf-8"))
        policy = {**policy, "train_columns": read_header(path)}
        history = load_research_prefix(path, policy, design.validation)
        tables = extended_tables(history, root, sealed)
        output.mkdir(parents=True)
        for name, table in tables.items():
            table.to_csv(output / f"{name}.csv", index=False, lineterminator="\n")
        report.write_text(write_extended_report(tables, output, report), encoding="utf-8", newline="\n")
        code = "src/research/posthoc_extended.py"
        manifest = {"diagnostic_type": "posthoc_extended", "analysis_label": DIAGNOSTIC_TYPE,
            "git_commit": run_commit, "git_branch": git(root, "branch", "--show-current").decode().strip(),
            "git_worktree_dirty": source_dirty, "run_timestamp": started, "completed_at_utc": utc_timestamp(),
            "data_start": 0, "data_end": 7859, "data_rows": len(history), "data_identity_sha256": lock["data_sha256"],
            "source_phase3_commit": sealed.manifest["git_commit"], "source_phase3_tag": sealed.manifest["git_tag"],
            "long_lags": list(LONG_LAGS), "change_horizons": list(CHANGE_HORIZONS),
            "shift_scope": "full consecutive research grid 0–7859, then sealed OOS grid and masks",
            "internal_holdout_used": False, "final_180_used": False, "phase3_results_modified": False,
            "phase3_confirmatory_rerun": False, "new_pvalues_or_fdr": False,
            "protocol_version": lock["protocol_version"], "protocol_modified": False,
            "diagnostic_code_sha256": {code: hashlib.sha256((root / code).read_bytes().replace(b"\r\n", b"\n")).hexdigest()},
            "protected_phase3_sha256": sealed.protected_hashes, "prior_diagnostics_sha256": prior,
            "output_sha256": {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
            "report_sha256": sha256(report)}
        write_json(output / "extended_manifest.json", manifest)
        return manifest
    finally:
        assert_preserved(root, sealed.protected_hashes)
        after = {p: sha256(root / p) if (root / p).is_file() else None for p in PRIOR_DIAGNOSTICS}
        after.update({f.relative_to(root).as_posix(): sha256(f)
                      for f in sorted((root / PRIOR_DIAGNOSTICS[0]).rglob("*")) if f.is_file()})
        if after != prior:
            raise ValueError("Earlier post-hoc diagnostics changed; extended diagnostics must never alter them.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--run-extended", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_extended:
        parser.error("An explicit --run-extended is required; no formal Phase 3 run is invoked.")
    manifest = run_extended(args.root)
    print(json.dumps({key: manifest[key] for key in ["diagnostic_type", "git_commit", "data_start", "data_end",
        "long_lags", "change_horizons", "phase3_results_modified", "phase3_confirmatory_rerun"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
