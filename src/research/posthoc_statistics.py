"""Exploratory/post-hoc summaries only: no p-values, FDR or model selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.statistics.robust_inference import blockwise_rank_pairs

DIAGNOSTIC_TYPE = "exploratory / post-hoc diagnostic"


def pearson(x, y) -> tuple[float, int]:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or np.isinf(a).any() or np.isinf(b).any():
        raise ValueError("Use aligned finite-or-NaN one-dimensional arrays.")
    complete = np.isfinite(a) & np.isfinite(b)
    n = int(complete.sum())
    if n < 3 or np.std(a[complete]) == 0 or np.std(b[complete]) == 0:
        return np.nan, n
    return float(np.corrcoef(a[complete], b[complete])[0, 1]), n


def rank_ic(x, y, blocks=None) -> tuple[float, int]:
    """Use frozen block-wise ranks when block labels are supplied; never HAC."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if blocks is None:
        complete = np.isfinite(a) & np.isfinite(b)
        a = pd.Series(np.where(complete, a, np.nan)).rank(method="average").to_numpy()
        b = pd.Series(np.where(complete, b, np.nan)).rank(method="average").to_numpy()
    else:
        a, b = blockwise_rank_pairs(a, b, blocks)
    return pearson(a, b)


def lag_decay(x: pd.DataFrame, y: pd.Series, blocks: pd.Series,
              eligible: pd.DataFrame, max_lag: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shift on the entire original date grid, then select the fixed OOS grid.

    Eligibility is the archived k=0 fold mask; no lag-specific fitting occurs.
    Common-pair k0/k1 estimates expose missingness-driven changes in coverage.
    """
    if not x.index.equals(y.index) or not x.index.is_unique or not x.index.to_series().diff().dropna().eq(1).all():
        raise ValueError("Use consecutive aligned research dates before shifting.")
    if not eligible.index.equals(blocks.index) or list(eligible.columns) != list(x.columns):
        raise ValueError("Frozen OOS masks and block labels must align.")
    rows, summary = [], []
    dates = blocks.index
    for name in x:
        target = y.loc[dates].where(eligible[name]).to_numpy()
        estimates, counts = [], []
        for lag in range(max_lag + 1):
            values = x[name].shift(lag).loc[dates].where(eligible[name]).to_numpy()
            ic, n = rank_ic(values, target, blocks.to_numpy())
            estimates.append(ic)
            counts.append(n)
            rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name,
                         "lag_k": lag, "rank_ic": ic, "n_obs": n})
        current = x[name].loc[dates].where(eligible[name]).to_numpy()
        previous = x[name].shift(1).loc[dates].where(eligible[name]).to_numpy()
        common = np.isfinite(current) & np.isfinite(previous) & np.isfinite(target)
        common_y = np.where(common, target, np.nan)
        ic0, n = rank_ic(np.where(common, current, np.nan), common_y, blocks.to_numpy())
        ic1, _ = rank_ic(np.where(common, previous, np.nan), common_y, blocks.to_numpy())
        drop = abs(estimates[0]) - abs(estimates[1])
        summary.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name,
                        **{f"IC_k{k}": value for k, value in enumerate(estimates)},
                        "drop_k0_to_k1": drop,
                        "relative_drop_k0_to_k1": drop / abs(estimates[0]) if estimates[0] else np.nan,
                        "signed_change_k0_to_k1": estimates[1] - estimates[0],
                        "sign_flip_k0_to_k1": bool(estimates[0] * estimates[1] < 0),
                        "n_obs_k0": counts[0], "n_obs_k1": counts[1], "common_pair_n_obs": n,
                        "common_pair_ic_k0": ic0, "common_pair_ic_k1": ic1,
                        "common_pair_magnitude_drop": abs(ic0) - abs(ic1)})
    return pd.DataFrame(rows), pd.DataFrame(summary)


def reverse_return_ic(x: pd.DataFrame, y: pd.Series, blocks: pd.Series,
                      eligible: pd.DataFrame, max_lag: int = 5) -> pd.DataFrame:
    rows = []
    for name in x:
        current = x[name].loc[blocks.index].where(eligible[name]).to_numpy()
        for lag in range(1, max_lag + 1):
            past = y.shift(lag).loc[blocks.index].where(eligible[name]).to_numpy()
            ic, n = rank_ic(current, past, blocks.to_numpy())
            rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name,
                         "past_return_lag": lag, "rank_ic": ic, "n_obs": n})
    return pd.DataFrame(rows)


def decile_analysis(x: pd.DataFrame, y: pd.Series, scope: str) -> pd.DataFrame:
    rows = []
    for name in x:
        pairs = pd.DataFrame({"x": x[name], "y": y}).dropna()
        if pairs.empty:
            continue
        bins, edges = pd.qcut(pairs["x"], q=10, labels=False, retbins=True, duplicates="drop")
        for decile in sorted(bins.dropna().unique()):
            group = pairs.loc[bins.eq(decile)]
            rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "scope": scope, "feature": name,
                         "decile": int(decile) + 1, "n_bins": len(edges) - 1, "n_obs": len(group),
                         "mean_y": group["y"].mean(), "median_y": group["y"].median(),
                         "mean_x": group["x"].mean(), "median_x": group["x"].median(),
                         "lower_x": edges[int(decile)], "upper_x": edges[int(decile) + 1]})
    return pd.DataFrame(rows)


def divergence_summary(raw_x: pd.DataFrame, winsor_x: pd.DataFrame, y: pd.Series,
                       blocks: pd.Series, original: pd.DataFrame) -> pd.DataFrame:
    """Additive Pearson contributions, not new outlier-removal decision gates."""
    rows = []
    indexed = original.set_index("feature")
    for name in raw_x:
        pairs = pd.DataFrame({"x": raw_x[name], "w": winsor_x[name], "y": y}).dropna()
        raw, n = pearson(pairs["x"], pairs["y"])
        winsor, _ = pearson(pairs["w"], pairs["y"])
        global_rank, _ = rank_ic(pairs["x"], pairs["y"])
        block_rank, _ = rank_ic(raw_x[name], y, blocks)
        low, high = pairs["x"].quantile([.01, .99])
        x_tail = (pairs["x"] < low) | (pairs["x"] > high)
        y_centered = pairs["y"] - pairs["y"].mean()
        y_tail = y_centered.abs() >= y_centered.abs().quantile(.99)
        centered_w = pairs["w"] - pairs["w"].mean()
        denominator = n * pairs["w"].std(ddof=0) * pairs["y"].std(ddof=0)
        contributions = centered_w * y_centered / denominator if denominator > 0 else pairs["w"] * np.nan
        inner_x, _ = pearson(pairs.loc[~x_tail, "w"], pairs.loc[~x_tail, "y"])
        inner_y, _ = pearson(pairs.loc[~y_tail, "w"], pairs.loc[~y_tail, "y"])
        rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature": name, "n_obs": n,
                     "frozen_oos_rank_ic": indexed.loc[name, "oos_rank_ic"],
                     "frozen_winsorized_pearson_ic": indexed.loc[name, "winsorized_pearson_ic"],
                     "reconstructed_blockwise_rank_ic": block_rank,
                     "oos_global_spearman_ic": global_rank, "oos_raw_pearson_ic": raw,
                     "reconstructed_winsorized_pearson_ic": winsor,
                     "x_tail_n": int(x_tail.sum()), "x_q01": low, "x_q99": high,
                     "x_min": pairs["x"].min(), "x_max": pairs["x"].max(),
                     "x_tail_winsor_corr_contribution": contributions[x_tail].sum(),
                     "x_core_winsor_corr_contribution": contributions[~x_tail].sum(),
                     "y_extreme_n": int(y_tail.sum()),
                     "y_extreme_winsor_corr_contribution": contributions[y_tail].sum(),
                     "y_core_winsor_corr_contribution": contributions[~y_tail].sum(),
                     "pearson_excluding_x_tail_descriptive": inner_x,
                     "pearson_excluding_extreme_y_descriptive": inner_y})
    return pd.DataFrame(rows)


def correlation_structure(x: pd.DataFrame, blocks: pd.Series):
    names = list(x.columns)
    p, s, b, n = [pd.DataFrame(np.nan, index=names, columns=names) for _ in range(4)]
    rows = []
    for i, a in enumerate(names):
        for j in range(i, len(names)):
            z = names[j]
            pc, count = pearson(x[a], x[z])
            sc, _ = rank_ic(x[a], x[z])
            bc, _ = rank_ic(x[a], x[z], blocks)
            for matrix, value in [(p, pc), (s, sc), (b, bc), (n, count)]:
                matrix.loc[a, z] = matrix.loc[z, a] = value
            if i != j:
                rows.append({"diagnostic_type": DIAGNOSTIC_TYPE, "feature_a": a, "feature_b": z,
                             "pearson_corr": pc, "spearman_corr": sc,
                             "blockwise_spearman_corr": bc, "n_obs": count})
    return p, s, b, n, pd.DataFrame(rows)


def average_linkage(correlation: pd.DataFrame, cut_distance: float = .2):
    """Deterministic average linkage on 1-|rho|, with an exploratory cut.

    Missing pair correlations are conservatively assigned distance 1. Groups
    are oriented-equivalent (strong negative correlations also mean overlap).
    No feature is selected or removed, and clusters do not prove independence.
    """
    if (correlation.shape[0] != correlation.shape[1] or not correlation.index.equals(correlation.columns)
            or not 0 <= cut_distance <= 1):
        raise ValueError("Use an aligned square correlation matrix and cut in [0,1].")
    names = list(correlation.index)
    distance = 1 - np.clip(np.nan_to_num(np.abs(correlation.to_numpy(dtype=float)), nan=0.), 0, 1)
    if not np.allclose(distance, distance.T):
        raise ValueError("Correlation matrix must be symmetric.")
    np.fill_diagonal(distance, 0.)
    groups = {i: (i,) for i in range(len(names))}
    cut_groups, merges = None, []
    for step in range(max(0, len(names) - 1)):
        keys = sorted(groups)
        choices = [(float(distance[np.ix_(groups[a], groups[b])].mean()), a, b)
                   for i, a in enumerate(keys) for b in keys[i + 1:]]
        value, a, b = min(choices)
        if cut_groups is None and value > cut_distance:
            cut_groups = list(groups.values())
        members = groups.pop(a) + groups.pop(b)
        identifier = len(names) + step
        groups[identifier] = members
        merges.append({"diagnostic_type": DIAGNOSTIC_TYPE, "node": identifier, "left": a, "right": b,
                       "distance": value, "member_count": len(members)})
    if cut_groups is None:
        cut_groups = list(groups.values())
    summary = []
    for label, members in enumerate(sorted(cut_groups, key=min), 1):
        values = np.abs(correlation.iloc[list(members), list(members)].to_numpy())
        off_diagonal = values[np.triu_indices(len(members), k=1)]
        finite = off_diagonal[np.isfinite(off_diagonal)]
        summary.append({"diagnostic_type": DIAGNOSTIC_TYPE, "cluster": label,
                        "member_count": len(members), "members": ";".join(names[i] for i in sorted(members)),
                        "mean_abs_within_cluster_correlation": finite.mean() if len(finite) else np.nan,
                        "minimum_abs_within_cluster_correlation": finite.min() if len(finite) else np.nan,
                        "cut_distance": cut_distance, "exploratory_choice": True})
    return pd.DataFrame(merges), pd.DataFrame(summary)
