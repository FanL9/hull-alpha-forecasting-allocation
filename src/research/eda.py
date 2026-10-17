"""Descriptive development-only EDA. X and target have separate entry points."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.feature_groups import feature_family, group_features

QUANTILES = {"q01": .01, "q05": .05, "q25": .25, "q75": .75, "q95": .95, "q99": .99}


def check_dates(dates: pd.Series, length: int) -> None:
    if (length == 0 or len(dates) != length or dates.isna().any()
            or not pd.api.types.is_integer_dtype(dates) or dates.duplicated().any()
            or not dates.is_monotonic_increasing or dates.min() < 0 or dates.max() > 8867):
        raise ValueError("EDA accepts ordered development identifiers only (0..8867).")


def check_x(frame: pd.DataFrame, dates: pd.Series) -> None:
    check_dates(dates, len(frame))
    if not len(frame.columns) or any(feature_family(c) is None for c in frame):
        raise ValueError("Feature EDA accepts X only; outcomes/metadata are forbidden.")
    if not frame.index.equals(dates.index):
        raise ValueError("Feature/date indices must align.")
    if any(not pd.api.types.is_numeric_dtype(frame[c]) for c in frame):
        raise ValueError("Anonymous features must be numeric.")


def descriptive(series: pd.Series) -> dict:
    values = series.dropna()
    result = {"observed": int(len(values)), "missing_count": int(series.isna().sum()),
              "unique_nonmissing": int(values.nunique()),
              "mean": values.mean(), "std": values.std(ddof=1), "median": values.median(),
              "min": values.min(), "max": values.max(),
              "skewness": values.skew(), "excess_kurtosis": values.kurt()}
    result.update({name: values.quantile(q) for name, q in QUANTILES.items()})
    return result


def missing_intervals(frame: pd.DataFrame, dates: pd.Series) -> pd.DataFrame:
    check_x(frame, dates)
    rows = []
    for column in frame:
        mask = frame[column].isna().to_numpy()
        edges = np.diff(np.r_[False, mask, False].astype(int))
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            rows.append({"feature": column, "family": feature_family(column),
                         "start_date_id": int(dates.iloc[start]), "end_date_id": int(dates.iloc[end - 1]),
                         "rows": int(end - start), "leading": bool(start == 0), "trailing": bool(end == len(frame))})
    return pd.DataFrame(rows, columns=["feature", "family", "start_date_id", "end_date_id", "rows", "leading", "trailing"])


def trailing_statistics(frame: pd.DataFrame, dates: pd.Series, settings: dict) -> pd.DataFrame:
    """Each output at t uses at most the trailing window ending at t."""
    check_x(frame, dates)
    rolling = frame.rolling(settings["rolling_window_rows"], min_periods=settings["rolling_min_observations"])
    means, stds = rolling.mean(), rolling.std(ddof=1)
    counts = frame.notna().astype(int).rolling(settings["rolling_window_rows"], min_periods=1).sum()
    positions = sorted(set(range(0, len(frame), settings["rolling_output_stride"])) | {len(frame) - 1})
    parts = []
    for column in frame:
        parts.append(pd.DataFrame({"date_id": dates.iloc[positions].to_numpy(), "feature": column,
            "family": feature_family(column), "rolling_mean": means[column].iloc[positions].to_numpy(),
            "rolling_std": stds[column].iloc[positions].to_numpy(),
            "observed": counts[column].iloc[positions].to_numpy(dtype=int)}))
    return pd.concat(parts, ignore_index=True)


def ecdf_distance(previous: np.ndarray, current: np.ndarray) -> float:
    """Two empirical CDFs' maximum distance; no hypothesis test or p-value."""
    a, b = np.sort(previous), np.sort(current)
    if not len(a) or not len(b):
        return np.nan
    grid = np.union1d(a, b)
    return float(np.max(np.abs(np.searchsorted(a, grid, side="right") / len(a)
                             - np.searchsorted(b, grid, side="right") / len(b))))


def temporal_blocks(frame: pd.DataFrame, dates: pd.Series, block_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological blocks; compare a block only with its immediate predecessor."""
    check_x(frame, dates)
    if block_rows < 2:
        raise ValueError("Block size must be at least two rows.")
    blocks, drift = [], []
    for start in range(0, len(frame), block_rows):
        end = min(start + block_rows, len(frame))
        for column in frame:
            chunk = frame[column].iloc[start:end]
            current = chunk.dropna()
            base = {"feature": column, "family": feature_family(column),
                    "start_date_id": int(dates.iloc[start]), "end_date_id": int(dates.iloc[end - 1]),
                    "rows": end - start, "observed": len(current), "missing_fraction": float(chunk.isna().mean()),
                    "mean": current.mean(), "std": current.std(ddof=1), "median": current.median(),
                    "q05": current.quantile(.05), "q95": current.quantile(.95)}
            blocks.append(base)
            if start:
                previous = frame[column].iloc[max(0, start - block_rows):start].dropna()
                previous_std = previous.std(ddof=1)
                enough = len(previous) >= 20 and len(current) >= 20
                drift.append({"feature": column, "family": feature_family(column),
                    "previous_start_date_id": int(dates.iloc[max(0, start - block_rows)]),
                    "previous_end_date_id": int(dates.iloc[start - 1]),
                    "current_start_date_id": int(dates.iloc[start]), "end_date_id": int(dates.iloc[end - 1]),
                    "previous_observed": len(previous), "current_observed": len(current),
                    "sufficient_observations": bool(enough),
                    "ecdf_distance": ecdf_distance(previous.to_numpy(), current.to_numpy()) if enough else np.nan,
                    "mean_change_in_prior_std": (current.mean() - previous.mean()) / previous_std
                        if enough and previous_std > 0 else np.nan,
                    "std_ratio_to_prior": current.std(ddof=1) / previous_std
                        if enough and previous_std > 0 else np.nan})
    return pd.DataFrame(blocks), pd.DataFrame(drift)


def xx_structure(frame: pd.DataFrame, dates: pd.Series, min_pairs: int, threshold: float) -> dict[str, pd.DataFrame]:
    """Pairwise-complete Pearson X-X only, without filling or selecting columns."""
    check_x(frame, dates)
    corr = frame.corr(method="pearson", min_periods=min_pairs)
    present = frame.notna().astype("int64")
    counts = present.T @ present
    pairs = []
    all_pairs = []
    for i, a in enumerate(frame.columns):
        for b in frame.columns[i + 1:]:
            row = {"feature_a": a, "feature_b": b, "family_a": feature_family(a),
                   "family_b": feature_family(b), "pearson_xx": corr.loc[a, b],
                   "pairwise_observations": int(counts.loc[a, b])}
            all_pairs.append(row)
            if pd.notna(row["pearson_xx"]) and abs(row["pearson_xx"]) >= threshold:
                pairs.append(row)
    columns = ["feature_a", "feature_b", "family_a", "family_b", "pearson_xx", "pairwise_observations"]
    families = [f for f, names in group_features(frame.columns).items() if names]
    family_rows = []
    for i, a in enumerate(families):
        for b in families[i:]:
            relevant = [abs(r["pearson_xx"]) for r in all_pairs if pd.notna(r["pearson_xx"])
                        and sorted([r["family_a"], r["family_b"]]) == sorted([a, b])]
            family_rows.append({"family_a": a, "family_b": b, "defined_pairs": len(relevant),
                "median_abs_pearson_xx": float(np.median(relevant)) if relevant else np.nan,
                "max_abs_pearson_xx": float(np.max(relevant)) if relevant else np.nan})
    return {"xx_correlation": corr.rename_axis("feature"), "xx_pair_counts": counts.rename_axis("feature"),
            "high_xx_pairs": pd.DataFrame(pairs, columns=columns), "family_xx_correlation": pd.DataFrame(family_rows)}


def feature_eda(frame: pd.DataFrame, dates: pd.Series, settings: dict) -> dict[str, pd.DataFrame]:
    check_x(frame, dates)
    intervals = missing_intervals(frame, dates)
    statistics, missing, discrete = [], [], []
    for column in frame:
        series = frame[column]
        stats = descriptive(series)
        statistics.append({"feature": column, "family": feature_family(column), "dtype": str(series.dtype), **stats})
        runs = intervals[intervals["feature"] == column]
        observed_dates = dates[series.notna()]
        missing.append({"feature": column, "family": feature_family(column), "rows": len(frame),
            "missing_count": int(series.isna().sum()), "missing_fraction": float(series.isna().mean()),
            "missing_intervals": len(runs), "longest_missing_run": int(runs["rows"].max()) if len(runs) else 0,
            "leading_missing_rows": int(runs.loc[runs["leading"], "rows"].sum()),
            "trailing_missing_rows": int(runs.loc[runs["trailing"], "rows"].sum()),
            "first_observed_date_id": int(observed_dates.iloc[0]) if len(observed_dates) else np.nan,
            "last_observed_date_id": int(observed_dates.iloc[-1]) if len(observed_dates) else np.nan})
        unique = stats["unique_nonmissing"]
        if unique <= 20:
            kind = "binary" if unique == 2 else "constant" if unique == 1 else "all_missing" if unique == 0 else "discrete"
            for value, count in series.value_counts(dropna=True).sort_index().items():
                discrete.append({"feature": column, "family": feature_family(column), "kind": kind,
                                 "value": value, "count": int(count)})
    stats_df, missing_df = pd.DataFrame(statistics), pd.DataFrame(missing)
    blocks, drift = temporal_blocks(frame, dates, settings["time_block_rows"])
    family_rows, family_time = [], []
    for family, names in group_features(frame.columns).items():
        if not names:
            continue
        subset = stats_df[stats_df["family"] == family]
        family_rows.append({"family": family, "features": len(names), "missing_count": int(frame[names].isna().sum().sum()),
            "missing_fraction": float(frame[names].isna().to_numpy().mean()),
            "median_feature_mean": subset["mean"].median(), "median_feature_std": subset["std"].median(),
            "smallest_feature_std": subset["std"].min(), "largest_feature_std": subset["std"].max(),
            "median_abs_feature_skewness": subset["skewness"].abs().median(),
            "median_feature_excess_kurtosis": subset["excess_kurtosis"].median()})
        for (_, end), chunk in blocks[blocks["family"] == family].groupby(["start_date_id", "end_date_id"], sort=True):
            family_time.append({"family": family, "start_date_id": int(chunk["start_date_id"].iloc[0]),
                "end_date_id": int(end), "rows": int(chunk["rows"].iloc[0]),
                "missing_fraction": float(chunk["missing_fraction"].mean()),
                "median_feature_std": chunk["std"].median()})
    result = {"feature_statistics": stats_df, "missingness_summary": missing_df, "missing_intervals": intervals,
        "discrete_value_counts": pd.DataFrame(discrete, columns=["feature", "family", "kind", "value", "count"]),
        "family_statistics": pd.DataFrame(family_rows), "family_time_blocks": pd.DataFrame(family_time),
        "feature_time_blocks": blocks, "distribution_drift": drift,
        "feature_rolling_statistics": trailing_statistics(frame, dates, settings)}
    result.update(xx_structure(frame, dates, settings["correlation_min_pairs"], settings["high_xx_correlation_threshold"]))
    return result


def target_eda(target: pd.Series, dates: pd.Series, settings: dict) -> dict:
    """Target-only retrospective description. No feature argument or join exists."""
    check_dates(dates, len(target))
    if target.name != "market_forward_excess_returns" or not target.index.equals(dates.index):
        raise ValueError("Expected the aligned development forecasting target only.")
    rolling = target.rolling(settings["rolling_window_rows"], min_periods=settings["rolling_min_observations"])
    positions = sorted(set(range(0, len(target), settings["rolling_output_stride"])) | {len(target) - 1})
    trace = pd.DataFrame({"date_id": dates.iloc[positions].to_numpy(),
        "rolling_mean": rolling.mean().iloc[positions].to_numpy(),
        "rolling_std": rolling.std(ddof=1).iloc[positions].to_numpy()})
    autocorr = pd.DataFrame([{"lag_rows": lag, "autocorrelation": target.autocorr(lag),
        "paired_observations": int((target.notna() & target.shift(lag).notna()).sum())}
        for lag in range(1, settings["target_autocorrelation_max_lag"] + 1)])
    extreme_positions = sorted(set(target.nsmallest(5).index) | set(target.nlargest(5).index))
    extremes = pd.DataFrame({"date_id": dates.loc[extreme_positions].to_numpy(),
                             "target_value": target.loc[extreme_positions].to_numpy()})
    clean = target.dropna().to_numpy()
    counts, edges = np.histogram(clean, bins=80)
    histogram = pd.DataFrame({"left": edges[:-1], "right": edges[1:], "count": counts})
    return {"summary": {"target": target.name, "analysis_start": int(dates.min()),
        "analysis_end": int(dates.max()), "rows": len(target), **descriptive(target)},
        "tables": {"target_rolling_statistics": trace, "target_autocorrelation": autocorr,
                   "target_extreme_observations": extremes, "target_histogram": histogram}}
