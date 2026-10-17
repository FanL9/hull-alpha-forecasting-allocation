"""Static scientific figures from development-only descriptive tables."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_figures(tables: dict[str, pd.DataFrame], directory: Path) -> list[str]:
    directory.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    names = []

    def save(fig, name):
        filename = f"phase2_eda_{name}.png"
        fig.savefig(directory / filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
        names.append(filename)

    family_time = tables["family_time_blocks"]
    matrix = family_time.pivot(index="family", columns="end_date_id", values="missing_fraction")
    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
    ax.set_yticks(range(len(matrix)), matrix.index)
    ticks = list(range(0, len(matrix.columns), 5))
    ax.set_xticks(ticks, [str(matrix.columns[i]) for i in ticks])
    ax.set(xlabel="Chronological block end date_id (252 rows; final block shorter)",
           title="Development feature missingness by family | date_id 0–8867")
    fig.colorbar(im, ax=ax, label="Missing fraction")
    save(fig, "missingness")

    corr = tables["xx_correlation"]
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    groups = [c[0] for c in corr.columns]
    centers, labels = [], []
    for group in dict.fromkeys(groups):
        positions = [i for i, value in enumerate(groups) if value == group]
        # Labels follow the CSV's order; no feature ranking is applied.
        centers.append(float(np.mean(positions)))
        labels.append(group)
    ax.set_xticks(centers, labels)
    ax.set_yticks(centers, labels)
    ax.set(title="Development X–X Pearson correlation | pairwise complete observations",
           xlabel="94 features by family in original CSV order", ylabel="94 features by family in original CSV order")
    fig.colorbar(im, ax=ax, label="Pearson correlation (X–X only)", fraction=.046)
    save(fig, "xx_correlation")

    families = list(family_time["family"].drop_duplicates())
    family_matrix = pd.DataFrame(np.nan, index=families, columns=families)
    for row in tables["family_xx_correlation"].itertuples():
        family_matrix.loc[row.family_a, row.family_b] = row.median_abs_pearson_xx
        family_matrix.loc[row.family_b, row.family_a] = row.median_abs_pearson_xx
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(family_matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(len(families)), families)
    ax.set_yticks(range(len(families)), families)
    for i in range(len(families)):
        for j in range(len(families)):
            v = family_matrix.iloc[i, j]
            ax.text(j, i, f"{v:.2f}" if pd.notna(v) else "—", ha="center", va="center",
                    color="white" if v > .65 else "black")
    ax.set_title("Median absolute X–X correlation by family\nWithin-family diagonal excludes self-correlations")
    fig.colorbar(im, ax=ax, label="Median |Pearson X–X|", fraction=.046)
    save(fig, "family_xx")

    rolling = tables["feature_rolling_statistics"]
    representative = ["M1", "E1", "I1", "P1", "V1", "S1", "D1"]
    available = [c for c in representative if c in set(rolling["feature"])]
    if not available:
        available = list(rolling["feature"].drop_duplicates()[:4])
    fig, axes = plt.subplots(len(available), 2, figsize=(12, 2.1 * len(available)), sharex=True, squeeze=False)
    for i, feature in enumerate(available):
        subset = rolling[rolling["feature"] == feature]
        for j, statistic in enumerate(["rolling_mean", "rolling_std"]):
            axes[i, j].plot(subset["date_id"], subset[statistic], linewidth=1)
            axes[i, j].set_ylabel(feature)
            if i == 0:
                axes[i, j].set_title(statistic.replace("_", " ").title())
    for ax in axes[-1]:
        ax.set_xlabel("date_id (development only)")
    fig.suptitle("Representative features: trailing 252 rows, ≥126 observations\nFirst numbered feature per family; display choice uses no target", y=1.01)
    fig.tight_layout()
    save(fig, "feature_time")

    drift = tables["distribution_drift"]
    fig, ax = plt.subplots(figsize=(11, 4))
    for family, subset in drift.groupby("family", sort=False):
        values = subset.groupby("end_date_id")["ecdf_distance"].median()
        ax.plot(values.index, values, label=family, linewidth=1)
    ax.set(title="Adjacent chronological blocks: median empirical CDF distance by family",
           xlabel="Current block end date_id", ylabel="ECDF distance (0–1)", ylim=(0, 1))
    ax.legend(ncols=7, loc="upper right")
    save(fig, "distribution_drift")

    histogram, target_rolling, acf = (tables[k] for k in
        ["target_histogram", "target_rolling_statistics", "target_autocorrelation"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0, 0].bar(histogram["left"], histogram["count"], width=histogram["right"] - histogram["left"], align="edge")
    axes[0, 0].set(title="Target distribution (unmodified)", xlabel="market_forward_excess_returns", ylabel="Observations")
    axes[0, 1].bar(acf["lag_rows"], acf["autocorrelation"])
    axes[0, 1].axhline(0, color="black", linewidth=.5)
    axes[0, 1].set(title="Target autocorrelation (descriptive, no significance bands)", xlabel="Lag in rows", ylabel="Autocorrelation")
    for ax, statistic in zip(axes[1], ["rolling_mean", "rolling_std"], strict=True):
        ax.plot(target_rolling["date_id"], target_rolling[statistic], linewidth=1)
        ax.set(title=statistic.replace("_", " ").title() + " (trailing 252 rows)", xlabel="date_id", ylabel="Target units")
    fig.suptitle("Target-only retrospective EDA | development date_id 0–8867 | no link to X")
    fig.tight_layout()
    save(fig, "target_only")
    return names
