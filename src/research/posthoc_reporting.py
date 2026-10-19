"""Plots and descriptive prose from post-hoc diagnostic summary tables only."""
from pathlib import Path

import numpy as np
import pandas as pd


def _table(frame, columns, formats=None):
    formats = formats or {}
    lines = ["| " + " | ".join(label.replace("|", "\\|") for _, label in columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.to_dict("records"):
        values = []
        for key, _ in columns:
            value = row[key]
            if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                values.append("undefined")
            else:
                values.append(format(value, formats.get(key, ".5f")) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(value.replace("|", "\\|") for value in values) + " |")
    return lines


def make_figures(tables: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir()
    lag = tables["lag_decay"]
    names = lag["feature"].drop_duplicates().tolist()
    fig, axes = plt.subplots(6, 4, figsize=(16, 19), sharex=True)
    for ax, name in zip(axes.flat, names, strict=True):
        rows = lag[lag["feature"].eq(name)]
        ax.plot(rows["lag_k"], rows["rank_ic"], marker="o", linewidth=1.5)
        ax.axhline(0, color="gray", linewidth=.6)
        ax.set_title(name.replace("lagged_market_forward_excess_returns", "lagged_excess_return").replace("lagged_forward_returns", "lagged_forward_return"), fontsize=9)
        ax.set_xticks(range(6))
        ax.set_xlabel("Past feature lag k", fontsize=8)
        ax.set_ylabel("Rank IC", fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Exploratory / post-hoc diagnostic: single-feature lag decay", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .98))
    fig.savefig(figure_dir / "single_feature_lag_decay.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    for name, rows in tables["composite_lag_decay"].groupby("feature", sort=False):
        ax.plot(rows["lag_k"], rows["rank_ic"], marker="o", label=name)
    ax.axhline(0, color="gray", linewidth=.7)
    ax.set(xlabel="Past composite lag k", ylabel="Block-wise Rank IC",
           title="Exploratory / post-hoc diagnostic: frozen composite lag decay")
    ax.legend(title="Composite")
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figure_dir / "composite_lag_decay.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for column, feature in enumerate(["M9", "M11", "E19"]):
        for row, scope in enumerate(["research_prefix_0_7859", "phase3_eligible_oos"]):
            ax = axes[row, column]
            groups = tables["decile_analysis"].query("feature == @feature and scope == @scope")
            ax.plot(groups["decile"], groups["mean_y"], marker="o", label="Mean target")
            ax.plot(groups["decile"], groups["median_y"], marker="s", label="Median target")
            ax.axhline(0, color="gray", linewidth=.6)
            ax.set(title=f"{feature}: {scope}", xlabel="Pooled X decile", ylabel="Future target")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
    fig.suptitle("Exploratory / post-hoc diagnostic: pooled decile response", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .96))
    fig.savefig(figure_dir / "focus_decile_response.png", dpi=150)
    plt.close(fig)

    matrix = tables["spearman_correlation_matrix"].set_index("feature").drop(columns="diagnostic_type")
    names = matrix.index.tolist()
    linkage = tables["hierarchical_linkage"]
    children = {int(r.node): (int(r.left), int(r.right)) for r in linkage.itertuples()}
    def leaves(node):
        return [node] if node < len(names) else leaves(children[node][0]) + leaves(children[node][1])
    order = leaves(int(linkage["node"].iloc[-1]))
    labels = [names[i] for i in order]
    fig, ax = plt.subplots(figsize=(14, 12))
    image = ax.imshow(matrix.loc[labels, labels], vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(range(len(names)), labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(names)), labels, fontsize=8)
    ax.set_title("Exploratory / post-hoc diagnostic: OOS Spearman redundancy")
    fig.colorbar(image, ax=ax, label="Spearman correlation", shrink=.8)
    fig.tight_layout()
    fig.savefig(figure_dir / "spearman_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    position = {node: (i, 0.) for i, node in enumerate(order)}
    fig, ax = plt.subplots(figsize=(15, 7))
    for row in linkage.itertuples():
        xa, ya = position[int(row.left)]
        xb, yb = position[int(row.right)]
        ax.plot([xa, xa, xb, xb], [ya, row.distance, row.distance, yb], color="tab:blue", linewidth=1)
        position[int(row.node)] = ((xa + xb) / 2, row.distance)
    ax.axhline(.2, color="tab:red", linestyle="--", label="Exploratory cut = 0.2")
    ax.set_xticks(range(len(names)), labels, rotation=75, ha="right", fontsize=8)
    ax.set(ylabel="Average linkage distance: 1 - |Spearman|",
           title="Exploratory / post-hoc diagnostic: correlation hierarchy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figure_dir / "correlation_dendrogram.png", dpi=150)
    plt.close(fig)


def write_diagnostic_report(tables: dict, source_manifest: dict, output: Path, report: Path) -> str:
    make_figures(tables, output)
    lag = tables["lag_decay_summary"]
    composite = tables["composite_lag_decay_summary"]
    reverse = tables["reverse_return_ic"]
    divergence = tables["rank_pearson_divergence"].set_index("feature")
    pairs = tables["high_correlation_pairs"].sort_values("spearman_corr", key=lambda s: s.abs(), ascending=False)
    clusters = tables["cluster_summary"]
    sharp = lag[lag["relative_drop_k0_to_k1"].ge(.75)].sort_values("relative_drop_k0_to_k1", ascending=False)
    lagged_names = ["lagged_forward_returns", "lagged_market_forward_excess_returns"]
    sharp_anonymous = sharp[~sharp["feature"].isin(lagged_names)]
    reversals = lag[lag["sign_flip_k0_to_k1"]]
    other = lag[~lag["feature"].isin(list(sharp["feature"]) + list(reversals["feature"]))]
    return_max = reverse.loc[reverse.groupby("feature")["rank_ic"].apply(lambda s: s.abs().idxmax())].sort_values("rank_ic", key=lambda s: s.abs(), ascending=False)
    link = "phase3_posthoc_diagnostics/"
    lines = ["# Phase 3 Post-hoc Diagnostic Analysis", "", "## Purpose", "",
        "**This analysis is post-hoc and does not alter the pre-registered Phase 3 conclusions.**",
        "All tables and figures are exploratory / post-hoc diagnostics. No new p-values, BH/BY tests, formal grades, model features or trading decisions are produced.",
        f"The diagnostic objects are the sealed 24 single-feature PASS and 4 composite PASS from `{source_manifest['git_commit']}` / `{source_manifest['git_tag']}`; their formal labels remain unchanged.", "",
        "Value-bearing input is restricted to date_id 0–7859 via the existing safe research-prefix loader. Primary diagnostic IC uses the original eligible OOS folds (5088–7859), frozen complete-pair block ranks and original time positions. Shifts occur before slicing; no gap compression, refitted availability or validation direction choice is used.",
        "Composites replay saved members/directions/required counts and hash-verified training percentile references. Lagging C_t uses the historical fold's actual composite; the first k OOS rows are missing instead of inventing pre-OOS composites. k=0 Rank IC and pair counts reconcile with all 28 sealed PASS; saved-bound Pearson also reconciles for all 24 singles.", "",
        "## 1. Lag Decay", "",
        "Magnitude drop = |IC_k0| − |IC_k1|; relative drop divides by |IC_k0|. A ≥75% magnitude loss is an **exploratory display flag only**, chosen for this diagnostic and never a significance, exclusion or PASS rule. Negative relative drops indicate stronger lagged associations. Common-pair estimates separate coverage changes from lag changes.", "",
        f"The display flag is present for {len(sharp)} of 24 signals: " + (", ".join(sharp["feature"]) if len(sharp) else "none") + ". This flags a lag pattern for interpretation, not evidence of leakage. In particular, a legal t−1 return field can lose its association when moved to t−2; its exact construction audit takes precedence over a generic cliff interpretation.", "",
        "Other signals retain more k=1 magnitude, but retention is not proof of correct intraday availability. The complete signed trajectories below show whether later lags decay, reverse or rebound; smoothness is not assigned a formal label.", ""]
    columns = [("feature", "Signal")] + [(f"IC_k{k}", f"IC k={k}") for k in range(6)] + [("relative_drop_k0_to_k1", "Relative magnitude drop")]
    lines += _table(lag, columns)
    lines += ["", "Signed k=0→1 reversals: " + (", ".join(reversals["feature"]) if len(reversals) else "none") + ". Magnitude retention alone can hide these reversals; they require caution even without the 75% display flag."]
    lines += ["", "Composite trajectories (identical exploratory flag, no FDR rerun):", ""] + _table(composite, columns)
    lines += ["", f"![Single lag decay]({link}figures/single_feature_lag_decay.png)",
              f"![Composite lag decay]({link}figures/composite_lag_decay.png)", "",
              f"[Long lag table]({link}lag_decay.csv), [common-pair diagnostics]({link}lag_decay_summary.csv), [k=0 reconciliation]({link}k0_reconciliation.csv).", "",
        "## 2. Reverse Return Relationship", "",
        "For each signal, the following row reports its largest absolute past-return IC among j=1..5. Choosing that row is descriptive reporting, not a new hypothesis or feature selection.", ""]
    lines += _table(return_max, [("feature", "Signal"), ("past_return_lag", "Past-return lag j"), ("rank_ic", "Rank IC"), ("n_obs", "Pairs")])
    lines += ["", "The two lagged fields require a separate interpretation: lagged_market_forward_excess_returns is exactly y_(t−1), so reverse j=1 Rank IC=1 is mechanical. lagged_forward_returns is the supplied raw forward return from t−1; its strong relation to the adjusted past target is also expected. Both were reconstructed with an exact one-row backward shift, with first-row NaN and source date offset −1. This is past information, not evidence of a future shift.",
        f"[Exact lag timing audit]({link}lagged_field_timing_audit.csv). Anonymous features can encode past market performance; stronger reverse IC alone does not establish future-label contamination.", "",
        "## 3. Rank/Pearson Divergence", "",
        "Deciles are pooled X-only empirical quantile groups in two scopes: the full research prefix and the original eligible OOS subset. They are retrospective descriptive groups, not trained trading thresholds. Tied quantile edges are dropped instead of splitting equal values arbitrarily.",
        "The following additive contributions use the frozen-bound winsorized X, paired centering and the full-sample Pearson denominator. Core + tail contribution equals the saved Pearson IC. The X-tail display uses pooled 1%/99% raw-X quantiles; the target-extreme display uses the largest 1% absolute centered y. Neither diagnostic deletes observations from formal results or winsorizes y.", ""]
    focus = divergence.loc[["M9", "M11", "E19"]].reset_index()
    lines += _table(focus, [("feature", "Signal"), ("frozen_oos_rank_ic", "Frozen block Rank IC"),
        ("oos_global_spearman_ic", "Pooled OOS Rank IC"), ("frozen_winsorized_pearson_ic", "Frozen winsor Pearson"),
        ("x_core_winsor_corr_contribution", "X-core contribution"), ("x_tail_winsor_corr_contribution", "X-tail contribution"),
        ("y_core_winsor_corr_contribution", "Y-core contribution"), ("y_extreme_winsor_corr_contribution", "Y-extreme contribution")])
    for name in ["M9", "M11", "E19"]:
        row = divergence.loc[name]
        groups = tables["decile_analysis"].query("feature == @name and scope == 'phase3_eligible_oos'").sort_values("decile")
        first, last = groups.iloc[0], groups.iloc[-1]
        oppose_x = row["x_core_winsor_corr_contribution"] * row["x_tail_winsor_corr_contribution"] < 0
        oppose_y = row["y_core_winsor_corr_contribution"] * row["y_extreme_winsor_corr_contribution"] < 0
        lines += ["", f"**{name}:** OOS lowest/highest-decile mean y = {first['mean_y']:.6f}/{last['mean_y']:.6f}; median y = {first['median_y']:.6f}/{last['median_y']:.6f}. "
            f"Raw-X tail/core contributions {'oppose' if oppose_x else 'have the same sign'}; extreme-y/core contributions {'oppose' if oppose_y else 'have the same sign'}. "
            f"Descriptive Pearson excluding the X-tail is {row['pearson_excluding_x_tail_descriptive']:.5f}, and excluding extreme-y days is {row['pearson_excluding_extreme_y_descriptive']:.5f}. "
            "The numbers and decile curve are consistent with nonlinear, tail-sensitive and/or time-varying aggregation; they do not uniquely identify a causal mechanism."]
    lines += ["", "All 24 PASS signals, including other pooled-versus-block divergences (no new selection or gate):", ""]
    lines += _table(divergence.reset_index(), [("feature", "Signal"), ("frozen_oos_rank_ic", "Frozen block Rank IC"),
        ("oos_global_spearman_ic", "Pooled OOS Rank IC"), ("oos_raw_pearson_ic", "Raw Pearson"),
        ("frozen_winsorized_pearson_ic", "Frozen winsor Pearson")])
    lines += ["", f"![Focus deciles]({link}figures/focus_decile_response.png)", "",
        f"All 24 signals: [decile table]({link}decile_analysis.csv), [Rank/Pearson and tail contributions]({link}rank_pearson_divergence.csv), [frozen fold diagnostics]({link}fold_level_diagnostics.csv).", "",
        "## 4. Redundancy", "",
        f"Average linkage on 1−|pooled OOS Spearman| gives **{len(clusters)} exploratory clusters** at cut distance 0.2. Negative orientation is treated as redundant information. This is a descriptive grouping, not a count of statistically independent Alphas.",
        "Only pairwise complete observations in each feature's frozen eligible OOS folds are used. Pair counts are saved; a supplementary block-wise correlation matrix removes pooled block-level shifts. Undefined pair distances, if any, are set to 1 for clustering.", ""]
    lines += _table(clusters, [("cluster", "Cluster"), ("member_count", "Count"), ("members", "Members"),
                              ("mean_abs_within_cluster_correlation", "Mean |within correlation|")])
    lines += ["", "Exploratory cut sensitivity:", ""] + _table(tables["cluster_cut_sensitivity"], [("cut_distance", "Distance cut"), ("cluster_count", "Clusters")])
    lines += ["", "All high-correlation pairs (|Spearman|≥0.8; display threshold only):", ""]
    lines += _table(pairs, [("feature_a", "Feature A"), ("feature_b", "Feature B"), ("pearson_corr", "Pearson"),
                           ("spearman_corr", "Spearman"), ("blockwise_spearman_corr", "Block-wise Spearman"), ("n_obs", "Pairs")])
    lines += ["", f"![Correlation heatmap]({link}figures/spearman_correlation_heatmap.png)",
              f"![Hierarchy]({link}figures/correlation_dendrogram.png)", "",
        "## 5. Overall Diagnostic Assessment", "",
        "These are overlapping diagnostic labels only; none changes PASS/FAIL/NOT_TESTABLE:", "",
        "- **Potential timing concern:** " + (", ".join(sharp_anonymous["feature"]) if len(sharp_anonymous) else "no anonymous feature meets the exploratory 75% k0→1 magnitude-loss flag") + ". Signed k0→1 reversals occur in " + (", ".join(reversals["feature"]) if len(reversals) else "none") + ". Such lag sensitivity requires caution, not an automatic leakage conclusion. The legal lagged fields' cliff is interpreted separately as past-return dependence; their exact t−1 audit found no future shift. Anonymous publication cutoffs remain unverified.",
        "- **No obvious timing concern:** " + (", ".join(other["feature"]) if len(other) else "none under the display rule") + ". This means only that the chosen lag-cliff flag is absent; it is not a certification of executable availability. The two lagged outcomes additionally pass the exact t−1 construction audit.",
        "- **Strong redundancy:** " + (", ".join(clusters.loc[clusters["member_count"].gt(1), "members"]) if clusters["member_count"].gt(1).any() else "no multi-member cluster at the exploratory cut") + ". No representative feature has been selected.",
        "- **Unusual nonlinear structure:** M9, M11 and E19 show the sealed Rank/Pearson divergence discussed above. Deciles, tails and pooled-versus-block ranks describe possible mechanisms; none creates a new formal grade.", "",
        "No Phase 4 model, feature selection, allocation, backtest, Kaggle score, Internal Holdout or Final-180 evaluation was performed. Formal report/artifact hashes were checked before and after diagnostics.", "",
        "This post-hoc diagnostic analysis does not alter the frozen Phase 3 PASS/FAIL/NOT_TESTABLE conclusions.",
        "No Internal Holdout or Final-180 data were used.", ""]
    return "\n".join(lines)
