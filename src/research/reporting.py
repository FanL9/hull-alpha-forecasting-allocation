"""Generate evidence-based prose from the new data-only EDA result."""
import numpy as np
import pandas as pd


def strict_value(value):
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [strict_value(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    return strict_value(frame.to_dict(orient="records"))


def md_table(frame: pd.DataFrame) -> str:
    def cell(value):
        if pd.isna(value):
            return "undefined"
        if isinstance(value, (float, np.floating)):
            return f"{value:.6g}"
        return str(value).replace("|", "\\|")
    return "\n".join(["| " + " | ".join(map(str, frame.columns)) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |"] +
        ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)])


def render_report(summary: dict, tables: dict[str, pd.DataFrame], figure_names: list[str]) -> str:
    q, m, target = summary["development_quality"], summary["feature_missingness"], summary["target_only"]
    leading = tables["missingness_summary"].sort_values("missing_fraction", ascending=False).head(10)
    unusual = tables["feature_statistics"].assign(abs_skew=lambda f: f["skewness"].abs()).sort_values("abs_skew", ascending=False).head(8)
    drift = tables["distribution_drift"]
    usable = drift[drift["sufficient_observations"]]
    family_time = tables["family_time_blocks"]
    binary = tables["discrete_value_counts"]
    sets = binary.groupby(["feature", "kind"], sort=False)["value"].apply(lambda s: ", ".join(map(str, s))).reset_index()
    lines = ["# Phase 2 — Data Processing & EDA", "",
        f"Generated (UTC): {summary['generated_at_utc']}. Source: existing hash-verified official Hull Tactical files only.", "",
        "The superseded Phase 2 alpha/model results and split were withdrawn. This report describes only the rebuilt descriptive phase.", "",
        "## Boundary and provenance", "",
        "| Area | date_id | Rows | Columns | Permitted access |", "| --- | --- | --- | --- | --- |",
        "| Full train | 0–9047 | 9048 | 98 | Hashes, headers, IDs |",
        "| Development | 0–8867 | 8868 | 98 | Data processing and EDA |",
        "| Competition Simulation | 8868–9047 | 180 | 98 | Locked; headers, IDs, counts only |", "",
        "No gap or overlap exists. Simulation feature/target values were not parsed. Schema compatibility means a shared header in train.csv; no simulation dtype inference, duplicate-value check or missingness statistics were performed.", "",
        "Official test.csv is inference-schema/API material only. Its 10 IDs (8980–8989) overlap the locked period. Only its header and IDs were inspected; no feature, lagged-label or target statistics were computed.", "",
        "The original Phase 1 full-file audit and dictionary predate this firewall and remain byte-identical historical evidence. They are not inputs to new processing choices. This rebuild cannot erase earlier inspection or make these periods historically unseen.", "",
        f"All {summary['provenance']['official_file_count']} manifest-listed official files and the ZIP passed integrity verification. Train SHA-256: `{summary['provenance']['source_train_sha256']}`.", "",
        "## Development schema and quality", "",
        f"All 94 anonymous features, date_id and all three original outcomes are retained. Development shape: {q['rows']} × {q['columns']}; duplicate full rows: {q['duplicate_rows']}; duplicate IDs: {q['duplicate_date_ids']}; chronological: {q['chronological_order']}; infinite cells: {q['infinite_cells']}.", "",
        "Outcomes are forward_returns, risk_free_rate and market_forward_excess_returns. No target shifting, imputation, winsorization, outlier deletion or feature filtering was performed. Dtypes for all 98 development columns are recorded in phase2_data_summary.json.", "",
        "## Missingness", "",
        f"{m['features_with_missing']} of 94 features contain missing values: {m['missing_cells']:,} cells ({m['missing_fraction']:.2%} of development feature cells). All-missing features: {m['all_missing_features']}; constant observed features: {m['constant_features']}.", "",
        "Missingness summary includes every feature's count/fraction, first/last observation, leading/trailing runs and longest contiguous missing run. missing_intervals.csv records every contiguous interval; feature_time_blocks.csv and family_time_blocks.csv record chronological changes.", "",
        md_table(leading[["feature", "missing_count", "missing_fraction", "leading_missing_rows", "longest_missing_run"]]), "",
        "## Family scale and distributions", "",
        "Only the observed M, E, I, P, V, S and D families are used; no MOM family exists. Family summaries are medians across member-level statistics, not pooled standardized scores. Units vary; family scale differences have no inferred economic meaning.", "",
        md_table(tables["family_statistics"]), "",
        "All 94 feature_statistics.csv rows include mean, sample standard deviation, median, 1/5/25/75/95/99% quantiles, min/max, sample skewness and excess kurtosis. The following largest absolute-skew examples are descriptive display choices, not candidate features or a predictive ranking. No observations are removed.", "",
        md_table(unusual[["feature", "observed", "min", "max", "skewness", "excess_kurtosis"]]), "",
        "## Discrete encodings", "", md_table(sets), "",
        "D6's two-state -1/0 encoding is preserved. No numbered anonymous field is assigned a specific economic interpretation.", "",
        "## Temporal distributions", "",
        "Rolling means/std use trailing 252 rows including the displayed row and require 126 observed values; they are sampled every 21 rows for display. Target rolling summaries describe realized labels retrospectively and are not forecast-time features. No centered, backward-looking-from-the-future or full-sample normalizer is applied.", "",
        "Drift compares each chronological block of up to 252 rows with its immediately preceding block, using only information available at that block end. It reports empirical CDF distance, mean change in prior standard deviations and scale ratio; each block needs at least 20 observations. Missing/constant histories yield undefined statistics. These are descriptive distribution changes, not formal structural breaks, regimes or causal effects.", "",
        f"Among {len(usable):,} sufficiently observed adjacent feature-block pairs, median ECDF distance is {usable['ecdf_distance'].median():.4f} and the 95th percentile is {usable['ecdf_distance'].quantile(.95):.4f}. The final block has {int(family_time['rows'].iloc[-1])} rows; shorter-block noise and missingness affect comparisons.", "",
        "## X–X correlation structure", "",
        f"Pairwise-complete Pearson correlations use development X only and require {summary['settings']['correlation_min_pairs']} paired observations. {summary['xx_structure']['high_correlation_pairs']} distinct pairs have |correlation| ≥ {summary['settings']['high_xx_correlation_threshold']:.2f}; no column is dropped. Every pair's observation count is retained. Family summaries use absolute off-diagonal correlations, including within-family diagonals that exclude self-pairs.", "",
        "Pairwise deletion can produce a non-positive-semidefinite matrix. Overlapping coverage, common trends, nonstationarity and dependent rows limit interpretation; no predictive or causal conclusion follows.", "",
        "## Target-only EDA", "",
        "No feature appears in this analysis. Target statistics use only development market_forward_excess_returns in its supplied units and original row alignment; the label is not reconstructed as the difference of other outcomes.", "",
        md_table(pd.DataFrame([{k: target[k] for k in ["rows", "missing_count", "mean", "median", "std", "min", "max", "skewness", "excess_kurtosis"]}])), "",
        md_table(pd.DataFrame([{k: target[k] for k in ["q01", "q05", "q25", "q75", "q95", "q99"]}])), "",
        "The five smallest and five largest development observations are retained in the ignored target_extreme_observations.csv. Target autocorrelation at lags 1–20 is descriptive, with pair counts and no hypothesis tests. No target value is joined to any feature.", "",
        "## Processing framework", "",
        "Default processed development.csv preserves all 8,868 rows, 98 columns, original values, encodings and NaNs. This is an identity export, not fitted preprocessing. Missingness is investigated before choosing a treatment.", "",
        "Future candidates: retain NaN; a median fitted inside each historical training window and used only on subsequent observations; expanding strictly prior median; contemporaneous missing indicators. Optional past-only median/indicator helpers are software-tested but are not applied to the default export. Leading gaps stay NaN when no past value exists. No global mean/median, backward fill or full-development statistic is used to transform historical rows.", "",
        "## Future validation — design only", "",
        "Phase 2.5 fixes future research at date_id 0–7859: 5087 initial training rows, 11 validation blocks of 252, step 252, purge 1 and embargo 0. Internal Holdout is 7860–8867 (1008 rows); the final 180-row simulation stays locked. Models remain unselected. See docs/research_protocol.md and configs/validation.yaml. This descriptive EDA does not execute alpha/model validation, signals, allocation or backtests.", "",
        "## Figures", ""]
    for name in figure_names:
        lines += [f"![{name.removeprefix('phase2_eda_').removesuffix('.png').replace('_', ' ')}](figures/{name})", ""]
    lines += ["## Reproduction and artifacts", "",
        "Run `python -m src.cli phase1`, `python -m src.cli phase2` and `python -m pytest`. The protected Phase 1 command verifies existing files and preserves its historical audit; it cannot auto-download while the split policy is active. Notebook 02 executes the same data-only pipeline from raw files, writes all artifacts and displays actual aggregate results.", "",
        "Generated artifacts include split_definition.json, phase2_data_summary.json, missingness_summary.csv, feature_statistics.csv, family_statistics.csv, missing_intervals.csv, discrete_value_counts.csv, feature/family time blocks, trailing statistics, descriptive drift, X–X correlations/pair counts and target-only tables. Raw, processed and observation-level artifacts remain ignored.", "",
        "No alpha discovery, predictive feature selection, model fitting, allocation optimization, or backtesting has been performed in the rebuilt Phase 2.", ""]
    return "\n".join(lines)
