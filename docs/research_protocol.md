# Research Protocol

Research rules: **phase2.5-v2**; **frozen**. Phase 3 Alpha Research is **not started**.
Executable authority: [validation configuration](../configs/validation.yaml), [research configuration](../configs/research.yaml), [split policy](../configs/split_policy.json) and their implementations.

## Version history

**Phase 2.5 v2 was defined before any real feature–target Alpha Research was executed.**
v2 supersedes phase2.5-v1 only in its statistical rules: primary and secondary metrics, inference roles, a second pre-registered hypothesis family and a global sensitivity analysis.
Its reasons come only from properties of X already described in Phase 2 EDA: heavy-tailed feature distributions, temporal distribution drift, and missingness/availability structure.
No real IC, predictive result, feature ranking, model performance or backtest result informed v2.
Boundaries, folds, purge/embargo, availability thresholds, HAC lag, baselines and the official metric are unchanged from v1.
The v1 locks remain byte-for-byte in configs/protocol_lock_phase2.5-v1.json (original format) and configs/protocol_lock_phase2.5-v1_lockformat2.json (lock-format migration).

## Data boundaries

| Area | Inclusive date_id | Rows | Permitted use |
| --- | --- | --- | --- |
| Research / Walk-Forward | 0–7859 | 7860 | Training, validation and selection |
| Internal Holdout | 7860–8867 | 1008 | One frozen-pipeline evaluation; no tuning |
| Competition Simulation | 8868–9047 | 180 | Locked until final sequential simulation |

Research and Internal Holdout together form Development 0–8867; the outer partition has no gap or overlap.
Official test.csv is interface/schema-only and never a research evaluation set.
The competition lock permits only identifiers, row counts and header compatibility before simulation authorization.
Holdout reservations control subsequent predictive selection; the development history has descriptive EDA coverage and is not described as historically unseen.

## Target, timing and candidates

Primary target: **market_forward_excess_returns**, kept on its supplied row without reconstruction or a second forward shift.
Its official adjustment motivates stability research, not a claim that alpha exists; definitions and availability are centralized in the [data dictionary](data_dictionary.md).
The prediction layer evaluates forecasts; the portfolio layer evaluates allocations and realized returns with the [official metric](official_metric_notes.md), not target IC.

The candidate universe has **94 anonymous inputs plus 3 competition-available lagged outcomes**: lagged_forward_returns, lagged_market_forward_excess_returns and lagged_risk_free_rate.
Reconstruct these one-row lags from consecutive historical IDs, initializing validation only with earlier observed history; do not import mock test values.
Exclude current outcomes, date_id and is_scored from predictors.

## Expanding-window validation

| Parameter | Value |
| --- | --- |
| Initial training size | 5087 |
| Validation window | 252 |
| Step size | 252 |
| Folds | 11 |
| Purge | 1 row |
| Embargo | 0 rows |

Never shuffle, silently reorder identifiers or overlap validation intervals.
Fold 1 trains 0–5086, purges 5087 and validates 5088–5339; fold 11 trains 0–7606, purges 7607 and validates 7608–7859.
Training expands and transforms/models stay frozen within each validation block; earlier validation labels may enter later training once historical and outside that fold's purge.

The one-row purge makes training outcome endpoints strictly earlier than the validation decision boundary for a one-day label.
A forward-only expanding split needs no extra post-validation embargo; supported nonzero embargo excludes post-validation rows from later training and validation, with a matching step size.

Internal Holdout is four 252-observation blocks, longer than the final simulation while retaining sufficient research history.
The initial history absorbs a 47-row remainder so every validation block is full-sized; neither choice is based on predictive performance or a statistical-power guarantee.
For a holdout refit, use only 0–7858, purge 7859 and freeze throughout 7860–8867; no automatic holdout or competition unlock is provided.

## Training-only feature availability

Every candidate must meet all conditions within the current training window:

- At least **252 nonmissing observations** and **2 distinct observed values**.
- At least **80% coverage from its first observed training row to the endpoint**.
- At most **25% missingness over the trailing 504 training rows**, or all training rows when shorter.

Post-launch coverage distinguishes structural leading gaps from intermittent holes, and recent coverage prevents stale inputs entering.
Record eligibility reasons per feature/fold without filling source values or using validation, holdout or Y to set thresholds.
Late-appearing inputs may have too few eligible folds for a discovery claim; do not relax the rules to rescue them.

## Metrics

**Primary Alpha metric: block-wise Spearman Rank IC.**
Within each of the 11 validation blocks, rank X and y separately on complete (X, y) pairs, with average ranks for ties; NaNs get no rank and keep their time positions.
Divide ranks by (complete pairs in the block + 1) so blocks with different coverage share one scale.
Concatenate the ranked blocks chronologically and **never re-rank** them; no other fold enters a block's ranks.
Compute the primary IC and its inference on this concatenated rank series, so level drift across folds cannot enter the Rank IC.

**Secondary robustness metric: training-window 1%/99% winsorized Pearson IC.**
For each feature and fold, estimate the 1st and 99th percentiles of X from that fold's training window only, and clip both training and validation X with those bounds.
Never estimate bounds from validation, pooled folds or future data; never winsorize y again, because market_forward_excess_returns is already winsorized by the publisher.
This metric is not confirmatory: significance under winsorized Pearson without primary Spearman significance is **not** a discovery.

Report fold IC, pair counts, coverage, equal-fold mean IC, IC std (ddof=1), sign consistency and positive-fold ratio; unavailable folds stay explicit.
A fold counts only with **126 validation pairs** and **50% validation coverage**; a hypothesis needs **3 eligible folds**.

## Inference roles

- **Newey–West → significance.** The single primary two-sided p-value uses Bartlett HAC **lag 5** on the correlation influence function of the block-wise ranked series, with a normal approximation. It is approximate inference on the correlation of block-wise ranked series, not an exact finite-sample Spearman test. NaN gaps keep their time positions; undefined uncertainty cannot reject.
- **Paired moving-block bootstrap → confidence interval and stability only.** Block length **20**, **2000 repetitions**, **95%** percentile interval, seed **42**, applied jointly to the already block-wise ranked (X, y) series. It yields no p-value and no alternative discovery rule.

These methods are approximate under dependent, nonstationary observations and data-dependent missingness; no nominal interval guarantees stable alpha.
Implementation: blockwise_rank, blockwise_spearman_hac, blockwise_spearman_interval, fit_winsor_bounds and apply_winsor_bounds in src/statistics/robust_inference.py.

## Hypothesis families

**Family 1 — 97 single features (primary).** 94 anonymous inputs plus the 3 lagged fields.
A candidate failing fold availability or the 3-eligible-fold rule is not_testable with p=1, so the denominator stays 97; late-starting inputs such as E7 are not rescued.

**Family 2 — 7 family composites (pre-registered).** One composite per M, E, I, P, V, S and D, rebuilt in every fold from training information only (src/features/family_composites.py):

1. **Members:** family features passing the availability rule in that fold's training window; eligibility in another fold does not count. The 3 lagged fields belong to no family.
2. **Direction:** sign of the member's training-window Spearman correlation with y; positive → +1, negative → −1, zero/undefined → member excluded. Qualified members are eligible members with a nonzero direction.
3. **Normalization:** rank-based, not z-score. Map each member through the mid-rank empirical CDF of its training values, p(v) = (#train < v + 0.5 × #train = v) / n_train; validation uses the same training map and never refits it. Multiply by the direction.
4. **Weighting:** equal-weight mean of oriented members observed in the row; no IC, p-value, volatility, regression or optimized weights.
5. **Missing members:** a row needs at least max(2, ceil(0.5 × qualified members)) observed members, otherwise NaN; fewer than 2 qualified members makes the fold composite NaN.
6. **Testing:** identical to Family 1 — block-wise Spearman, HAC lag 5, block-bootstrap interval, the same fold eligibility, not testable → p=1; the denominator stays 7.

The training percentile map that builds a composite is a different step from the block-wise validation ranking that tests it.
Interpretation of the D-family composite should be cautious because rank-based aggregation of discrete features may contain limited cross-sectional resolution; its test standard is unchanged.

## Multiple testing

| Analysis | Hypotheses | Method | Role |
| --- | --- | --- | --- |
| Family 1 | 97 single features | BH, FDR 0.05 | Primary, reported separately |
| Family 2 | 7 family composites | BH, FDR 0.05 | Primary, reported separately |
| Global sensitivity | 104 = 97 + 7 raw primary p-values | BH, FDR 0.05 | Sensitivity only; never replaces family results |
| Dependence sensitivity | each of the above | Benjamini–Yekutieli | Sensitivity only |

BH needs independence or suitable positive dependence; findings that fail BY are labeled dependence-sensitive.
Secondary metrics, family tables and repeated windows are not alternative discovery gates; new horizons or variants require a registered family before inspection.
No correction establishes persistence, causality or investability, and no stability threshold is retrofitted to results.
Implementation: adjust_fdr, family_pvalues and protocol_multiplicity in src/statistics/multiple_testing.py.
The tools only process caller-supplied arrays and are tested on synthetic data; no real-data Alpha pipeline exists yet.

## Baselines and selection

Forecast baseline: the arithmetic target mean in the fold's **purged training rows**, frozen throughout validation; zero forecast is a diagnostic comparison.
Portfolio baseline: **allocation=1.0**; **allocation=0.0** is a cash sanity comparison, with undefined zero-risk scores preserved as such.
Deflated Sharpe is reserved for later model/strategy selection involving multiple candidates, parameters or backtests, not single-feature screening.

Freeze preprocessing, selected inputs, model, timing and evaluation schedule before authorizing one Internal Holdout test; holdout outcomes do not guide redesign.
Freeze the complete strategy and update schedule before separately authorizing competition simulation; using either held-out interval for redesign removes its selection-independent interpretation.

## Integrity and maintenance

[Protocol lock](../configs/protocol_lock.json) (lock format 2, rules phase2.5-v2) binds configuration, critical executable code, reference tests, the archived v1 locks and expected data hashes; ordinary docs and reports are not hash-locked.
Run verify_protocol_lock with verify_data=True before research access; the default metadata-only check also supports a checkout without licensed data.
Scientific-rule changes require an explicitly versioned protocol before new outcome inspection; explanatory prose can be maintained independently.

References: [Benjamini–Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x), [Benjamini–Yekutieli](https://doi.org/10.1214/aos/1013699998), [Newey–West](https://doi.org/10.2307/1913610), [Künsch](https://doi.org/10.1214/aos/1176347265), [Deflated Sharpe](https://doi.org/10.3905/jpm.2014.40.5.094).
