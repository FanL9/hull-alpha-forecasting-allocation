# Research Protocol

Research rules: **phase2.5-v1**; **frozen**. Phase 3 Alpha Research is **not started**.
Executable authority: [validation configuration](../configs/validation.yaml), [research configuration](../configs/research.yaml), [split policy](../configs/split_policy.json) and their implementations.

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

## Statistical validation

- Report raw **Pearson IC** and secondary **Spearman Rank IC**, pair counts and coverage by fold; require **126 validation pairs**, **50% validation coverage** and **3 eligible folds** for a primary discovery test.
- Report equal-fold mean IC, sample IC std (ddof=1), sign consistency and positive-fold ratio over defined folds; unavailable folds stay explicit.
- Fix orientation from training Pearson sign only; a zero/undefined training association provides no oriented test for that fold, and raw/oriented results are reported separately.
- Form one primary two-sided p-value per feature from pooled chronological validation Pearson IC, with X centering/scaling and direction determined in each training window; preserve native timestamps, unavailable gaps and cross-boundary dependence.
- Use **Newey–West/Bartlett HAC lag 5** on the Pearson-correlation influence function with a normal approximation; undefined uncertainty cannot reject, and Rank-IC HAC is an approximate secondary diagnostic.
- Use **paired moving-block bootstrap** with block length **20**, **2000 repetitions**, **95%** percentile intervals and seed **42**; resample X/Y jointly on their original grid, preserve nonconsecutive gaps and record valid repetitions.

These uncertainty methods are approximate under dependent, nonstationary observations and data-dependent missingness; no nominal interval guarantees stable alpha.
The implementation provides statistical tools, not a real-data Alpha execution pipeline.

## Multiple testing

Apply **Benjamini–Hochberg FDR 0.05** to one prespecified family of **97 hypotheses**, one target and one horizon.
Assign p=1 to ineligible/unestimable candidates so they remain in the denominator; secondary Rank IC, family tables and repeated windows are not alternative discovery gates.
New horizons or variants require a registered testing family before inspection.
Report **Benjamini–Yekutieli** sensitivity because BH needs independence or suitable positive dependence; dependence-sensitive findings must be labeled accordingly.
No correction establishes persistence, causality or investability, and no stability threshold is retrofitted to results.

## Baselines and selection

Forecast baseline: the arithmetic target mean in the fold's **purged training rows**, frozen throughout validation; zero forecast is a diagnostic comparison.
Portfolio baseline: **allocation=1.0**; **allocation=0.0** is a cash sanity comparison, with undefined zero-risk scores preserved as such.
Deflated Sharpe is reserved for later model/strategy selection involving multiple candidates, parameters or backtests, not single-feature screening.

Freeze preprocessing, selected inputs, model, timing and evaluation schedule before authorizing one Internal Holdout test; holdout outcomes do not guide redesign.
Freeze the complete strategy and update schedule before separately authorizing competition simulation; using either held-out interval for redesign removes its selection-independent interpretation.

## Integrity and maintenance

[Protocol lock v2](../configs/protocol_lock.json) binds configuration, critical executable code, reference tests and expected data hashes; ordinary docs and reports are not hash-locked.
Run verify_protocol_lock with verify_data=True before research access; the default metadata-only check also supports a checkout without licensed data.
Scientific-rule changes require an explicitly versioned protocol before new outcome inspection; explanatory prose can be maintained independently.

References: [Benjamini–Hochberg](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x), [Newey–West](https://doi.org/10.2307/1913610), [Künsch](https://doi.org/10.1214/aos/1176347265), [Deflated Sharpe](https://doi.org/10.3905/jpm.2014.40.5.094).
