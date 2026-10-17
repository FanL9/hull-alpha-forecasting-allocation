# Methodology

## Phase 1: implemented

1. Read project-relative YAML configuration, set random seeds and inspect dependency/credential presence without printing secrets.
2. Use the installed official Kaggle CLI to download the complete competition archive. Reject missing credentials, denied access and network failures; never substitute another source.
3. Validate ZIP integrity, destination paths, duplicate members and symlinks before extraction. Retain the original archive and all official evaluation/API files locally.
4. Hash the archive and extracted files. The audit requires an official acquisition manifest and matching train/test hashes. This records local acquisition provenance; hashes alone do not cryptographically attest the publisher.
5. Load actual CSVs in file order. Reject empty inputs and duplicate headers. Do not sort, shuffle, fill missing values or derive returns.
6. Report complete columns, shapes, types, target presence, feature families, missing values, blockwise family missingness, constant/all-missing fields, infinities, D-family deviations and date structure.
7. Compare train/test feature schemas and date ranges. Treat overlapping/example test rows as a warning against assuming a research holdout.
8. Generate one JSON summary, a human-readable audit and an observed-column dictionary. When inputs are unavailable, generate explicit blocked outputs with unknown statistics.

Family matching uses complete numbered-prefix names with `MOM` recognized separately from `M`. `date_id`, outcome fields, `is_scored`, lagged fields and unclassified columns are not automatically eligible model features. Feature timing still requires official documentation review.

## Phase 2: planned research

Reserve the chronological final holdout before any target-driven EDA. Explore development-only missingness, target behavior, feature distributions and family relationships. Compare raw and historically available transformations, without backfilling or using the complete sample to select predictors.

Measure Pearson/Spearman association with future labels, univariate forecast coefficients and out-of-sample direction. Rolling windows and family summaries describe temporal variation. Use multiplicity-aware summaries and, where justified, dependence-aware uncertainty estimates. Correlation with the supplied future label is predictive association only after alignment is confirmed; economic alpha requires later out-of-sample evidence.

## Phases 3–4: planned forecasting and regimes

Use expanding windows, a horizon-consistent purge gap, training-only preprocessing and nested chronological tuning if tuning is performed. Compare historical mean, OLS, Ridge and ElasticNet before Random Forest, XGBoost and LightGBM. Track the same out-of-sample observations across models. Fit any ensemble weights within development data.

Construct volatility states from already realized returns. Under a verified one-step-forward row convention, historical return information at row `t` comes from outcome rows strictly before `t`; the current row's future return never defines its own state. Estimate volatility cutoffs from earlier history, then apply them forward. Define bull/bear/neutral from a predeclared trailing realized trend with an initial warm-up. Fit statistical regime models only in each training window if used.

Re-estimate predictive power and model importance through time. Report feature importance as model attribution, not causal explanation; correlated anonymous features may share or distort importance.

## Phase 5: planned signals and economics

Compare raw predictions with historically standardized/percentile signals, clipping and causal smoothing. Signal calibration must use training or earlier out-of-sample forecasts. Map signals into `[0,2]` with linear, threshold and volatility-adjusted rules selected within development data.

After official semantics are established, an explicit starting accounting convention is `R[p,t+1] = w[t] R[m,t+1] + (1-w[t]) R[f,t+1] - transaction_cost[t] - borrowing_spread[t]`. Verify units, return alignment and financing before using it. A negative cash weight represents borrowing; specify the additional financing spread. Charge one-way trading costs on absolute changes in exposure, including the initial move from an explicit starting weight. Rebalancing and weight timing must be documented in a row-level ledger.

Drawdown control uses portfolio wealth realized before the current decision. Volatility targeting uses earlier realized return estimates and an explicit warm-up. Report ex-post strategy/benchmark volatility and any official penalty separately. Risk controls cannot guarantee the future 120% ratio.

## Phase 6: planned robustness

Freeze the model/signal/risk specification before one final holdout evaluation. Assess alternative development windows, gaps, cost and borrowing scenarios, sizing sensitivities, subperiods and conditional regimes. Report fold-level dispersion, dependent observations, heavy tails, researcher degrees of freedom and all material negative results. Verify the official evaluation code before implementing a competition score; do not relabel a custom Sharpe as the official score.
