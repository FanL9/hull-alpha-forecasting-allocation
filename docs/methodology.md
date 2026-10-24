# Methodology

The project studies whether anonymous financial inputs forecast S&P 500 excess returns and whether those forecasts can support dynamic allocation.
See [Project Flow](project_flow.md) for execution order and [Research Protocol](research_protocol.md) for all frozen parameters and decision rules.

## Data and provenance

Use only official Hull Tactical competition inputs, preserve their original order and bytes, and verify acquisition hashes before analysis.
The official mock test file supports inference/schema checks; research evaluation uses reserved chronological training history.
Field meanings and availability are defined in the [data dictionary](data_dictionary.md).

## Data audit

Check schema, numeric types, identifiers, duplicates, missingness, infinities and feature-family membership without repairing inputs silently.
Keep data-quality findings separate from predictive evidence.

## Descriptive EDA

Study feature distributions, missing intervals, family structure, temporal changes and X–X redundancy; describe the target through a separate entry point.
Retain source values, encodings, extremes and NaNs by default, and do not interpret descriptive patterns as alpha.
Completed findings are recorded in the [EDA report](../reports/phase2_data_processing_eda.md).

## Research protocol

Freeze data boundaries, training-only availability, validation, inference, multiplicity and baselines before examining predictive relationships.
Configuration and executable code define the rules; the protocol lock verifies those files and input-data hashes while leaving explanatory documents editable.

## Alpha research

Evaluate eligible inputs strictly out of sample with expanding historical training windows, recording coverage and fold stability.
Primary Alpha metric: **block-wise Spearman Rank IC** (X and y ranked inside each validation block; ranked blocks concatenated without re-ranking).
Secondary robustness metric: **training-window 1%/99% winsorized Pearson IC** (X only; bounds fitted on training and reused on validation; never a discovery gate).
Newey–West HAC supplies p-values; the block bootstrap on the ranked series supplies confidence intervals only.
Test 97 single features and **7 predefined family-composite hypotheses** (M/E/I/P/V/S/D) as two families with separate BH-FDR 0.05, plus a **104-hypothesis global BH sensitivity analysis** and BY sensitivity, under the [current protocol](research_protocol.md) (phase2.5-v2).

## Forecasting

Fit preprocessing, feature selection and models inside each historical training window only; choose hyperparameters on nested inner blocks.
Compared forecasters: a nested raw-level composite (A), the same selector on 499 engineered candidates (B), Ridge/GRU/TCN on 63-day windows of 16 cluster representatives, their GRU+TCN average (C), an equal-weight combination of normalized A, B and C (D), and, for the competition track only, a recency-weighted ridge on the 97 raw levels (DynRidge).
A frozen rule selects the primary forecaster on research folds; it is then refit and evaluated once on the Internal Holdout without further tuning.

## Signal and allocation

Standardize the forecast with research out-of-sample location and scale, then map it to exposure in [0, 2]: linear clip(1 + kz) or volatility-scaled clip(c(1 + kz)), where c is a trailing 252/21-day volatility ratio of lagged returns.
The mapping is chosen by a frozen admissibility, near-best and simplicity rule on research blocks; a forecast-free volatility-only rule is reported as a comparator.
Prediction quality and portfolio utility are separate questions and are reported separately.

## Evaluation

Calculate strategy returns and the [official competition metric](official_metric_notes.md) from allocations, market returns and cash rates.
Turnover and 1/5 bp transaction-cost sensitivity are reported separately from the unmodified official score; block-bootstrap intervals are descriptive.

## Competition simulation

Refit once on 0–8866 (purge 8867), freeze all learned state, and process 8868–9047 in order using only current features, official lagged fields and the trailing volatility state.
Forecasts are sealed before any Final outcome is attached; the run is single-use. The research track and the disclosed competition track ([Protocol v5](protocol_v5_competition_track.md)) are run together and not ranked.

Historical result snapshots belong in [reports/](../reports/); current rules belong in this documentation and the linked configuration.
