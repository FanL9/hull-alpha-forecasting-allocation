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

Evaluate eligible inputs strictly out of sample with expanding historical training windows, recording raw and training-oriented correlations, coverage and fold stability.
Use dependence-aware uncertainty and multiplicity control under the [current protocol](research_protocol.md), rather than selecting features from raw p-values or one favorable interval.

## Forecasting

Fit preprocessing, any feature selection and candidate models inside the corresponding historical training windows, then compare later forecasts with fixed baselines.
Freeze the selected forecasting pipeline before its one-off internal holdout evaluation; its outcomes do not guide further selection.

## Signal and allocation

Translate forecasts into bounded exposure using research-period calibration, then freeze allocation rules without tuning on held-out outcomes.
Prediction quality and portfolio utility are separate questions.

## Evaluation

Calculate strategy returns and the [official competition metric](official_metric_notes.md) from allocations, market returns and cash rates.
Report any later transaction-cost or financing analysis separately from the unmodified official score; reserve Deflated Sharpe for later model/strategy searches.

## Competition simulation

Run the frozen pipeline sequentially through the locked final period with only inference-time inputs and the predeclared update policy.
The current implementation includes acquisition, EDA and protocol tools; Alpha Research, forecasting and portfolio evaluation have not started.

Historical result snapshots belong in [reports/](../reports/); current rules belong in this documentation and the linked configuration.
