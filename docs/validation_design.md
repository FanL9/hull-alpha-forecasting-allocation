# Time-series validation design

Status: design only. No split or final period has been instantiated, and no model has been trained.

## Chronology and label timing

Validate unique, finite, integer-like and monotonically increasing `date_id` values; do not silently reorder them. `date_id` does not by itself establish a calendar date or consecutive trading days. Confirm official feature publication timing, label horizon and evaluation API behavior before determining the precise gap.

The intended forecast label is the future excess-return label already attached to row `t`. Do not shift an already-forward label again without documentary justification. Retrospective outcomes, contemporaneous risk-free fields and unverified lagged API fields are excluded from predictors.

## Development and final holdout

After acquisition, define a chronological tail from labelled training history as the final holdout. The provisional configuration reserves 20%; this is a design value, not a completed split. Record its exact `date_id` cut, row counts, input hashes and configuration before development target exploration. Allow schema, file integrity and missingness inventory across files, but keep final labels unused in exploratory target analysis, feature selection, calibration and tuning.

Do not use official `test.csv` as a research holdout merely because of its filename. It may be an API example, overlap training or include evaluation-specific fields. Audit the actual relationship and review the official API.

## Expanding-window sequence

```text
earlier history     gap   validation 1
earlier + block 1   gap   validation 2
earlier + blocks 1–2 gap  validation 3
frozen development specification   gap   final holdout
```

All validation intervals are contiguous and strictly later than their training interval. No shuffled splitting or random KFold. The provisional one-row gap is conservative for an assumed one-step horizon and must be confirmed; for multi-row labels purge every training label whose outcome interval reaches into validation. Place a corresponding gap before the final holdout.

Fit imputers, missing indicators, scalers, transformations, feature selectors, normalization, regime thresholds and models only using each training window. Forward filling may carry historical values forward with a defined limit, but must never backfill future observations. All-missing training columns require an explicit stable policy. Do not use validation or final outcomes to repair features.

## Tuning and model selection

Train historical mean and linear models first. Compare all candidates using matched out-of-sample rows and a training-only baseline. If hyperparameters are tuned, use inner chronological splits. Select ensemble weights, signals, mappings, costs and risk overlays inside development evaluation only. Record evaluated candidates and selection criteria to expose research degrees of freedom.

Prediction correlation and directional accuracy supplement error metrics. Portfolio utility is assessed net of declared costs and financing at comparable risk. No single metric determines model superiority.

## Final evaluation protocol

Freeze a configuration, code revision, feature list and rule for allowable refits before opening final labels. Specify whether the final period uses a frozen model or a predeclared sequential refit schedule that only consumes already realized earlier labels. Evaluate once; any subsequent design changes make that period development data and require a new independent test for a fresh claim.

## Later implementation checks

Add tests when those phases are implemented: training and outcome intervals precede validation; perturbing future rows cannot change earlier states/signals/weights; preprocessing fit state is training-only; weights remain within `[0,2]`; and portfolio return timing, turnover and costs agree with an explicit ledger. These checks are planned and are not included in current Phase 1 test claims.
