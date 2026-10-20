# Phase 5 Synthetic Calibration

Synthetic engineering validation only. **No real Phase 5 market X-Y results were observed.**

## Pipeline Identity

Protocol `protocol-v3.1` → `4ce62ab8f5a1dde62b2d682b1e5b1ae008e71f7b`.
Development base commit: `c781453588d53e9c66ede15a2693d5fa9be0f67f`.
Pipeline code hash: `bf26dca9c36312707c9fd8e27bb32f6a0682c78e202f144be09ee4333cf63bcb`.
The annotated `phase5-deep-alpha-pipeline-v1` tag binds the final execution freeze.
Synthetic validation precedes that commit; its manifests bind the development
base commit AND exact implementation bytes. No model or generator was tuned
from calibration outcomes. The prospective specification remains unchanged.
For release verification, all three source CSVs and their split policies were
independently regenerated from the registered cases/seeds and matched byte for
byte. Their generator-verified SHA256 values are recorded in the evidence.
Environment: Python 3.12.6, PyTorch 2.10.0+cpu,
NumPy 2.5.4, scipy 1.18.1, pandas 3.0.6.
Machine-readable [evidence](phase5_synthetic_calibration/calibration_summary.json).

## Architecture Verification

Actual GRU parameters: **633**. Actual TCN parameters: **1137**. Five-seed combined
budget: **8850**. Ridge: 1008 slopes plus unpenalized intercept. All three models
use the same reference-specific representatives and 63 × 16 inputs. Neural CPU
float32; deterministic Ridge SVD float64. No architecture search was performed.

## Data Isolation Tests

The actual CLI/production execution chain reads a synthetic **9048 × 98** CSV and
production-shaped split policy. A non-invasive Python call observer records the
real pandas reader: exactly `nrows=7860`, 7860 numeric rows, last date_id 7859.
Explicit string sentinels in 7860–8867 and 8868–9047 never enter the parsed frame.
No production reader, split loader, feature/window builder, writer or state system
is mocked. The E2E-only one-epoch compute budget is rejected by real/calibration
modes. Full calibration uses the unmodified production 100-epoch/early-stop path.
All test/calibration workers prohibit access to the actual project `data/` tree.

## Window Leakage Tests

First admissible endpoint 62; each input is exactly t−62…t. Tests reject earlier
endpoints and dates ≥7860, mutate future X/y, and verify unchanged earlier windows,
representatives, clusters, scalers, initialization plan and epoch-selection setup.
Selectors/scalers are fitted separately on every purged reference and internal
stopping prefix. Stopping is [R−251,R], purge R−252, weights through R−253.
Each fold's subsequent y is absent from its fitting capsules; past realized
official lagged outcomes remain legal X. Earlier validation can become later
training history. OOF normalization/residualization accepts score arrays, not y.

## Determinism

Python/NumPy/Torch seeds are fixed. CPU deterministic algorithms, one numerical
thread and chronological batches are enforced. Repeated models serialize to
identical bytes. A fresh full E2E run with workers=1 matches workers=4 exactly:
all predictions, fold tables, summaries, fitted task artifacts, audit and manifest
bytes. Timestamp/wall-time logs are separate from research payloads.

## Resume / Cache Safety

Same-identity completed stages are immutable. Incomplete stages restart from the
original seed, without loading partial weights. Tests reject code/data identity
changes, missing/corrupt completed artifacts, altered receipts, missing canonical
state/anchor and deleted caches. The independent real completion journal lives
outside the output tree. Post-evaluation resume can only replay sealed forecasts;
it cannot train. COMPLETE refuses another run. Completed calibration evidence is
also revalidated against source bytes, identity, inventory, seal and output hashes.
The state machine is documented in [execution design](../docs/phase5_pipeline.md).
Windows sharing/path incidents and all abandoned development attempts are
[recorded](phase5_engineering_incident.md); no market results were inspected.

## Pure Noise Calibration

Independent Gaussian continuous X and y; D independent Bernoulli; lagged outcomes reconstructed normally.

Seed: `101`; all 11 folds and all five initialization seeds retained. Predeclared criterion: Absolute equal-fold mean IC <= 0.10 for each of Ridge, GRU, TCN, C and D; no systematic nonlinear alpha in this fixed engineering panel. Not a significance claim or guarantee of zero individual folds. **PASSED**.

| pipeline | finite_folds | mean_fold_ic | median_fold_ic | ic_std | positive_fold_count | worst_fold_ic |
| --- | --- | --- | --- | --- | --- | --- |
| A | 11 | -0.021503 | 0.019259 | 0.119367 | 6 | -0.265542 |
| B | 11 | -0.011019 | 0.007937 | 0.078833 | 7 | -0.147110 |
| C | 11 | 0.015840 | 0.020735 | 0.065539 | 6 | -0.114349 |
| D | 11 | -0.004179 | 0.043103 | 0.096141 | 6 | -0.162240 |
| GRU | 11 | 0.025995 | 0.021119 | 0.051132 | 8 | -0.049342 |
| Ridge | 11 | 0.004996 | -0.005660 | 0.066955 | 4 | -0.076080 |
| TCN | 11 | -0.000811 | -0.010435 | 0.071495 | 5 | -0.147464 |

Full fold results: [noise_folds.csv](phase5_synthetic_calibration/noise_folds.csv).
## Linear Signal Recovery

M1 stationary Gaussian AR(0.98); y[t] = mean(M1[t-62:t+1]) + Gaussian noise std 0.1; endpoints before 62 missing.

Seed: `211`; all 11 folds and all five initialization seeds retained. Predeclared criterion: Ridge mean IC > 0.30; GRU and TCN each > 0.15 with all 11 folds finite. Neural models need not beat Ridge. **PASSED**.

| pipeline | finite_folds | mean_fold_ic | median_fold_ic | ic_std | positive_fold_count | worst_fold_ic |
| --- | --- | --- | --- | --- | --- | --- |
| A | 11 | 0.173704 | 0.150023 | 0.091515 | 11 | 0.081964 |
| B | 11 | 0.505123 | 0.575095 | 0.198723 | 11 | 0.085504 |
| C | 11 | 0.922509 | 0.944883 | 0.065572 | 11 | 0.765228 |
| D | 11 | 0.670980 | 0.726010 | 0.152942 | 11 | 0.362859 |
| GRU | 11 | 0.935503 | 0.951879 | 0.037953 | 11 | 0.864969 |
| Ridge | 11 | 0.936459 | 0.943069 | 0.037197 | 11 | 0.856298 |
| TCN | 11 | 0.820614 | 0.949324 | 0.290969 | 11 | 0.137713 |

Full fold results: [linear_folds.csv](phase5_synthetic_calibration/linear_folds.csv).
## Nonlinear Temporal Signal Recovery

M1 and E1 independent IID Gaussian; y[t] = M1[t] + E1[t] + 2*M1[t]*E1[t-1] + Gaussian noise std 0.2; first endpoint missing. Marginal linear terms make both causal inputs discoverable by the frozen univariate selector, while the cross-lag interaction is outside flattened linear Ridge.

Seed: `307`; all 11 folds and all five initialization seeds retained. Predeclared criterion: At least one of GRU/TCN exceeds Ridge equal-fold mean IC by > 0.05 with all 11 folds finite. This is a capacity check on the predeclared process, not model selection for market data. **PASSED**.

| pipeline | finite_folds | mean_fold_ic | median_fold_ic | ic_std | positive_fold_count | worst_fold_ic |
| --- | --- | --- | --- | --- | --- | --- |
| A | 11 | 0.195752 | 0.224944 | 0.090260 | 11 | 0.044154 |
| B | 11 | 0.388403 | 0.400292 | 0.061883 | 11 | 0.285165 |
| C | 11 | 0.983888 | 0.987129 | 0.011128 | 11 | 0.952783 |
| D | 11 | 0.649630 | 0.662368 | 0.055902 | 11 | 0.529263 |
| GRU | 11 | 0.992028 | 0.992232 | 0.001456 | 11 | 0.988875 |
| Ridge | 11 | 0.501133 | 0.502076 | 0.037966 | 11 | 0.444303 |
| TCN | 11 | 0.214254 | 0.153347 | 0.139438 | 11 | 0.047688 |

Full fold results: [nonlinear_folds.csv](phase5_synthetic_calibration/nonlinear_folds.csv).

## Parallel Reproducibility

Fresh workers=1 versus workers=4 research outputs are byte-identical; task order
cannot select seeds. A separate four-process 120-task stress test verifies actual
read/rename/receipt concurrency. Permission-denial retries publish/read the same
bytes, and persistent denial/corrupt content remains fatal.

## Runtime Benchmark Summary

The actual architectures were timed at 4772 stopping-weight sequences, 252
stopping sequences and 5025 refit sequences, each 63 × 16. They use production
max-100 epochs and normal early stopping, not the E2E smoke budget. Detailed
measurements and extrapolation: [runtime report](phase5_runtime_benchmark.md).

## Conclusion

All three preregistered synthetic capacity/bias checks passed. These controlled
processes establish engineering behavior, not real-market alpha or profitability.
Full suite: **479 passed, 0 failed, 0 skipped** (399 previous +
80 new). All 213 original tracked/protected file bytes
remain unchanged, including Protocol v3.1 and all sealed Phase 3/4 evidence.

No real Phase 5 market X-Y results were observed.
Internal Holdout 7860–8867 and Final Simulation 8868–9047 remain untouched.
