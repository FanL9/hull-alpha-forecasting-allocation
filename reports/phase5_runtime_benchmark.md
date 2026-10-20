# Phase 5 Runtime Benchmark

**Artificial data only. No real Phase 5 X-Y was read.** Exact production
architectures/training code; CPU float32, one numerical thread per worker,
lookback 63, 16 channels. This is separate from the one-epoch E2E smoke test.
The final measurement was run after the long calibration jobs had stopped.

## Measured production-budget stages

First-outer-fold dimensions: 4772 chronological stopping-weight sequences,
252 stopping sequences and 5025 full-reference refit sequences. Seed initialization,
AdamW, gradient clipping, max=100, minimum checkpoint=10, patience=10 and
min_delta=0.0001 are unchanged. All measured stopping paths are recorded below.

Workers=1:

| model | seed | stopping_epochs_run | selected_epochs | stopping_seconds | refit_seconds | total_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| GRU | 11 | 38 | 28 | 55.140804 | 44.538333 | 99.679137 |
| TCN | 11 | 39 | 29 | 22.397632 | 16.853188 | 39.250820 |
| GRU | 22 | 23 | 13 | 35.999141 | 21.826662 | 57.825803 |
| TCN | 22 | 37 | 27 | 19.791797 | 12.306591 | 32.098388 |

Workers=4:

| model | seed | stopping_epochs_run | selected_epochs | stopping_seconds | refit_seconds | total_seconds |
| --- | --- | --- | --- | --- | --- | --- |
| GRU | 11 | 38 | 28 | 48.442117 | 40.820432 | 89.262548 |
| TCN | 11 | 39 | 29 | 16.351840 | 13.020953 | 29.372793 |
| GRU | 22 | 23 | 13 | 29.056986 | 16.982309 | 46.039294 |
| TCN | 22 | 37 | 27 | 15.694763 | 11.530613 | 27.225375 |

Measured serial wall time for the SAME four tasks: **228.98s**.
Measured parallel wall time: **93.57s**, including
worker startup. This small batch is unbalanced; the longest GRU task bounds its
critical path. Do not assume a fourfold speedup. Peak per-worker RSS was not
profiled. The runtime materializes at most the configured number of task capsules.

## Extrapolation, not real-run measurement

The honest nested DAG has **36 neural fitting references**, GRU/TCN × five seeds
× stopping/refit = **720 neural stages / 360 seed fits**. Reference reuse is
deterministic, not a model/seed-selection shortcut. Scale measured stage costs
by the exact mean reference training endpoint counts (stopping ratio
1.1802, refit ratio 1.1711), then use the measured parallel
wall/serial ratio 0.4086.

- Benchmark-like epoch paths: about **6.73 serial training hours**,
  or **2.75 hours with four workers**.
- If every stopping AND refit stage reaches 100 epochs: about
  **10.13 hours with the same measured throughput factor**.
  This scenario assumes constant cost per epoch; it is **not an upper bound**.
  Learning trajectories can also change per-epoch cost, so the isolated short
  benchmark paths cannot guarantee the duration of a complete research run.
- Add representative selection, A/B selectors, Ridge SVD, task I/O and evaluation;
  these are outside the neural-stage estimate. Different stopping paths,
  competing applications, power limits and file scanning can change wall time.

These are extrapolations from artificial dimensions and epoch paths, not an
observed real-market runtime or a guarantee. No real X-Y was inspected to choose
an epoch/runtime estimate. **Recommend four workers**, one CPU thread each;
worker count is an engineering option with no statistical effect.

Raw [measurements](phase5_synthetic_calibration/runtime_benchmark.json) and
[extrapolation](phase5_synthetic_calibration/runtime_extrapolation.json).
Pipeline hash: `bf26dca9c36312707c9fd8e27bb32f6a0682c78e202f144be09ee4333cf63bcb`.
