# Phase 3 Synthetic Calibration

Software-validation snapshot only. No real X–Y data, Internal Holdout or Final 180 was read.
Current statistical rules remain [phase2.5-v2](../docs/research_protocol.md); the execution policy is documented in [Phase 3](../docs/phase3_alpha_research.md).

## Panel and acceptance

Each of 8 cases runs 11 folds, 97 single features, 7 composites and global-104 sensitivity, with all 2000 paired moving-block bootstrap repetitions. No hypothesis is removed from FDR.
Null inputs are IID normal/Bernoulli streams, including an IID target. The three lagged fields are reconstructed normally; the lagged IID target has zero population predictive IC.
Power signals use Gaussian copulas with population Spearman IC 0, ±0.03, ±0.05 and ±0.08, sharing an independent noise stream within each seed.

Fixed regression checks allow null mean absolute IC up to 0.04, absolute panel mean signed IC below 0.015, at least 4/5 null seeds with at most 2 BH discoveries, no seed above 8 and at most 15 total. These deliberately broad checks do not certify nominal FDR.
Power checks allow a 0.04 error in seed-mean IC, require the expected mean signs, and require an overall decline in median NW p-values as absolute strength increases. No weak signal must pass BH.

## Pure noise

| Seed | Mean signed IC | Mean absolute IC | BH pass | BY pass | Singles |
| --- | --- | --- | --- | --- | --- |
| 11 | -0.002015 | 0.015362 | 0 | 0 | 97 |
| 29 | 0.002766 | 0.018897 | 0 | 0 | 97 |
| 47 | -0.000807 | 0.015452 | 0 | 0 | 97 |
| 71 | -0.000479 | 0.014199 | 0 | 0 | 97 |
| 101 | 0.000658 | 0.014257 | 0 | 0 | 97 |

## Weak-signal power

Estimated IC is the mean across 3 fixed seeds (137, 271, 409); NW p-value is their median. BH/BY counts are descriptive counts out of 3 runs.

| Injected Rank IC | Estimated Rank IC | NW p-value | BH pass / 3 | BY pass / 3 |
| --- | --- | --- | --- | --- |
| -0.08 | -0.066156 | 0.000276067 | 2 | 1 |
| -0.05 | -0.036410 | 0.0420126 | 0 | 0 |
| -0.03 | -0.016589 | 0.311555 | 0 | 0 |
| +0.00 | +0.013192 | 0.586742 | 0 | 0 |
| +0.03 | +0.043350 | 0.0320427 | 1 | 0 |
| +0.05 | +0.063397 | 0.00145908 | 2 | 1 |
| +0.08 | +0.093387 | 1.56707e-06 | 3 | 3 |

Per-seed rows: [null CSV](phase3_calibration/pure_noise_calibration.csv), [power CSV](phase3_calibration/weak_signal_calibration.csv).
Provenance: [calibration manifest](phase3_calibration/calibration_manifest.json) records pipeline/file hashes, seeds, protocol version, full bootstrap settings and CSV hashes.

## Verification

**224 passed, 0 failed, 0 skipped.** Null calibration, power calibration, canonical single-run/concurrency guard, dirty-worktree guard, annotated-tag guard and enhanced manifest tests passed.
All 26 frozen Phase 2.5-v2 files still verify. No models, allocation or backtests were run. The Phase 3 freeze is a separate local execution tag; the Phase 2.5 protocol lock is unchanged.

**Real Phase 3 Alpha Research has NOT been executed.**
