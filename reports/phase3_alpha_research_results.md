# Phase 3 Real Alpha Research

## Run identity

- Git commit: `5b0b25b08f93be373bcb0049340ce5ce5cd45987`
- Git tag: `phase3-pipeline-v1` (annotated); worktree clean at execution.
- Protocol: `phase2.5-v2`; pipeline hash: `987ac1a342c2a551155eb855997aaa842d528f52ed7d9495971406bab9f4ddba`.
- Research boundary: date_id 0–7859 (7860 rows), 11 frozen walk-forward folds.
- Run timestamp: `2026-10-18T12:39:17.931904+00:00`; started: `2026-10-18T12:38:19.136200+00:00`; completed: `2026-10-18T12:39:18.052109+00:00`.
- Execution count: **1**; reruns: **0**. Primary statistic: block-wise Spearman Rank IC; two-sided NW HAC lag 5; BH-FDR 0.05 applied separately to 97 and 7 hypotheses.
- Bootstrap: 2000 paired moving-block draws, block length 20, 95% interval, seed 42; confidence intervals only. Secondary Pearson clips X at training-window 1%/99% bounds and does not winsorize y.

Integrity checks streamed original bytes for SHA-256. Value-bearing reads used the existing load_research_prefix with nrows=7860. Neither Internal Holdout (7860–8867) nor Final 180 (8868–9047) was parsed or used.
The [read-only snapshot](phase3_real_run/) preserves original output bytes, including line endings. [Snapshot seal](phase3_real_run/snapshot_seal.json) hashes were checked before reporting; snapshot-specific Git attributes disable newline conversion. No statistical results were recomputed.

## Single-feature results

Total hypotheses: **97**. **PASS 24 / FAIL 72 / NOT_TESTABLE 1**.

All PASS: `M1`, `M3`, `M9`, `M11`, `E2`, `E3`, `E19`, `P2`, `P3`, `P4`, `P5`, `P8`, `P10`, `P11`, `V7`, `V9`, `V10`, `V13`, `S2`, `S3`, `S5`, `S8`, `lagged_forward_returns`, `lagged_market_forward_excess_returns`.
`E7` is NOT_TESTABLE: insufficient_eligible_folds, 2 eligible folds; its p=1 remains in the 97-hypothesis denominator.

[Complete single-feature results](phase3_real_run/single_feature_results.csv).

## Family-composite results

Total hypotheses: **7**. **PASS 4 / FAIL 3 / NOT_TESTABLE 0**.
All PASS: `M`, `P`, `V`, `S`.

| Family | Formal conclusion |
| --- | --- |
| M | PASS |
| E | FAIL |
| I | FAIL |
| P | PASS |
| V | PASS |
| S | PASS |
| D | FAIL |

[Complete composite results](phase3_real_run/family_composite_results.csv).

## Sensitivity analysis

| Analysis | Hypotheses | Rejected | Role |
| --- | --- | --- | --- |
| single_feature_bh | 97 | 24 | family_primary |
| family_composite_bh | 7 | 4 | family_primary |
| global_bh_sensitivity | 104 | 27 | sensitivity |
| single_feature_by_sensitivity | 97 | 18 | sensitivity |
| family_composite_by_sensitivity | 7 | 3 | sensitivity |
| global_by_sensitivity | 104 | 20 | sensitivity |

BY and global-104 results do not change the primary family-specific BH conclusions. BY, global BH/BY, bootstrap intervals, Pearson IC and fold consistency are not alternative discovery gates.

## Descriptive diagnostics

The following tables show every PASS in the original hypothesis order. Values are displayed with rounding; the sealed CSV files retain the original full precision. These diagnostics introduce no additional formal grades or pass thresholds.

### Single-feature PASS diagnostics

| Hypothesis | OOS Rank IC | NW p-value | BH q-value | Bootstrap 95% CI | Winsorized Pearson IC | Eligible folds | Direction consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | +0.096789 | 1.24719e-05 | 0.000111986 | [+0.053197, +0.138598] | +0.074534 | 7 | 100.0% |
| M3 | +0.059770 | 0.000897208 | 0.00511936 | [+0.023305, +0.096827] | +0.063445 | 11 | 90.9% |
| M9 | -0.111541 | 4.96459e-10 | 2.40782e-08 | [-0.145148, -0.080466] | -0.000038 | 11 | 100.0% |
| M11 | +0.098769 | 8.87265e-08 | 1.2295e-06 | [+0.065834, +0.133452] | +0.020970 | 11 | 100.0% |
| E2 | -0.106599 | 4.51205e-09 | 1.4589e-07 | [-0.137927, -0.073848] | -0.047404 | 11 | 100.0% |
| E3 | -0.097614 | 6.89675e-08 | 1.2295e-06 | [-0.129867, -0.063961] | -0.050961 | 11 | 90.9% |
| E19 | +0.084320 | 4.03783e-06 | 4.89587e-05 | [+0.049728, +0.118545] | +0.020658 | 11 | 100.0% |
| P2 | +0.099403 | 3.05849e-08 | 7.41684e-07 | [+0.067749, +0.131177] | +0.048587 | 11 | 100.0% |
| P3 | -0.067217 | 0.000422749 | 0.00273377 | [-0.103564, -0.030875] | -0.072200 | 11 | 90.9% |
| P4 | +0.056502 | 0.00408113 | 0.0179941 | [+0.015971, +0.093525] | +0.059916 | 11 | 81.8% |
| P5 | -0.056510 | 0.00169785 | 0.00914955 | [-0.091406, -0.024156] | -0.019149 | 11 | 90.9% |
| P8 | -0.074048 | 1.26994e-05 | 0.000111986 | [-0.104334, -0.045226] | -0.062812 | 11 | 90.9% |
| P10 | -0.113534 | 1.98561e-10 | 1.92604e-08 | [-0.144765, -0.081669] | -0.055587 | 11 | 100.0% |
| P11 | -0.096909 | 7.70865e-08 | 1.2295e-06 | [-0.128063, -0.064040] | -0.038479 | 11 | 100.0% |
| V7 | +0.071953 | 9.92512e-05 | 0.000740567 | [+0.037861, +0.104620] | +0.040141 | 11 | 90.9% |
| V9 | +0.069133 | 0.000139983 | 0.000969885 | [+0.034287, +0.102229] | +0.043294 | 11 | 72.7% |
| V10 | +0.069326 | 0.0105185 | 0.0425124 | [+0.017868, +0.122338] | +0.043670 | 5 | 100.0% |
| V13 | +0.065618 | 0.000836622 | 0.00507202 | [+0.033693, +0.100587] | +0.038218 | 11 | 81.8% |
| S2 | -0.073719 | 4.73781e-05 | 0.000382973 | [-0.105412, -0.039700] | -0.030981 | 11 | 90.9% |
| S3 | -0.066517 | 0.00542218 | 0.0228675 | [-0.105077, -0.027313] | -0.030695 | 6 | 100.0% |
| S5 | +0.056478 | 0.00360701 | 0.0166609 | [+0.021184, +0.093695] | +0.059335 | 11 | 81.8% |
| S8 | -0.080605 | 7.55424e-06 | 8.14179e-05 | [-0.111454, -0.047230] | -0.044716 | 11 | 90.9% |
| lagged_forward_returns | -0.057175 | 0.0027842 | 0.0135857 | [-0.094946, -0.019641] | -0.073601 | 11 | 81.8% |
| lagged_market_forward_excess_returns | -0.057144 | 0.00280117 | 0.0135857 | [-0.094884, -0.019734] | -0.073460 | 11 | 81.8% |

### Family-composite PASS diagnostics

| Hypothesis | OOS Rank IC | NW p-value | BH q-value | Bootstrap 95% CI | Winsorized Pearson IC | Eligible folds | Direction consistency |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M | +0.082506 | 4.57143e-06 | 1.99063e-05 | [+0.047629, +0.117647] | +0.029535 | 11 | 90.9% |
| P | +0.055053 | 0.0039011 | 0.00910258 | [+0.017319, +0.092520] | +0.043872 | 11 | 81.8% |
| V | +0.044620 | 0.0152868 | 0.0267519 | [+0.008755, +0.079234] | +0.011708 | 11 | 90.9% |
| S | +0.082395 | 5.68751e-06 | 1.99063e-05 | [+0.049977, +0.115201] | +0.047184 | 11 | 100.0% |

[Fold IC](phase3_real_run/fold_level_ic.csv), [training availability](phase3_real_run/feature_availability.csv), [transform audit](phase3_real_run/fold_transform_audit.json), [multiplicity summary](phase3_real_run/multiple_testing_summary.csv) and [run manifest](phase3_real_run/phase3_run_manifest.json) are archived without alteration.

## Formal conclusion

Single features: **PASS 24, FAIL 72, NOT_TESTABLE 1**. Family composites: **PASS 4, FAIL 3, NOT_TESTABLE 0**.
These are the frozen screening decisions. No research rule, threshold, hypothesis, feature direction, preprocessing or pipeline code was changed after observing results. No Phase 4 model, feature reselection, allocation, backtest or Kaggle score was produced.

Real Phase 3 Alpha Research completed once under the frozen phase3-pipeline-v1 protocol.
No Internal Holdout or Final-180 data were used.
