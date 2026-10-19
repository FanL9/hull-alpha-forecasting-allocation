# Phase 3 Extended Post-hoc Timing Diagnostics

**This analysis is post-hoc and does not alter the pre-registered Phase 3 conclusions.**
It is exploratory: no p-values, FDR or new pass criteria are computed.

## Why this extension

The earlier 0–5 day lag test can flag short-lag instability, but it cannot rule out a one-day misalignment, certify publication timing, or reliably distinguish slow look-ahead (for example centered smoothing, later revisions or backfilled levels) from persistent inputs.
This extension shifts X by k ∈ {0, 1, 5, 21, 63, 126, 252} rows and tests changes ΔX_h = X_t − X_(t−h) for h ∈ {1, 5, 21}.
All shifts use only earlier rows of date_id 0–7859 and the sealed Phase 3 OOS grid, eligibility masks and composite recipes; k=0 reconciles to the sealed results.

Reading guide (descriptive, not decision rules):

- IC that stays similar out to k=126–252 is consistent with a slow level/regime association. Dependence can reduce effective information, but these diagnostics do not estimate an effective independent-observation count.
- IC that holds for small k and weakens at longer k is consistent with a finite dependence horizon. A smoothing or publication window is one possible explanation; these sparse lag points cannot identify a window length or establish look-ahead.
- A daily change ΔX_1 with IC comparable to the level suggests the information sits in the latest update and needs timing scrutiny.

Local execution: source commit 9c6a0da9d3ed679406348b3216f52927a946a565, clean worktree; the existing safe loader returned exactly 7860 rows, date_id 0–7859. Formal Phase 3 was not rerun. The 13 new synthetic/unit tests passed locally (0 failed, 0 skipped).

## Single-feature long-lag decay (common sample)

| feature | IC_k0 | IC_k1 | IC_k5 | IC_k21 | IC_k63 | IC_k126 | IC_k252 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| M1 | +0.097 | +0.091 | +0.082 | +0.049 | +0.028 | -0.032 | +0.029 |
| M3 | +0.060 | +0.020 | +0.046 | +0.030 | +0.018 | +0.006 | +0.021 |
| M9 | -0.112 | -0.097 | -0.080 | -0.035 | -0.028 | -0.006 | -0.019 |
| M11 | +0.099 | +0.095 | +0.061 | +0.042 | +0.019 | -0.003 | +0.013 |
| E2 | -0.107 | -0.093 | -0.068 | -0.040 | -0.008 | +0.028 | -0.002 |
| E3 | -0.098 | -0.082 | -0.063 | -0.033 | -0.002 | +0.033 | -0.008 |
| E19 | +0.084 | +0.055 | +0.051 | +0.053 | +0.004 | -0.019 | +0.006 |
| P2 | +0.099 | +0.093 | +0.066 | +0.040 | +0.015 | -0.023 | +0.006 |
| P3 | -0.067 | +0.043 | -0.011 | +0.006 | -0.009 | +0.011 | -0.028 |
| P4 | +0.057 | -0.034 | +0.013 | +0.021 | -0.015 | +0.022 | +0.005 |
| P5 | -0.057 | -0.064 | -0.035 | -0.043 | +0.001 | +0.004 | -0.026 |
| P8 | -0.074 | -0.066 | -0.058 | -0.042 | -0.011 | +0.034 | +0.011 |
| P10 | -0.114 | -0.103 | -0.073 | -0.044 | -0.009 | +0.035 | -0.009 |
| P11 | -0.097 | -0.083 | -0.062 | -0.034 | -0.002 | +0.033 | -0.005 |
| V7 | +0.072 | +0.062 | +0.055 | +0.039 | +0.002 | -0.020 | +0.017 |
| V9 | +0.069 | +0.058 | +0.056 | +0.032 | +0.000 | -0.024 | +0.011 |
| V10 | +0.069 | +0.059 | +0.075 | +0.054 | +0.017 | -0.053 | +0.006 |
| V13 | +0.066 | +0.051 | +0.045 | +0.045 | -0.006 | -0.020 | +0.024 |
| S2 | -0.074 | -0.070 | -0.049 | -0.026 | -0.027 | -0.006 | -0.026 |
| S3 | -0.067 | -0.062 | -0.053 | -0.040 | -0.025 | +0.037 | -0.016 |
| S5 | +0.056 | +0.045 | +0.052 | +0.019 | +0.011 | -0.032 | +0.037 |
| S8 | -0.081 | -0.077 | -0.060 | -0.033 | -0.019 | -0.001 | -0.012 |
| lagged_forward_returns | -0.057 | -0.003 | -0.025 | -0.046 | -0.001 | -0.005 | -0.008 |
| lagged_market_forward_excess_returns | -0.057 | -0.003 | -0.025 | -0.046 | -0.001 | -0.005 | -0.008 |

## Composite long-lag decay (common sample)

Composites exist only on the OOS grid, so long shifts leave early rows missing. The common sample has 2520 rows rather than the natural k=0 sample of 2772. The displayed common-sample k=0 IC therefore need not equal the sealed full-OOS IC. The natural-sample k=0 IC and pair count reconcile for all 28 single/composite objects; both sample definitions are retained in the long-form CSVs. Historical composites keep their original fold transform when shifted across a fold boundary.

| feature | common_sample_n_obs | IC_k0 | IC_k1 | IC_k5 | IC_k21 | IC_k63 | IC_k126 | IC_k252 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| M | 2520 | +0.088 | +0.065 | +0.060 | +0.042 | +0.021 | -0.002 | +0.033 |
| P | 2520 | +0.067 | +0.048 | +0.046 | +0.069 | -0.002 | -0.041 | -0.002 |
| V | 2520 | +0.049 | +0.042 | +0.039 | +0.035 | +0.002 | -0.020 | +0.008 |
| S | 2520 | +0.084 | +0.075 | +0.046 | +0.011 | +0.031 | -0.007 | -0.009 |

## Level versus change signals

Each horizon h compares levels and changes on the same complete-pair dates for that horizon. The CSV records the matched level IC and n_obs separately for every h; the compact table uses the h=1 level column. Here the 24 features have identical matched counts and level ICs across h=1,5,21. The nonzero-change fraction uses exact numerical inequality, not a tolerance or a reliable count of economic information releases.

| feature | level IC | ΔX_1 IC | ΔX_5 IC | ΔX_21 IC |
| --- | --- | --- | --- | --- |
| M1 | +0.097 | +0.038 | +0.030 | +0.036 |
| M3 | +0.060 | +0.042 | +0.002 | +0.019 |
| M9 | -0.112 | -0.051 | -0.051 | -0.060 |
| M11 | +0.099 | -0.007 | +0.041 | +0.040 |
| E2 | -0.107 | -0.056 | -0.060 | -0.066 |
| E3 | -0.098 | -0.055 | -0.059 | -0.064 |
| E19 | +0.084 | +0.030 | +0.045 | +0.023 |
| P2 | +0.099 | +0.020 | +0.060 | +0.059 |
| P3 | -0.067 | -0.076 | -0.037 | -0.050 |
| P4 | +0.057 | +0.063 | +0.034 | +0.027 |
| P5 | -0.057 | +0.006 | -0.015 | -0.009 |
| P8 | -0.074 | -0.029 | -0.037 | -0.038 |
| P10 | -0.114 | -0.035 | -0.059 | -0.053 |
| P11 | -0.097 | -0.055 | -0.058 | -0.062 |
| V7 | +0.072 | +0.021 | +0.023 | +0.021 |
| V9 | +0.069 | +0.029 | +0.016 | +0.026 |
| V10 | +0.069 | +0.032 | -0.021 | -0.006 |
| V13 | +0.066 | +0.020 | +0.027 | +0.012 |
| S2 | -0.074 | +0.007 | -0.030 | -0.042 |
| S3 | -0.067 | -0.013 | -0.028 | -0.016 |
| S5 | +0.056 | +0.027 | +0.009 | +0.024 |
| S8 | -0.081 | -0.017 | -0.036 | -0.042 |
| lagged_forward_returns | -0.057 | -0.043 | -0.018 | -0.015 |
| lagged_market_forward_excess_returns | -0.057 | -0.043 | -0.018 | -0.015 |

Full tables: [long lag](phase3_posthoc_extended/long_lag_decay.csv), [long-lag summary](phase3_posthoc_extended/long_lag_summary.csv), [composite long lag](phase3_posthoc_extended/composite_long_lag_decay.csv), [change signals](phase3_posthoc_extended/change_signal_ic.csv).

## Limitations

Block-wise ranks are computed within the frozen 252-row validation blocks. A feature constant throughout a block has no defined within-block Rank IC there; this estimator targets within-block association and does not measure between-block level differences. Long-lag rank IC is not an effective sample-size estimate.

Long-lag and change diagnostics cannot certify feature release times. Rows inside a 252-row block are strongly dependent for slow inputs, so similar IC across lags does not imply many independent confirmations. The Internal Holdout remains the decisive out-of-sample check.

This post-hoc diagnostic analysis does not alter the frozen Phase 3 PASS/FAIL/NOT_TESTABLE conclusions.
No Internal Holdout or Final-180 data were used.
