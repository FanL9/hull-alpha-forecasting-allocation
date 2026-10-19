# Phase 3 Post-hoc Interpretation

**This interpretation is post-hoc and exploratory. It does not alter the pre-registered Phase 3 PASS/FAIL/NOT_TESTABLE conclusions.**
It summarizes two diagnostic snapshots that were generated, hashed and committed before this text was written:

- [First diagnostics](phase3_posthoc_diagnostics.md): k = 0–5 lag decay, reverse past-return IC, rank/Pearson divergence, deciles and correlation clusters.
- [Extended diagnostics](phase3_posthoc_extended.md): k = 0, 1, 5, 21, 63, 126, 252 lag decay on a fixed common sample, and level versus ΔX_h = X_t − X_(t−h) for h = 1, 5, 21.

Both used only date_id 0–7859 through the safe research-prefix loader, the sealed Phase 3 out-of-sample grid, eligibility masks and composite recipes. No p-values, FDR or new pass criteria were computed. No Internal Holdout or Final-180 data were used. The generated reports and their manifests are left unchanged; numbers below are copied from their CSV files.

## Scale of noise

With about 2772 pooled out-of-sample rows, the standard error of a single Rank IC is roughly 0.02. Diagnostic values with |IC| below about 0.04 are treated as indistinguishable from noise. Many PASS features are highly correlated, so agreement across them is not independent confirmation.

## 1. Timing: no sign of a one-day shift or of slow look-ahead

**One-day timing.** For about 20 of the 24 single-feature PASS signals, IC falls by only 4–23% from k = 0 to k = 1. A feature that used information released during t → t+1 would typically collapse at k = 1. The two lagged outcome fields do collapse (−0.057 → −0.003), but they pass an exact t−1 construction audit; their cliff reflects short-lived return autocorrelation, not leakage.

**Slow look-ahead.** A persistent input barely changes over a few rows, so the 0–5 day test cannot detect centered smoothing, revisions or backfilled levels. The extended shifts address this:

| Signal | k=0 | k=1 | k=5 | k=21 | k=63 | k=252 |
| --- | --- | --- | --- | --- | --- | --- |
| P10 | −0.114 | −0.103 | −0.073 | −0.044 | −0.009 | −0.009 |
| M9 | −0.112 | −0.097 | −0.080 | −0.035 | −0.028 | −0.019 |
| E2 | −0.107 | −0.093 | −0.068 | −0.040 | −0.008 | −0.002 |
| P2 | +0.099 | +0.093 | +0.066 | +0.040 | +0.015 | +0.006 |
| M composite | +0.088 | +0.065 | — | +0.042 | +0.021 | +0.033 |

Neither alarming pattern appears:

- A centered-smoothing leak keeps IC roughly flat up to its half-window and then drops to zero (synthetic tests reproduce this). The observed IC declines gradually instead.
- A pure slow level or era effect keeps IC through k = 126–252. The observed IC is near zero by k ≈ 63.

The most natural reading is that the features carry information whose relevance to next-day returns fades over several weeks to about two months. This is consistent with genuine predictive content, but it does not certify release timestamps.

## 2. Level versus change

- **State plus update** (E2, E3, M9, P11): ΔX IC is about 50–60% of the level IC (for example E2: level −0.107, ΔX₂₁ −0.066).
- **Mostly state** (P5, S2, S3, S8, V family): ΔX₁ IC is within noise; the information sits in the level.
- **Fast features** (P3, P4, and less strongly M3): ΔX₁ IC is as large as the level IC (P3: −0.076 vs −0.067; P4: +0.063 vs +0.057), and IC changes sign at k = 1. Their information lives in the latest daily update. They may be legitimate short-horizon signals or may have a timing issue; these diagnostics cannot tell which.
- Nonzero-change fractions are near 100% for every PASS feature, so no stale step-updated input was found.

## 3. Redundancy and other observations

- The 24 single-feature PASS signals form 16 exploratory clusters (12–20 depending on the cut). Strong pairs include the two lagged fields (≈1), V9/V10 (0.98), V7/V9 (0.98), E3/P11 (0.96) and M9/M11 (−0.84). The number of independent signals is far below 24.
- Many features encode past returns (for example V13 with the previous day's return: −0.33, consistent with the leverage effect). This alone does not indicate label contamination.
- P5 relates weakly to the previous day's return (−0.08) but strongly to returns two to three days back (+0.28), suggesting it is published with an extra day of delay. This is staleness, not look-ahead.
- E2, E3, P8, P10 and P11 turn to about +0.03 at k = 126. They are highly correlated, so this is effectively one observation of about 1.5 standard errors; possibly half-year mean reversion, possibly noise.
- Under block-wise ranking, a feature that is constant within a 252-row block has undefined IC, so the primary metric measures within-block timing, not level differences between blocks.

## 4. Implications for Phase 4

1. Treat the smoothly decaying signals (about 20) as the main candidate set, grouped by correlation cluster rather than entered as 24 separate inputs.
2. Keep the fast features P3, P4 and M3 in a separate sensitivity variant, and compare final out-of-sample results with and without them.
3. Information that fades over weeks supports daily allocation updates and periodic refitting; the frozen 252-row expanding schedule is adequate.
4. These diagnostics rule out two common leakage patterns but cannot certify anonymous feature release times. The single, frozen Internal Holdout evaluation remains the decisive check, and must not be used for redesign.

This post-hoc interpretation does not alter the frozen Phase 3 PASS/FAIL/NOT_TESTABLE conclusions.
No Internal Holdout or Final-180 data were used.
