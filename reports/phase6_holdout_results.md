# Phase 6 — Internal Holdout Results (Research Track)

Status: **final for the research track.** The Internal Holdout (date_id 7860–8867, 1008 rows, 4 blocks of 252) was opened once under Protocol v4 / v4.1 with the frozen pipeline tag `phase6-holdout-pipeline-v1`. Public files: [phase6_holdout_public](phase6_holdout_public/README.md). The Final Simulation (8868–9047) remains unread.

## 1. Pre-registered primary endpoints

| Endpoint | Rule (v4.1) | Result | Verdict |
| --- | --- | --- | --- |
| A block-wise Rank IC | one-sided H0: IC ≤ 0, NW lag 5, α = 0.05 | 0.0183 (SE 0.0295, p = 0.267, 95% CI [−0.040, 0.076]) | **not confirmed** |
| Selected allocation `volscaled_k0.10` vs `constant_1` | paired moving-block bootstrap (20, 2000, seed 42) | 0.436 vs 0.426; difference +0.010, CI [−0.265, 0.238] | indistinguishable |
| `volscaled_k0.10` vs `volatility_only` | same | −0.034, CI [−0.139, 0.068] | indistinguishable; point estimate favours the forecast-free rule |

A's four holdout block ICs were −0.090, 0.095, 0.007, 0.061 (research: 0.0765 mean over 11 folds, 10/11 positive). The research-track conclusion is therefore:

> The frozen A forecast did **not** confirm out of sample. The allocation built on it was not distinguishable from holding the market, and its small research-period advantage came mainly from the volatility state c_t rather than from the forecast.

The last point is visible already in the sealed research selection: `linear_k0.10` (forecast only) scored 0.866 versus 0.864 for `constant_1`, while `volscaled_k0.10` scored 0.968. On the holdout every `linear_k*` mapping scored below `constant_1`, and `volatility_only` (added in v4.1 before the holdout was read) scored best (0.470).

## 2. Secondary forecasts (descriptive only)

Per v4 §8 a better secondary never replaces the primary. Block-wise Rank IC on the holdout, alongside the research figure and a block-weighted pool over all 15 out-of-sample blocks (11 research + 4 holdout, 3780 days):

| Pipeline | Research IC (11) | Holdout IC (4) | Holdout one-sided p | 15-block pool |
| --- | --- | --- | --- | --- |
| A | 0.076 | 0.018 | 0.267 | 0.061 |
| B | 0.063 | 0.035 | 0.126 | 0.056 |
| C | 0.055 | 0.058 | 0.021 | 0.056 |
| D | 0.077 | 0.037 | 0.107 | 0.067 |
| Ridge | 0.052 | 0.056 | 0.040 | 0.053 |
| GRU | 0.050 | 0.055 | 0.024 | 0.052 |
| TCN | 0.052 | 0.041 | 0.092 | 0.049 |

Six secondary tests are reported without multiplicity correction; none is a confirmation. The pooled IC standard error over 15 blocks is roughly 0.016, so the pipelines are not separable from each other. What the pool does suggest is that a small positive IC in the 0.05–0.06 range is common to all pipelines; what failed was the claim that A in particular is better.

## 3. Post-hoc diagnostic: how much of A's selection score survives?

Script: `python -m src.research.posthoc_selection_shrinkage` (reads only committed reports; output [selection_shrinkage.csv](phase6_posthoc/selection_shrinkage.csv)). For each research fold it compares the inner score of the 16 selected raw levels with the same features' Phase 3 outer-fold Rank IC under the selected sign, against eligible same-sign features that were not selected.

| Group (11 folds pooled) | Feature-folds | Inner oriented IC | Outer oriented IC | Outer > 0 | Retained |
| --- | --- | --- | --- | --- | --- |
| Selected (Top-16) | 176 | 0.093 | 0.045 | 74% | 49% |
| Eligible, not selected | 406 | 0.046 | 0.025 | 65% | 54% |

Reading:

- **Winner's curse is large.** About half of the inner score of selected features disappears one fold later. Ranking on two 252-day blocks selects partly on noise.
- **Selection still carries information.** Selected features keep a higher outer IC than unselected ones (0.045 vs 0.025), and the inner-to-outer rank correlation is positive in 10 of 11 folds.
- **Aggregation helps.** A's equal-weight composite reached 0.076 in research, above the 0.045 of its average member, as expected from averaging weakly correlated signals.
- The holdout drop (0.076 → 0.018) is larger than this within-research shrinkage. With an SE near 0.03 over four blocks it is compatible both with ordinary noise and with decay of the specific features chosen at 7858; the data cannot separate the two.

This diagnostic is post hoc and changes no frozen result.

## 4. What follows

- **Research track** stays exactly as frozen: A + `volscaled_k0.10`, refit on 0–8866 and run once on the Final Simulation, whatever the holdout said.
- **Competition track** (Protocol v5) is a separate, disclosed choice made with all 15 out-of-sample blocks, frozen before the Final Simulation is read. Both tracks open the Final Simulation once, together. Its 180 rows are too few to rank the two tracks; they complete the process and are not used to choose between them.
