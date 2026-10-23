# Protocol v5 — Competition Track and Final Simulation

State: **FROZEN before any v5 score and before any Final Simulation row is read.** Freeze commit: the commit adding `configs/protocol_v5_lock.json`; annotated tag `protocol-v5`. Verify with `python -m src.validation.protocol_v5 --check`.
Machine authority: [protocol_v5.yaml](../configs/protocol_v5.yaml). Parents v3.1, [v4](protocol_v4_phase6_allocation.md) and [v4.1](protocol_v4_1_holdout_amendment.md) are unchanged.

## 1. Why a second track

The research track answered its pre-registered question: A was not confirmed on the Internal Holdout, and its allocation was indistinguishable from the market ([results](../reports/phase6_holdout_results.md)). That answer stands and is not revised.

A competition submission is a different task: build the best system from **all** evidence available before the Final Simulation. That evidence now includes the opened holdout. Using it is legitimate for building a submission, but a choice made with it cannot be called confirmed by it. v5 therefore defines a separate track:

| | Research track | Competition track |
| --- | --- | --- |
| Rules | v3.1 / v4 / v4.1, unchanged | v5 (this document) |
| Forecaster / mapping | A, `volscaled_k0.10`, μ_A and σ_A from research OOS | chosen by §5 on 15 OOS blocks |
| Evidence used to choose | research rows 5088–7859 | research + holdout, 5088–8867 |
| Final Simulation | run once | run once, in the same run |

**Final Simulation role.** Its 180 rows have an official-score standard error of roughly 1 on a Sharpe-like scale. They complete the process, are reported for both tracks, and are **not** used to say which track or system is better. Nothing feeds back from them.

**Disclosure.** v5 is written after the holdout was opened. The forecaster set in §3 and the DynRidge design in §4 were chosen knowing that C, Ridge and GRU did better than A on the holdout, and that a recency-weighted model might adapt to decay. This is a known garden of forking paths; it is the reason v5 results are labelled "competition track" and never "confirmed".

## 2. Evidence and update schedule

- Out-of-sample rows **5088–8867** (3780 days) in **15 blocks of 252**: the 11 research outer validation blocks and the 4 holdout blocks.
- Research forecasts come from the sealed Phase 5 predictions (refit at each fold end, frozen for 252 rows). Holdout forecasts come from the sealed holdout forecasts (refit at 7858, purge 7859, frozen for 1008 rows; byte hash fixed in the config). DynRidge is computed on the same schedule.
- Scoring uses `forward_returns` and `risk_free_rate` read through a bounded loader of exactly 8868 rows. The Final Simulation rows are never parsed by the selection run.
- The holdout block is refit less often than research blocks (one refit for 1008 rows). The Final Simulation uses one refit for 180 rows, which is closer to the research schedule; no schedule is changed to favour any candidate.

## 3. Forecasters

{A, Ridge, DynRidge, C, D}. A is the frozen primary; D was near-best in research; C and Ridge were the strongest pre-existing secondaries on the holdout; DynRidge is new (§4). B, GRU and TCN are excluded: B is the more complex sibling of A, GRU and TCN are the components of C. No other forecaster is admitted.

## 4. DynRidge (fully specified before any fit)

At reference end e (training rows 0..e, labels after e physically removed):

1. **Inputs:** the 97 raw levels (94 anonymous features and the 3 official lagged fields), restricted to those passing the v3.1 availability rules on rows 0..e.
2. **Scaling:** training mean and std (ddof 0); missing → 0 after scaling; clip to ±5. The target is standardized by its training mean and std (ddof 0).
3. **Weights:** w_i = 0.5^((e − i)/504) for training rows with a finite label (half-life two years).
4. **Fit:** minimize Σ w_i (y_i − b − x_iβ)² / Σ w_i + α‖β‖², intercept b unpenalized, solved in float64 by weighted normal equations.
5. **α ∈ {0.1, 1, 10, 100}:** chosen at each reference by the mean Rank IC over the two inner 252-row blocks in the last 504 training rows (purge 1), each forecast by a DynRidge fitted on its own earlier prefix. Ties → larger α. No finite score → no forecast.
6. **Forecast:** (b + x_tβ)·σ_y + μ_y from X_t only, frozen until the next scheduled refit.

## 5. Allocation candidates and selection

Standardization: z_t = (s_t − μ_f)/σ_f with μ_f, σ_f (ddof 1) the mean and std of forecaster f's 3780 OOS forecasts; no target enters; frozen for the Final Simulation. The volatility state c_t is v4 §4 unchanged.

Candidates:
- forecast-free: `constant_1` (w = 1), `volatility_only` (w = clip(c_t, 0, 2));
- forecast-based: `volscaled_k{k}/{f}` with w = clip(c_t (1 + k z_t), 0, 2), k ∈ {0.10, 0.20}, f in §3 (10 candidates). The `linear` family is dropped because no linear mapping beat `constant_1` on the holdout; this is disclosed, not tested.

Rule, scored with the unmodified official metric over all 3780 rows and within each of the 15 blocks:

1. **Default.** `volatility_only` if its overall score is strictly above `constant_1` and it is above `constant_1` in at least **8 of 15** blocks; otherwise `constant_1`.
2. **Admissible.** A forecast-based candidate whose overall score is strictly above the default and which is above the default in at least 8 of 15 blocks.
3. **Near-best.** Admissible candidates within **0.05** (strict) of the best admissible score.
4. **Choice.** The first near-best candidate in the simplicity order: k = 0.10 before 0.20; within k, A, Ridge, DynRidge, C, D. If nothing is admissible, the default is chosen.

Costs (0, 1, 5 bp one-way) and moving-block bootstrap intervals (20, 2000, seed 42) are reported and never change the choice.

## 6. Final Simulation (8868–9047): one run, both tracks

Requires the v5 selection to be completed, sealed and committed, and separate authorization.

1. Load all 9048 rows once. Refit every needed forecaster on **0–8866** (purge 8867; labels from 8867 on are removed) exactly as its frozen rule specifies; A, Ridge, C, D follow v3.1 (D with its two inner references 8361 and 8613), DynRidge §4.
2. Forecast 8868–9047 with frozen state, using at each t only X up to t and the official lagged fields. Seal the forecasts before any Final outcome is attached.
3. Research track: A, μ_A and σ_A from v4.1, `volscaled_k0.10`. Competition track: the v5 selection with its frozen μ_f, σ_f (no forecast if the default was chosen).
4. Report for both tracks and for `constant_1` and `volatility_only`: official score, cost sensitivity, mean position, turnover, and descriptively the 180-row Rank IC of each forecast used. No hypothesis test, ranking or winner is declared.

## 7. Forbidden adaptation

No forecaster, k, mapping, weight, half-life, α grid, block rule or margin may be added or changed after any v5 score is computed. Final outcomes change nothing. Technical failures follow the incident discipline: stop, record, fix code without changing rules, re-freeze, disclose.
