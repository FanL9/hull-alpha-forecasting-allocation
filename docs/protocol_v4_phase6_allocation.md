# Protocol v4 — Phase 6 Forecast-to-Allocation

State: **FROZEN before any Phase 6 allocation score.** Freeze commit: the commit adding `configs/protocol_v4_lock.json`; annotated tag `protocol-v4`. Verify with `python -m src.validation.protocol_v4 --check`.
Machine authority after freeze: [protocol_v4.yaml](../configs/protocol_v4.yaml). Parents: [Phase 2.5-v2](research_protocol.md), [v3](protocol_v3_engineered_and_deep_alpha.md), [v3.1](protocol_v3_1_engineered_alpha_fair_comparison.md); none is modified.

## 1. Question and starting point

Phases 3–5 asked whether the features forecast next-day excess returns. Phase 5 selected **A** under the frozen v3 rule (mean fold Rank IC 0.0765, 10/11 positive folds; near-best set {A, D}, simplest A). Phase 6 asks a different question:

> Given the frozen A forecast, how much S&P 500 exposure in [0, 2] should be held each day, judged by the unmodified official Hull score?

Phase 6 does not reselect features, retrain alternatives or revisit A versus B/C/D. It maps a fixed forecast to a position.

**What has already been observed.** While drafting, the location and scale of A's 2772 research out-of-sample forecasts (date_id 5088–7859) were read: mean −0.0391, std 0.2060, range [−0.573, 0.616]. These are forecast values only; no official score, return or target statistic of any mapping has been computed. Phase 3–5 IC results are known and disclosed.

## 2. Data and timing

- **Calibration rows:** A's existing research OOS forecasts, date_id **5088–7859** (2772 rows, 11 blocks of 252). Each forecast was produced by a selector fitted only on earlier, purged rows. Phase 6 never refits A on these rows.
- **Scoring fields:** `forward_returns` and `risk_free_rate` for the same rows, read only through the bounded research loader (`load_research_prefix` with the train header), never beyond 7859.
- **Decision timing:** position w_t is chosen at t from the A forecast for t (which uses X_t) and from realized `lagged_forward_returns` up to t; it earns `forward_returns[t]`. No later row enters w_t.
- The forecast target (`market_forward_excess_returns`) is not scored; the official score uses allocations, `forward_returns` and `risk_free_rate`.

## 3. Signal standardization

z_t = (s_t − μ_A) / σ_A, where s_t is A's forecast and μ_A, σ_A (ddof = 1) are the mean and standard deviation of the 2772 research OOS forecasts. They contain no target information and are **frozen** for the Internal Holdout and Final Simulation. Without centering, the negative mean forecast would hold average exposure below 1 and trigger the official underperformance penalty for reasons unrelated to timing. A missing forecast maps to position 1.0.

## 4. Volatility state

c_t = clip(σ_long,t / σ_short,t, 0.5, 1.5), with σ_short the 21-row and σ_long the 252-row trailing sample std (ddof = 1, minimum 15 and 126 observations) of `lagged_forward_returns` up to and including t. Undefined → c_t = 1. It uses only returns already realized at t (volatility timing in the spirit of Moreira & Muir, 2017). Its windows and clip bounds are fixed here and not searched.

## 5. Registered candidates

| ID | Position w_t | k values |
| --- | --- | --- |
| `constant_1` (baseline) | 1.0 | — |
| `linear_k{k}` | clip(1 + k·z_t, 0, 2) | 0.10, 0.20, 0.35, 0.50, 0.75 |
| `volscaled_k{k}` | clip(c_t·(1 + k·z_t), 0, 2) | 0.10, 0.20, 0.35, 0.50, 0.75 |

Ten candidates plus the baseline; no other functional form, k value, window, threshold or smoothing is admitted. With A's IC near 0.08, small k keeps exposure mostly within about 0.6–1.4; the official score penalizes strategy volatility above 1.2× market volatility and underperformance of the market, so larger k is not presumed better.

## 6. Selection on research OOS rows

Score every candidate and the baseline with the unmodified official metric (`src/evaluation/kaggle_metric.py`) over 5088–7859, and separately within each of the 11 blocks.

1. **Admissible:** overall score strictly above `constant_1` **and** a higher score than `constant_1` in at least **6 of 11** blocks.
2. Let M be the best admissible overall score. The **near-best set** contains admissible candidates with M − score **< 0.05** (strict).
3. Choose the first near-best candidate in the fixed **simplicity order**: all `linear` before all `volscaled`; within each, smaller k first.
4. If no candidate is admissible, the selected allocation is **`constant_1`**: the forecast is then reported as not converted into an allocation advantage on research rows. No rescue mapping is added.

Selection uses no Internal Holdout or Final row. The 0.05 margin acknowledges that a Sharpe-like score estimated over about 11 years has a standard error of roughly 0.3; ranking differences within the margin are treated as indistinguishable. Block-bootstrap intervals of each score (block 20, 2000 draws, seed 42) are reported descriptively and never change the choice.

## 7. Costs

The official score has no transaction costs and is reported unmodified. Turnover Σ|w_t − w_(t−1)| and score sensitivity at **0, 1 and 5 bp** one-way cost per unit of exposure change are reported separately for every candidate. Costs never change the selection.

## 8. Internal Holdout (7860–8867): one run

Requires: protocol-v4 frozen, Phase 6 implementation frozen at its own tag, the research selection sealed and committed, and separate explicit authorization.

1. Refit A exactly as v3 §14 specifies: fit on **0–7858**, purge **7859**, freeze all learned state (selected features, signs, maps).
2. Forecast 7860–8867 with frozen state; online state may use only current X, official lagged fields and the trailing volatility state.
3. Apply the frozen μ_A, σ_A and the selected mapping; never recompute them on holdout rows.
4. **Primary endpoints:** (a) A's block-wise Rank IC over the 1008 rows; (b) official score of the selected allocation versus `constant_1`.
5. **Secondary reports:** all 10 mappings, `constant_1`, and B, C, D, Ridge, GRU, TCN forecasts refit under their frozen v3 rules (their allocations use the selected mapping with their own research-OOS μ, σ). A better secondary **never replaces** the primary.

The holdout is a prospective reservation: it never informed any predictive selection, but Phase 2 descriptive EDA covered Development 0–8867, as disclosed in v3. It must not be described as never seen.

## 9. Final Simulation (8868–9047): one run

After the holdout is completed and sealed, and with separate authorization:

- Refit A on **0–8866**, purge **8867**; keep the holdout-frozen mapping and the unchanged μ_A, σ_A.
- Process rows **sequentially**, using at each t only current X and official lagged fields, as in the competition API.
- Report the official score of the allocation and of `constant_1`. No parameter, mapping or model may be changed after holdout results; Final results feed back into nothing.

## 10. Forbidden adaptation

No new mapping, k, window, clip, smoothing, regime rule, cost-aware selection or ensemble may be added after any Phase 6 score is seen. Holdout or Final outcomes cannot change μ_A, σ_A, the mapping, A's state or the reported primary. Technical failures follow the incident discipline of Phase 4 v1: stop, record, fix the code without altering rules, re-freeze, and disclose.

Protocol v4 was frozen before any Phase 6 allocation score, return statistic or holdout/final observation was computed.
