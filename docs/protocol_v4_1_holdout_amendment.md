# Protocol v4.1 — Holdout Amendment

State: **FROZEN before any Internal Holdout row is read.** Tag `protocol-v4.1`; lock `configs/protocol_v4_1_lock.json`; machine authority [protocol_v4_1.yaml](../configs/protocol_v4_1.yaml).
Parent: [Protocol v4](protocol_v4_phase6_allocation.md), unchanged (tag `protocol-v4`).

## Why this amendment

Phase 6A ([sealed selection](../reports/phase6_selection/)) chose **`volscaled_k0.10`** under the frozen v4 rule (official score 0.968 versus 0.864 for `constant_1` on the 2772 research OOS rows). A post-hoc, descriptive decomposition on the same research rows found that a volatility-only position clip(c_t, 0, 2) scores 0.934: about two thirds of the improvement comes from the pre-registered volatility state rather than from A's forecast. v4 registered no holdout comparator that separates the two. This amendment adds that comparator and fixes how the holdout will be interpreted. It is written before any holdout row is read and changes no v4 rule, selection or parameter.

## Additions

1. **New holdout secondary `volatility_only`:** w_t = clip(c_t, 0, 2), with the identical frozen volatility state of v4 §4. It is a comparator only and can never become the primary allocation.
2. **Paired differences:** official score of `volscaled_k0.10` minus `volatility_only`, and minus `constant_1`, over the same 1008 days. Report each with a paired moving-block bootstrap interval (block 20, 2000 draws, seed 42, 95%), resampling the same day indices for both positions. Descriptive only; not a gate.
3. **Forecast confirmation of A (primary evidence of forecast value):** block-wise Spearman Rank IC of A over the four 252-row holdout blocks, with the Phase 3 inference (Newey–West HAC lag 5 on the block-ranked series, normal approximation). Pre-registered one-sided test **H0: IC ≤ 0, α = 0.05**. Report the estimate, HAC standard error, one-sided p-value, 95% normal interval and each block's IC.
4. **Interpretation fixed in advance:**
   - Whether A carries forecast information out of sample is decided **only** by item 3.
   - Allocation scores and their rankings over about four years are economic description. A Sharpe-like score over 1008 days has a standard error of roughly 0.5, larger than the research-period increment attributable to A (about 0.03), so a ranking of `volscaled_k0.10`, `volatility_only` and `constant_1` cannot by itself confirm or refute A.
   - Secondary forecasts (B, C, D, Ridge, GRU, TCN) and secondary mappings never replace the primary, whatever their holdout results.

## Unchanged

Primary forecast A; the frozen μ_A = −0.0391475, σ_A = 0.2059859; the selected mapping `volscaled_k0.10`; the volatility state; the A refit on 0–7858 with purge 7859 and frozen state through 7860–8867; one holdout run; and the Final Simulation procedure of v4 §9. Secondary forecasts use their own research-OOS mean and standard deviation from the sealed Phase 5 predictions with the selected mapping.

Protocol v4.1 was frozen before any Internal Holdout observation was read.
