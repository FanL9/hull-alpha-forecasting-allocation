# Project Flow

The execution order from official data to the Final Simulation, with each step's method, governing protocol and result. Every step is **complete**.

```text
date_id  0 ───────── 5087 │ 5088 ───── 7859 │ 7860 ───── 8867 │ 8868 ── 9047
         first training     11 × 252-day       Internal Holdout   Final Simulation
         window             walk-forward OOS   (opened once)      (opened once)
```

```text
Data ─► Protocol ─► Alpha research ─► Feature engineering ─► Models ─► Allocation ─► Holdout ─► Competition track ─► Final
 1-2       2.5           3                    4                  5          6A           6B            7                 7
```

## Working rules used throughout

1. **Freeze before scoring.** Each step has a protocol (YAML + document) locked by SHA-256 and an annotated Git tag. The formal run checks the lock, a clean checkout at its tag, and an unused single-use state file.
2. **Synthetic first.** Each pipeline is calibrated and tested on artificial data with known signal and known noise. Only then is it run once on real data.
3. **No look-ahead.** Training uses rows ≤ t only. A 1-day purge separates training from validation. Forecasts are sealed before outcomes are attached. Tests change future rows and check that past forecasts do not move.
4. **Report everything.** Failures become incident reports, post-hoc analyses are labelled post hoc, and deviations from plan are disclosed.

## Steps

### 1–2. Data acquisition, audit and descriptive EDA
- Official Kaggle files are downloaded and verified by SHA-256; raw data stay local.
- The audit covers schema, ordering, duplicates, missingness and families (M, E, I, P, V, S, D).
- EDA covers distributions, drift, missing intervals and X–X correlation, without any X–y testing. It used rows 0–8867, which is disclosed.
- Results: [data audit](../reports/data_audit.md), [EDA](../reports/phase2_data_processing_eda.md).

### 2.5. Research protocol (phase2.5-v2)
- Chronological split: research 0–7859, holdout 7860–8867, final 8868–9047.
- Walk-forward design: first training window 0–5087, then 11 validation blocks of 252 days, expanding windows, purge 1.
- Feature availability rules are fitted on training rows only.
- Primary metric: block-wise Spearman Rank IC. Secondary: winsorized Pearson IC.
- Inference: Newey–West HAC p-values, moving-block bootstrap intervals, Benjamini–Hochberg FDR per hypothesis family, BY sensitivity.
- Document: [research protocol](research_protocol.md); the official metric is reimplemented and tested ([notes](official_metric_notes.md)).

### 3. Single-feature alpha research
- 97 single features (94 anonymous + 3 official lagged returns) and 7 family composites. Each one is oriented and checked for eligibility on training rows, then scored on the next block.
- Result: 24/97 features and 4/7 families pass BH-FDR 0.05 ([results](../reports/phase3_alpha_research_results.md)).
- Post-hoc timing checks (lag 0–252 decay, ΔX signals) found no sign of a one-day shift or slow look-ahead (they cannot certify release times) ([interpretation](../reports/phase3_posthoc_interpretation.md)).

### 4. Engineered alpha (Protocol v3.1)
- **A (raw control):** within each fold, two inner blocks rank the 97 raw levels by mean inner Rank IC (same sign in both blocks required). The selector prunes pairs with |ρ| ≥ 0.90, keeps the top 16, maps each through its training ECDF and averages them with equal weights.
- **B (engineered):** the same selector over 499 candidates (deltas, z-scores, ranks, family composites, high-volatility interactions).
- Result: B did not beat A (mean IC difference −0.014, lost 7 of 11 folds) ([results](../reports/phase4_engineered_alpha_results.md)).

### 5. Forecasting models (Protocol v3.1)
- The models are:
  - Ridge, GRU and TCN on 63-day windows of 16 cluster-representative inputs;
  - C, the average of GRU and TCN;
  - D, an equal-weight combination of normalized A, B and C.
- Hyperparameters are chosen on inner blocks only, with early stopping on a fixed block and three seeds per network.
- Frozen rule: among A, B, C and D, take the best mean fold IC; when the difference is below 0.01, take the simplest. Result: **A** (IC 0.077; D 0.077, B 0.063, C 0.055, Ridge 0.052) ([analysis](../reports/phase5_real_analysis.md)).

### 6A. Forecast to allocation (Protocol v4)
- z = (forecast − μ)/σ, with μ and σ taken from research out-of-sample forecasts only.
- c is a trailing volatility ratio, σ₂₅₂/σ₂₁ of lagged returns, clipped to [0.5, 1.5].
- Candidates: linear clip(1 + kz, 0, 2) and volatility-scaled clip(c(1 + kz), 0, 2), each with 5 values of k.
- Rule: beat buy-and-hold overall and in at least 6 of 11 blocks; among candidates within 0.05 of the best, take the simplest.
- Result: `volscaled_k0.10` (0.968 vs 0.864). The forecast-only mapping scored 0.866 ([selection](../reports/phase6_selection/)).

### 6B. Internal Holdout (Protocol v4.1)
- All models were refit on 0–7858, frozen, and run once on 7860–8867.
- Pre-registered test: A's IC > 0, one-sided, α = 0.05.
- Result: **not confirmed** (IC 0.018, p = 0.27). The allocation was indistinguishable from buy-and-hold and from a volatility-only rule ([results](../reports/phase6_holdout_results.md)).
- A post-hoc check found that selected features lose about half of their inner score one fold later ([diagnostic](../reports/phase6_posthoc/selection_shrinkage.csv)).

### 7. Competition track and Final Simulation (Protocol v5)
- **Research track:** the pre-registered system, kept unchanged.
- **Competition track:** chosen with all 15 out-of-sample blocks (research plus holdout). This is disclosed as designed after the holdout. The candidates were A, Ridge, C, D and a new recency-weighted ridge (DynRidge), each with k ∈ {0.10, 0.20}, plus two forecast-free defaults. The same rule then picked the simplest near-best: **A + `volscaled_k0.10`** ([selection](../reports/phase7_selection/)).
- **Final Simulation:** A was refit on 0–8866 and frozen. Rows 8868–9047 were processed in order, using only current features and official lagged fields; forecasts were sealed before outcomes were attached.
- Final result: both tracks scored 1.48, buy-and-hold 1.13, volatility-only 1.48 ([final files](../reports/phase7_final/), [report](../reports/final_project_report.md)).

## Where each step lives

| Step | Protocol | Pipeline note | Code |
| --- | --- | --- | --- |
| 3 | [phase2.5-v2](research_protocol.md) | [phase3](phase3_alpha_research.md) | `src/research/phase3*.py`, `alpha_pipeline.py` |
| 4–5 | [v3](protocol_v3_engineered_and_deep_alpha.md), [v3.1](protocol_v3_1_engineered_alpha_fair_comparison.md) | [phase4](phase4_pipeline.md), [phase5](phase5_pipeline.md) | `phase4_engine.py`, `phase5_engine.py`, `src/models/deep_alpha.py` |
| 6 | [v4](protocol_v4_phase6_allocation.md), [v4.1](protocol_v4_1_holdout_amendment.md) | [6A](phase6_pipeline.md), [6B](phase6_holdout_pipeline.md) | `phase6_allocation.py`, `phase6_holdout.py` |
| 7 | [v5](protocol_v5_competition_track.md) | [phase7](phase7_pipeline.md) | `phase7_selection.py`, `phase7_refit.py`, `phase7_final.py`, `src/models/dynamic_ridge.py` |
