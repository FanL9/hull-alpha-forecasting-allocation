# Final Project Report — Forecasting and Allocating the S&P 500 on Hull Tactical Data

Status: **complete.** All planned phases are finished, with one completed Internal Holdout evaluation and one completed Final Simulation. The full 9048-row history has now been used across research 0–7859, Internal Holdout 7860–8867 and Final Simulation 8868–9047. The recorded Phase 4 v1 pre-data failure preceded the completed v1.1 run.

## 1. Question

Can 94 anonymous daily features (plus three official lagged returns) forecast next-day S&P 500 excess returns? And can such a forecast earn a better official Hull score than holding the market? The score is Sharpe-like, with penalties for excess volatility and underperformance; positions lie in [0, 2].

## 2. Design

- **Chronology.** Research 0–7859 with 11 walk-forward folds of 252 days (purge 1). After that, a one-shot Internal Holdout (1008 days) and a one-shot Final Simulation (180 days).
- **Pre-registration.** Versioned protocols, hashed locks and annotated pipeline tags fixed rules before the corresponding formal runs. This does not mean the data had never been inspected: Phase 2 EDA covered rows through 8867, and v5 was explicitly designed after holdout evaluation. Real entry points checked clean, version-matched code and execution state; outputs were sealed by hash. Incidents were recorded rather than rerun silently ([Phase 4 v1](phase4_incidents/v1_real_entry_key_error.md)).
- **Primary statistic.** Block-wise Spearman Rank IC with Newey–West inference.

## 3. Results by phase

| Phase | Evidence | Result |
| --- | --- | --- |
| 3. Single features ([report](phase3_alpha_research_results.md)) | 97 features, 7 family composites, BH-FDR 0.05 | 24/97 features and 4/7 families (M, P, V, S) pass on research rows; these discoveries are not independent holdout confirmation |
| 4. Engineered universe ([report](phase4_engineered_alpha_results.md)) | 499 transformed candidates (B) vs raw-level control (A), same nested selector | B did not beat A: mean IC −0.014, lost 7/11 folds |
| 5. Nonlinear models ([analysis](phase5_real_analysis.md)) | A, B, C (GRU+TCN), D (stack), Ridge, GRU, TCN | Mean fold IC: A 0.077, D 0.077, B 0.063, C 0.055, Ridge 0.052. Frozen rule selected **A** |
| 6A. Allocation ([selection](phase6_selection/)) | 10 mappings scored on research OOS | `volscaled_k0.10` selected (0.968 vs constant 0.864). Forecast-only `linear_k0.10` scored 0.866: the lift came from volatility timing |
| 6B. Internal Holdout ([results](phase6_holdout_results.md)) | 1008 days, one run | **A not confirmed:** IC 0.018, p = 0.27. Allocation 0.436 vs market 0.426 vs volatility-only 0.470, all within noise |
| 7. v5 competition selection ([selection](phase7_selection/)) | all 15 OOS blocks, 12 candidates | Volatility-only failed the 8/15 block bar, so the default stayed `constant_1`. 5 forecast candidates admissible, all 0.81–0.86 with overlapping intervals; the rule chose the simplest, **A + volscaled_k0.10** |
| 7. Final Simulation ([results](phase7_final/)) | 180 days, one run, both tracks | See §4 |

## 4. Final Simulation (8868–9047)

A was refit once on 0–8866 (purge 8867) and frozen. Its forecasts were sealed before any Final outcome was attached.

| Allocation | Official score | 5 bp cost | Mean position | Turnover |
| --- | --- | --- | --- | --- |
| Research track (A, μ/σ from research OOS) | 1.481 | 1.446 | 1.08 | 7.4 |
| Competition track (A, μ/σ from 15 OOS blocks) | 1.482 | 1.449 | 1.08 | 7.0 |
| `volatility_only` | 1.479 | 1.463 | 1.17 | 3.5 |
| `constant_1` (market) | 1.126 | 1.126 | 1.00 | 0 |

- The two tracks ended with the same forecaster and mapping and differ only in standardization. Their daily positions correlate at 0.9999.
- Both beat the market over these 180 days. But `volatility_only`, which uses no forecast, scored the same: the gap is 0.002 before costs and goes the other way at 5 bp.
- A's Rank IC over the 180 days was 0.121, reported descriptively. Per Protocol v5 no confirmatory test is run on these rows, and the tracks are not ranked. This result does not overturn the failed pre-registered holdout confirmation.
- The score difference over these 180 days is descriptive. The frozen Final pipeline did not estimate score standard errors or establish statistical significance for that difference.

## 5. Conclusions

1. **Forecasting.** Research IC was around 0.05–0.08 across the model comparisons, without a stable incremental benefit from increased complexity. The pre-registered primary A did not confirm on the holdout. Its feature selection loses about half of its in-sample score one fold later ([post-hoc diagnostic](phase6_posthoc/selection_shrinkage.csv)).
2. **Complexity.** Engineered features (Phase 4) and neural networks (Phase 5) did not establish a robust improvement over the nested raw-feature composite. Its final refit selected 16 raw levels with equal absolute weights.
3. **Allocation.** The forecast-free volatility-only comparator accounts for much of the observed score improvement over the market. Final scores were nearly identical before costs; at 5 bp the volatility-only score was higher. These are descriptive comparisons, not a causal attribution or a new confirmation test.
4. **Process.** The competition track converged to the same forecaster and mapping as the research track, with different standardization. It remains a post-holdout exploratory choice; matching the earlier choice does not rule out selection bias or overfitting.

## 6. Limitations

- Anonymous features prevent any economic interpretation of the signals.
- One market, one history. 15 out-of-sample blocks give limited power to tell apart forecasts whose ICs differ by about 0.02.
- Phase 2 descriptive EDA covered rows through 8867, so the holdout was reserved prospectively rather than never seen.
- The competition-track forecaster set and DynRidge were designed after the holdout was opened (disclosed in [Protocol v5](../docs/protocol_v5_competition_track.md)).
- Transaction costs are sensitivity analyses only; the official metric excludes them.
