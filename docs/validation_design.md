# Research split design

Status: fixed data boundary implemented; future walk-forward remains design only. No folds have been generated or executed.

| Area | Inclusive date_id | Rows | Current permitted use |
| --- | --- | --- | --- |
| Development | 0..8867 | 8868 | Data processing and EDA |
| Competition Simulation | 8868..9047 | 180 | Headers/IDs/count metadata only |

The two areas partition all 9048 training rows, without a gap, overlap, sorting or shuffle. configs/split_policy.json is the tracked firewall. artifacts/split_definition.json persists source hashes, creation timestamp, exact boundaries and policies. A rerun verifies rather than overwrites a different reservation. The raw value parser reads only nrows=8868. Full-file operations are byte hashing and header/ID inspection.

The final 180 rows cannot inform missing-value handling, feature engineering, selection, EDA, regimes, model/tuning/allocation choices or backtest optimization. No simulation dtype inference, missingness or target summary is allowed. Shared train headers provide schema compatibility; value checks wait for a future authorized simulation. Official test.csv is inference-schema-only and never a model test set.

Future sequence, entirely unexecuted:

1. Agree on expanding-window walk-forward within Development Area.
2. Decide fold count, initial window, block lengths, release/label timing, purge rules, models and tuning procedures. All are currently undecided/null.
3. Fit preprocessing, feature selection, regime thresholds and models inside each historical training window only; evaluate strictly later rows. No random split.
4. Define an internal chronological holdout inside development; its length is not selected now.
5. Freeze the strategy, implementation and any sequential update schedule.
6. Separately authorize and run the 180-row Kaggle-style competition simulation under that frozen protocol.

No internal test or fold length is implied by the 252-row descriptive EDA windows. Target-only rolling summaries are retrospective observations, not inputs to future forecast-time features. An outcome-availability gap must be determined later from verified semantics; the fixed Development/Simulation partition itself has no gap.

The original Phase 1 full-file audit and withdrawn previous analyses predate this rebuild. Preserving them as history does not make the new simulation historically unseen. New work enforces a prospective firewall and discloses prior access. Final independent-performance claims will require acknowledging that history.
