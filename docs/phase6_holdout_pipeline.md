# Phase 6B: Internal Holdout pipeline

Authority: [Protocol v4](protocol_v4_phase6_allocation.md) and [v4.1](protocol_v4_1_holdout_amendment.md).
One run on date_id **7860–8867**. All learned state is fitted on **0–7858** (purge 7859) and frozen.

## What it computes

- **Forecasts** (`src/research/phase6_holdout.py`): A and B refit with the frozen nested selector at 7858;
  Ridge/GRU/TCN refit with the frozen Phase 5 design (five seeds each); C = ½GRU + ½TCN; D = equal-weight
  z-average of A/B/C normalized on their inner OOF forecasts at the two inner references (7353, 7605).
  A `HoldoutEngine` reuses the Phase 5 engine, removes labels from 7859 onward from every fitting
  series, and uses a bounded holdout window function. Official lagged fields remain legal online X.
- **Forecast confirmation** (v4.1 §3): block-wise Spearman Rank IC over four 252-row blocks with
  Newey–West lag 5; one-sided p for H0: IC ≤ 0 at α = 0.05 for A; per-block ICs for every pipeline.
- **Allocation**: frozen μ_A, σ_A and `volscaled_k0.10` (primary); `constant_1`, `volatility_only` and
  all registered mappings (secondary); costs at 0/1/5 bp; block-bootstrap intervals; paired bootstrap
  differences of the primary minus `volatility_only` and minus `constant_1`; B/C/D/Ridge/GRU/TCN with the
  same mapping and their own research-OOS μ, σ.

## Execution and safety (`src/research/phase6_holdout_run.py`)

```powershell
.\.venv\Scripts\python.exe -m src.research.phase6_holdout_run --check
.\.venv\Scripts\python.exe -m src.research.phase6_holdout_run --run-holdout --workers 4
```

- Requires a clean checkout at the annotated tag `phase6-holdout-pipeline-v1`, the verified v4.1 lock
  (binding the sealed Phase 6A selection) and data-hash verification before loading.
- The loader reads exactly `nrows=8868`; the Final Simulation rows are never parsed.
- States: `reserved → forecasting → forecasts_sealed → completed`. Before sealing, a failed run
  (`failed_pre_forecast`) may resume with the same identity; targets are not attached yet and cached
  tasks are reused. After sealing, only the deterministic evaluation of the sealed forecasts may be
  replayed (`failed_post_forecast`); no model is refitted. `completed` refuses any further run.
- Outputs: `artifacts/phase6_holdout/` (sealed forecasts, metrics, allocation scores, daily positions,
  summary, fit audit, manifest). Seal them into `reports/` and commit after the run.

Expected runtime: 30 neural fits (3 references × GRU/TCN × 5 seeds, stop + refit), well under the
Phase 5 run. Tests use artificial data only (`tests/test_phase6_holdout.py`).
