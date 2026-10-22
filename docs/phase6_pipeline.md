# Phase 6A: allocation selection pipeline

Authority: [Protocol v4](protocol_v4_phase6_allocation.md) (lock `configs/protocol_v4_lock.json`, tag `protocol-v4`).
This stage selects the forecast-to-position mapping on A's existing research out-of-sample forecasts
(date_id 5088–7859). It does not retrain A, open the Internal Holdout or run the Final Simulation.

## Implementation

- `src/research/phase6_allocation.py`: standardization from research OOS forecasts only, causal
  volatility state from realized lagged returns, the ten registered mappings plus `constant_1`,
  the unmodified official score, cost sensitivity (equal to the official arithmetic at 0 bp),
  per-block scores, descriptive block-bootstrap intervals and the frozen selection rule.
- `src/research/phase6.py`: read-only preflight, synthetic mode and the single real selection entry.

## Commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase6_allocation.py tests/test_protocol_v4.py
.\.venv\Scripts\python.exe -m src.research.phase6 --synthetic --output work/phase6/example
.\.venv\Scripts\python.exe -m src.research.phase6 --check
.\.venv\Scripts\python.exe -m src.research.phase6 --run-selection
```

The real entry requires a clean checkout at the annotated tag `phase6-allocation-pipeline-v1`, the
verified v4 lock (which binds the sealed Phase 5 predictions) and an unused canonical state
`artifacts/phase6_selection_state.json`. It claims that state atomically before any data access,
verifies data hashes, reads the research prefix only through the bounded loader (`nrows=7860`) and
writes `artifacts/phase6_selection/`. A failed or interrupted attempt keeps the reservation; there is
no rerun switch. Seal the outputs into `reports/` and commit before any holdout work.

## Tests (synthetic only)

Mapping arithmetic and clipping, missing forecast → 1.0, causal volatility state, zero-cost score equal
to the official metric, admissibility (overall and ≥6/11 blocks), strict 0.05 near-best margin,
simplicity order, `constant_1` fallback, range checks, outputs/manifest, no overwrite, preflight
refusals, failed-run reservation and an integration test with the unpatched bounded loader.

The Internal Holdout execution (A refit on 0–7858 and frozen evaluation on 7860–8867) is a separate
implementation step with its own freeze, after the research selection is sealed.
