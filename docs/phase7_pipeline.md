# Phase 7 Pipeline — v5 Selection and Final Simulation

Rules: [Protocol v5](protocol_v5_competition_track.md). Code: `src/models/dynamic_ridge.py`, `src/research/phase7_selection.py`, `src/research/phase7_refit.py`, `src/research/phase7_final.py`. Tests: `tests/test_phase7_selection.py`, `tests/test_phase7_final.py` (artificial data only).

## What each run reads

| Run | Market rows parsed | Forecast evidence | Writes |
| --- | --- | --- | --- |
| v5 selection | 0–8867 (`nrows=8868`) | committed Phase 5 research forecasts, committed holdout forecasts; DynRidge computed | `artifacts/phase7_selection/` |
| Final Simulation | 0–9047 | committed `reports/phase7_selection/` | `artifacts/phase7_final/` |

No selection output contains a daily return; every file in `artifacts/phase7_selection/` may be published. The Final run seals its forecasts (`final_forecasts_sealed.csv`) before any Final outcome is attached.

## Engineering notes

- `RefitEngine` generalizes the frozen `HoldoutEngine` to any single refit. The Phase 3–5 helpers guard date_id 7859; `phase7_refit` carries copies that take the bound from the engine, and moves only the protocol's reference guard (`research.end`) to the engine's purge row. Tests show the copies equal the frozen helpers inside research, and that `RefitEngine(7858, 7860, 8868)` reproduces `HoldoutEngine` forecasts and neural capsules exactly.
- Final forecasts at t depend on no row after t (test changes every column from 8958 on and compares 8868–8957), and no fitted state depends on any outcome from 8867 on.
- Only the forecasters the two tracks need are refit: A, plus the competition forecaster. GRU/TCN run only if C or D was selected; D adds its inner references 8361 and 8613.

## Local commands (real data, Windows `.venv`)

```powershell
git switch main; git pull
python -m src.validation.protocol_v5 --check
python -m pytest -q tests/test_protocol_v5.py tests/test_phase7_selection.py tests/test_phase7_final.py

# 1. Tags (cloud cannot push tags)
git tag -a protocol-v5 -m "Protocol v5 frozen before v5 selection and Final Simulation"
git tag -a phase7-pipeline-v1 -m "Phase 7 v5 selection and Final Simulation pipeline"
git push origin protocol-v5 phase7-pipeline-v1

# 2. v5 selection: once
python -m src.research.phase7_selection --check
python -m src.research.phase7_selection --run-selection

# 3. Publish the selection (no daily returns inside), commit, push
New-Item -ItemType Directory reports/phase7_selection
Copy-Item artifacts/phase7_selection/* reports/phase7_selection/
Copy-Item artifacts/phase7_selection_state.json reports/phase7_selection/
git add reports/phase7_selection; git commit -m "Seal Protocol v5 competition-track selection"; git push

# 4. Final Simulation: once, after explicit authorization
python -m src.research.phase7_final --check
python -m src.research.phase7_final --run-final --workers 4
```

After step 3 the checkout differs from the tag only under `reports/phase7_selection/`, which is the only change the Final preflight accepts. Do not commit anything else between steps 3 and 4.
