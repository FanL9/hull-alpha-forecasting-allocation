# Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation

This project uses official [Kaggle Hull Tactical data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data) to study whether anonymous financial features predict future S&P 500 excess returns.
The planned pipeline converts forecasts into allocations under chronological validation and a final competition simulation.

## Data

- train.csv: **9048 rows × 98 columns**, including **94 anonymous features** in M, E, I, P, V, S and D.
- Primary target: market_forward_excess_returns.
- Official test.csv is for interface/schema checks, not an independent research test set.

## Chronological Split

| Area | date_id | Rows |
| --- | --- | --- |
| Research / Walk-Forward | 0–7859 | 7860 |
| Internal Holdout | 7860–8867 | 1008 |
| Competition Simulation | 8868–9047 | 180 |

The first two areas form Development; holdout outcomes do not guide selection, and the final 180 rows remain locked until simulation.

## Current Status

- **Done:** official data acquisition, hash verification, data audit and descriptive EDA.
- **Frozen:** research protocol **phase2.5-v2** (primary Alpha metric: block-wise Rank IC), official metric and validation rules. See the [Research Protocol](docs/research_protocol.md).
- **Phase 3:** frozen pipeline completed one real Alpha Research run; [sealed results and report](reports/phase3_alpha_research_results.md). Post-hoc timing diagnostics: [interpretation](reports/phase3_posthoc_interpretation.md).
- **Phase 4A:** [Raw-Control / Engineered pipeline](docs/phase4_pipeline.md) validated with synthetic data under frozen [Protocol v3.1](docs/protocol_v3_1_engineered_alpha_fair_comparison.md); [calibration report](reports/phase4_pipeline_synthetic_calibration.md). Real Phase 4 has not run; the v1 attempt failed before reading data and is remediated by pipeline v1.1 ([incident](reports/phase4_incidents/v1_real_entry_key_error.md)).
- **Not started:** model training, signals, allocation and backtesting.

## Pipeline

```text
Official Data → Audit & EDA → Protocol Freeze → Alpha Research
→ Forecasting → Internal Holdout → Signal & Allocation → Evaluation → Simulation
```

## Repository

```text
configs/  data/  docs/  notebooks/  src/  tests/  reports/  artifacts/
```

## Run

Python 3.12+, existing official data and acquisition artifacts are required; run from the repository root in Windows PowerShell.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli phase1
.\.venv\Scripts\python.exe -m src.cli phase2
.\.venv\Scripts\python.exe -m pytest
```

Keep raw data, generated artifacts and credentials out of Git; result snapshots are in [reports/](reports/).

## Read Next

[Project Flow](docs/project_flow.md) · [Methodology](docs/methodology.md) · [Research Protocol](docs/research_protocol.md) · [Data Dictionary](docs/data_dictionary.md)
