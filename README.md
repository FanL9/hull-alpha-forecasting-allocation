# Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation

This project uses official [Kaggle Hull Tactical Market Prediction data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data) to study whether anonymous financial features can predict future S&P 500 excess returns.
Later stages will turn forecasts into allocations, using strict chronological validation and a Kaggle-style competition simulation.

## Data

- `train.csv`: **9,048 rows × 98 columns**, including **94 anonymous features**.
- Feature families: **M, E, I, P, V, S, D**.
- Forecasting target: `market_forward_excess_returns`.
- Official `test.csv` is used only for interface/schema testing, not as an independent research test set.

## Data Split

| Area | `date_id` | Rows |
| --- | --- | --- |
| Development | 0–8867 | 8,868 |
| Competition Simulation | 8868–9047 | 180 |

The final **180 rows are locked** for a future Kaggle-style simulation and excluded from development analysis and decisions.

## Current Status

- **Phase 1 complete:** official data acquisition, SHA-256 verification and data audit.
- **Phase 2 complete:** data processing and EDA.
- **Phase 2.5 complete:** labels verified, official metric implemented and research protocol frozen at **phase2.5-v2** (primary Alpha metric: block-wise Rank IC). See the [research protocol](docs/research_protocol.md).
- **Phase 3 Alpha Research: not started.**
- **Not started:** model training, Signal, Allocation and Backtest.

Development EDA found missing values in **85 features**, with clear temporal structure. Default processing preserves NaNs; no predictive alpha or profitable strategy has been established.

## Workflow

```text
Official Data
→ Data Audit & EDA
→ Alpha Research
→ Forecasting Model
→ Signal & Allocation
→ 180-day Competition Simulation
```

## Repository Structure

```text
configs/
data/
docs/
notebooks/
src/
tests/
reports/
artifacts/
```

## Quick Start

Requires Python 3.12+, existing official data and preserved Phase 1 audit artifacts. Run from the repository root (Windows PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli phase1
.\.venv\Scripts\python.exe -m src.cli phase2
.\.venv\Scripts\python.exe -m pytest
```

Keep raw data, generated artifacts and credentials out of Git.

## Documentation

[Methodology](docs/methodology.md) · [Research protocol](docs/research_protocol.md) · [Official metric](docs/official_metric_notes.md) · [Research questions](docs/research_questions.md) · [Validation design](docs/validation_design.md) · [Data dictionary](docs/data_dictionary.md)
