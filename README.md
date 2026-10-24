# Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation

Can anonymous daily market features predict tomorrow's S&P 500 excess return? And if so, can that prediction be turned into a daily position that beats simply holding the index?

This project answers both questions on the official [Kaggle Hull Tactical](https://www.kaggle.com/competitions/hull-tactical-market-prediction) data (9048 trading days, 94 anonymous features). It is built like a small pre-registered study rather than a leaderboard chase:

- the data are split in time;
- every rule is written down and frozen before the data it governs are scored;
- each held-out period is opened exactly once;
- every result is reported, including the negative ones.

## Main findings

| Question | Answer |
| --- | --- |
| Do the features predict next-day excess returns? | **Weakly.** Out-of-sample Rank IC is about 0.05–0.08 in research for every model tried; 24 of 97 features pass multiple-testing control. |
| Did the pre-registered best model confirm on unseen data? | **No.** On the 1008-day Internal Holdout the primary forecaster A had IC 0.018 (one-sided p = 0.27). |
| Do more features or deep learning help? | **No.** 499 engineered features and GRU/TCN networks did not beat an equal-weight composite of 16 raw features. |
| Does the forecast improve the portfolio? | **Not detectably.** Scaling exposure by recent volatility explains almost all of the gain over buy-and-hold. The forecast adds little on top. |
| Final 180-day simulation (run once) | Selected system 1.48, buy-and-hold 1.13, volatility-only rule 1.48 (official Hull score). Descriptive, not a significance test. |

**In one sentence:** the features carry a real but small and unstable signal; under a fair, frozen evaluation, risk management (volatility timing) mattered more than return prediction.

The full write-up is in the [Final Project Report](reports/final_project_report.md).

## How the study is organised

```text
date_id  0 ─────────────── 5087 │ 5088 ──────── 7859 │ 7860 ──────── 8867 │ 8868 ── 9047
         first training window   11 walk-forward      Internal Holdout     Final Simulation
                                  validation blocks    (1008 days, once)    (180 days, once)
         └───────────── Research: all model and rule choices ───────────┘
```

| Phase | What was done | Result |
| --- | --- | --- |
| 1–2 Data | Official download with SHA-256 checks, audit, descriptive EDA (no X–y testing) | [Audit](reports/data_audit.md), [EDA](reports/phase2_data_processing_eda.md) |
| Protocol | Splits, purge, availability rules, metrics, inference and multiple-testing rules frozen in code | [Research protocol](docs/research_protocol.md) |
| 3 Single-feature alpha | 97 features + 7 family composites, walk-forward Rank IC, Newey–West, BH-FDR | 24/97 features, 4/7 families pass ([results](reports/phase3_alpha_research_results.md)) |
| 4 Engineered alpha | 499 transformed candidates vs raw levels, same nested selector | No improvement ([results](reports/phase4_engineered_alpha_results.md)) |
| 5 Models | Raw composite (A), engineered (B), GRU+TCN (C), A+B+C combination (D), Ridge | A selected, IC 0.077; differences within noise ([analysis](reports/phase5_real_analysis.md)) |
| 6 Allocation | Forecast → position in [0, 2]; 10 pre-registered mappings | Volatility-scaled mapping chosen ([selection](reports/phase6_selection/)) |
| 6 Holdout | One run on 7860–8867 | A not confirmed ([results](reports/phase6_holdout_results.md)) |
| 7 Final | Disclosed competition track chosen on all evidence; one Final run for both tracks | Both tracks chose the same system ([report](reports/final_project_report.md)) |

The [Project Flow](docs/project_flow.md) explains each step and the methods behind it.

## Methods at a glance

- **Validation:** expanding-window walk-forward, 11 folds of 252 days, 1-day purge. Nested inner blocks are used for any choice made inside a fold. No shuffling, and no information from after the forecast date.
- **Forecast metric:** block-wise Spearman Rank IC. Newey–West (HAC) standard errors, moving-block bootstrap intervals, Benjamini–Hochberg FDR across hypothesis families.
- **Forecasters:**
  - nested feature selection with redundancy pruning and equal weights (A);
  - an engineered-feature universe (B);
  - Ridge, GRU and TCN on 63-day sequences, with a GRU+TCN average (C) and a normalized A+B+C combination (D);
  - a recency-weighted ridge (DynRidge).
- **Allocation:** standardized forecast mapped to w = clip(c·(1 + k·z), 0, 2), where c is a trailing volatility ratio. Positions are scored with an exact reimplementation of the official Hull metric (a Sharpe-like score with volatility and underperformance penalties), with transaction-cost sensitivity reported separately.

## Research integrity

- **Freeze, then run.** Each phase has a versioned protocol (`configs/protocol_v*.yaml` + `docs/protocol_v*.md`) locked by file hashes and an annotated Git tag before its formal run. Run scripts refuse to start from a modified or untagged checkout.
- **One shot.** Each real run reserves a single-use state file; reruns are refused. The Internal Holdout and the Final Simulation were each evaluated once.
- **Sealed outputs.** Forecasts are hashed and sealed before outcomes are attached.
- **Disclosed deviations.** A pre-data software failure in Phase 4 was recorded, not hidden ([incident](reports/phase4_incidents/v1_real_entry_key_error.md)). The Phase 2 EDA covered rows up to 8867. The competition track (Protocol v5) was designed after the holdout was seen and is labelled as such.
- **Tested code.** About 560 automated tests run on synthetic data, covering look-ahead checks, loader boundaries, lock tampering and metric equivalence.

## Limitations

- The features are anonymous, so the signals cannot be interpreted economically.
- There is one market and one history; 15 out-of-sample years give limited power to separate forecasts whose ICs differ by about 0.02.
- The Final 180 days are short. Their scores describe one period and do not establish skill.
- The official metric has no transaction costs; costs (1–5 bp) are reported as sensitivity only.

## Repository guide

```text
configs/   frozen protocols, locks, split policy, metric settings
docs/      project flow, methodology, protocols, pipeline notes, data dictionary
src/       data, features, models, validation, statistics, research pipelines
tests/     synthetic-data tests (never read real data)
reports/   sealed results of every real run and the written reports
notebooks/ audit and EDA notebooks
```

Suggested reading order:
1. this README;
2. [Final Project Report](reports/final_project_report.md);
3. [Project Flow](docs/project_flow.md);
4. [Methodology](docs/methodology.md);
5. any phase report linked above.

## Reproduce

Python 3.12+, Windows PowerShell, from the repository root. Kaggle credentials and raw data stay local and are never committed.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase5.txt   # base requirements + PyTorch
.\.venv\Scripts\python.exe -m src.cli phase1      # official download + hash checks
.\.venv\Scripts\python.exe -m src.cli phase2      # audit and EDA
.\.venv\Scripts\python.exe -m pytest              # synthetic-data test suite
```

Protocol locks can be verified without data, for example `python -m src.validation.protocol_v5 --check`. Each phase's pipeline note in `docs/` lists its run commands. Real runs are single-use and already recorded in `reports/`.

Data: Hull Tactical competition data, used under the Kaggle competition rules. Code: see [LICENSE](LICENSE).
