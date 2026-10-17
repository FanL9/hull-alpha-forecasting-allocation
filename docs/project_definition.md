# Project definition

**Project:** `hull-alpha-forecasting-allocation`

**Title:** Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation

## Purpose

Build an independent, reproducible quantitative research project connecting anonymous financial predictors, expected excess-return forecasts, market regimes and dynamic S&P 500 exposure. The intended audience is research reviewers, quantitative finance / financial engineering graduate admissions and potential research collaborators.

The contribution is the research chain and its evidence: alpha identification → out-of-sample forecasts → causal signals → position sizing → portfolio utility. It does not claim to reproduce a Kaggle winning solution or to establish alpha by training one tree model.

## Source and research contract

- Use only [official Kaggle competition inputs](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data).
- Keep official inputs and derivative datasets out of Git.
- Preserve file bytes and original row ordering; record acquisition hashes.
- Never invent a dataset, column count, economic definition, result or score formula.
- Record deviations from the expected schema; actual official data prevails.
- Use `market_forward_excess_returns` as the proposed forecast label and retain `forward_returns` / `risk_free_rate` for verified portfolio accounting.
- Keep contemporaneous labels and unverified lagged fields outside predictors.

## Phases and acceptance criteria

| Phase | Scope | Acceptance evidence |
| --- | --- | --- |
| 1 | Scaffold, environment, official acquisition, audit, dictionary, initial documentation | Official provenance and true file statistics, or explicit blocked status with working tooling |
| 2 | Development-only EDA, missingness, target and preliminary alpha analysis | Reproducible notebooks and descriptive reports without final-label exploration |
| 3 | Chronological splits and linear baselines | Saved fold boundaries, fitted training-only pipelines, out-of-sample predictions |
| 4 | Advanced ML, regimes, alpha stability | Fold-level comparisons, historical regime calibration and selection-bias discussion |
| 5 | Signals, sizing, backtests, risk management | Causal weight/return ledger with costs, benchmarks and full risk metrics |
| 6 | Robustness, final evaluation, final report | Frozen configuration, one final holdout evaluation, sensitivity results and limitations |

## Current boundary

This delivery executes Phase 1 only. Official acquisition and real-data auditing are complete as of 2026-10-16: training is 9,048 rows × 98 columns, the supplied test is 10 × 99, and all 13 extracted files match the recorded hashes. The future module names and eight notebooks make the intended research structure reviewable, but future-phase modules are explicitly unimplemented.

Observed qualifications: there are 94 anonymous predictors and no MOM-prefixed columns; D6 is binary with -1/0 values; all 10 supplied test identifiers overlap training; 85 training feature columns have missing observations. No prediction, target-driven EDA or portfolio evaluation has been performed. A chronological final research holdout remains to be reserved before Phase 2.

## Non-goals

No third-party dataset fallback, competition submission, automated rules acceptance, live execution, investment recommendation, deep-learning-first workflow or claims based on a leaderboard score alone.
