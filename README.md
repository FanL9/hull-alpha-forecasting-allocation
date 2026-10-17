# Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation

An end-to-end quantitative research framework for forecasting S&P 500 excess returns and translating predictive alpha into dynamic market exposure.

**Implementation status:** Phase 1 is complete: official competition data was downloaded on 2026-10-16, all 13 extracted files were hash-verified, and real-data audit/dictionary outputs were generated. No predictive-alpha findings, trained models or backtest results are claimed. Later phases are explicitly planned, not implemented.

## Project Overview

This independent research project studies whether anonymous financial features contain useful forecasts of future S&P 500 excess returns, whether their predictive power changes across market regimes, and whether forecasts create economic value after risk management and trading costs. Intended uses include a research portfolio and quantitative finance / financial engineering applications.

Research chain:

```text
Financial features → alpha analysis → expected return forecast → market regime
                   → trading signal → position size → portfolio return → risk evaluation
```

## Research Motivation

Forecast accuracy alone does not establish investable alpha. The research separates statistical predictability, stability, regime dependence and portfolio utility. Simple benchmarks precede nonlinear models, and all comparisons must use chronological out-of-sample predictions.

## Research Questions

1. Do the supplied features predict future S&P 500 excess returns out of sample?
2. Is predictive power stable through time?
3. Which signals strengthen or fail across market and volatility regimes?
4. Do nonlinear models improve on historical mean and linear benchmarks?
5. Do better forecasts necessarily deliver better portfolio outcomes?
6. Does volatility-aware sizing improve risk-adjusted performance?

See [research questions](docs/research_questions.md) for planned evidence and failure criteria.

## Data

The only permitted raw data source is **Kaggle — Hull Tactical - Market Prediction**, competition slug `hull-tactical-market-prediction`:

- [Official competition data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data)
- [Official evaluation page](https://www.kaggle.com/competitions/hull-tactical-market-prediction/overview/evaluation)
- [Competition rules](https://www.kaggle.com/competitions/hull-tactical-market-prediction/rules)

**Never upload Kaggle raw data to GitHub.** Raw/derived data, archives, credentials and generated artifacts are ignored. Official evaluation/API files remain in the ignored raw directory under their own terms. The MIT license applies to project code and documentation, not competition assets.

The primary planned label is `market_forward_excess_returns`. `forward_returns` and `risk_free_rate` are reserved for economic accounting. Their precise definitions, units and availability will be checked against official files. The forecasting label is not assumed to equal the simple difference of the other two fields.

Numbered predictors are **anonymous features**. The family taxonomy in the project brief is M (market/technical), E (macro), I (rates), P (price/valuation), V (volatility), S (sentiment), MOM (momentum), D (dummy/binary). Individual economic meanings are not inferred. Actual names and counts will come from the downloaded CSV headers.

Observed official files:

| File | Rows | Columns | `date_id` range |
| --- | --- | --- | --- |
| `train.csv` | 9,048 | 98 | 0–9,047 |
| `test.csv` | 10 | 99 | 8,980–8,989 |

Both files contain the same 94 recognized anonymous predictors: M=18, E=20, I=9, P=13, V=13, S=12, D=9. **No `MOM*` column is present.** The training file adds `date_id` and three outcome fields; the test file instead adds `date_id`, `is_scored` and three `lagged_*` fields.

The test file's 10 date identifiers all overlap training history, so it is unsuitable as an independent final research holdout. Training has 137,675 missing cells across 85 feature columns; all three outcome fields have no missing values. The supplied test file has no missing cells. `D6` uses the binary encoding `{-1, 0}`, which is preserved rather than forced to 0/1. See the [full audit](reports/data_audit.md) and [observed data dictionary](docs/data_dictionary.md).

## Methodology

Phase 1 implements official-only acquisition, safe ZIP extraction, SHA-256 provenance, strict loading, schema/time checks, per-column missingness, feature-family missingness across input-order blocks, and generated documentation. It does not fit or transform data. The detailed later-phase plan is in [methodology](docs/methodology.md).

## Alpha Research

Planned analyses include Pearson/Spearman relationships with future labels, univariate coefficients, walk-forward predictive direction, rolling predictive power, family comparisons, sign stability and conditional regime performance. Training-window analysis selects candidates; only subsequent out-of-sample evidence can support a candidate alpha claim. In-sample correlations alone will not be called alpha.

Lagging signals to examine decay is distinct from constructing longer-horizon returns. Multi-period compounded labels will be built only after timing and consecutiveness are verified, with a matching purge gap. Multiple testing and selection bias must be reported.

## Validation Framework

The planned design is an expanding-window development sequence followed by a reserved chronological final holdout. No random KFold or shuffled split is permitted. Imputers, scalers, feature selectors, regime calibrations and model tuning are fitted on each training window only.

The supplied official `test.csv` overlaps training on all 10 identifiers and has no contemporaneous outcome fields. The downloaded gateway reads it and serves ordered batches; detailed label availability and the hidden evaluation still require verification. The research holdout comes from the labelled training history. Its cut is not yet created. Final labels remain unused for tuning and exploratory target analysis; schema/missingness inventory is permitted. See [validation design](docs/validation_design.md) and [official API source review](docs/official_api_notes.md).

## Models

Planned order: historical mean → linear regression → Ridge → ElasticNet → Random Forest → XGBoost → LightGBM. Optional MLP work requires stable baselines and evidence of need. No model is currently fitted. Current requirements contain only Phase 1 dependencies; model-specific packages will be added with their implementation.

## Signal Construction

Planned comparisons: raw forecasts, historical normalization, historical percentiles, clipping and causal smoothing. Calibration uses training or earlier out-of-sample predictions only. Full-sample z-scores, bidirectional smoothing and future information are prohibited.

## Portfolio Construction

Planned exposure is bounded by `0 ≤ w_t ≤ 2`, with linear, threshold and volatility-adjusted mappings. Benchmarks are 100% buy-and-hold, a predeclared simple rule, the development-selected ML strategy and a regime-aware strategy. Cash returns, leveraged financing, turnover and transaction costs must be explicitly accounted for after the official return semantics are verified. `w=1` and constant 100% exposure represent the same benchmark.

## Risk Management

Planned controls include exposure clipping, past-only volatility estimates, volatility targeting, signal smoothing and a causal drawdown overlay. A 1.20 volatility-ratio reference is recorded in configuration from the brief; the exact official score/penalty must be verified before implementation. An ex-post ratio check is not a guarantee of future volatility. Standard financial metrics will be reported separately from any competition metric.

## Evaluation Metrics

Prediction: MAE, RMSE, R², Pearson correlation, Spearman correlation and directional accuracy. Portfolio: cumulative/annualized return, annualized volatility, excess-return Sharpe, Sortino, maximum drawdown, Calmar, downside volatility, turnover and hit rate. Regime and robustness analyses must report sample sizes, costs and uncertainty. No unverified metric values are currently available.

## Repository Structure

```text
configs/       Base, model-plan and portfolio-plan YAML configuration
data/          Ignored raw, interim and processed inputs (.gitkeep only in Git)
docs/          Definition, questions, methodology, dictionary, validation
notebooks/     Eight numbered research notebooks; Phase 1 is executable
src/
  cli.py       check-env / download / audit / phase1 commands
  data/        Official download, strict loader, audit and eligibility filter
  features/    Implemented family taxonomy; engineering modules planned
  alpha/       Planned alpha research modules
  validation/  Planned chronological split and walk-forward modules
  models/      Planned ordered forecasting benchmarks
  signals/     Planned signal construction and smoothing
  portfolio/   Planned sizing, volatility targeting and backtest
  evaluation/  Planned prediction, portfolio and regime metrics
  utils/       Portable paths, configuration, RNG seeds, strict JSON
tests/         Phase 1 acquisition, audit and leakage-boundary tests
reports/       Human-readable data audit (includes a truthful blocked state)
artifacts/     Ignored summary/status JSON and future generated artifacts
```

Future-phase modules and notebooks are marked as planned. They contain no fabricated outputs or purported model implementations.

## Reproduction

Run commands from the repository root with Python 3.12+.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.cli check-env
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

Configure authentication **locally**, without pasting credentials into chat or committing them. The [official Kaggle CLI documentation](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md) describes an API token environment variable or `~/.kaggle/access_token`, and legacy `~/.kaggle/kaggle.json`. On Windows the latter is `%USERPROFILE%\.kaggle\kaggle.json`. The project also checks `KAGGLE_CONFIG_DIR` and the legacy username/key environment pair. Presence does not prove server validity. Check competition access and accept rules yourself on Kaggle if required; this project does not accept rules for you.

```powershell
.\.venv\Scripts\python.exe -m src.cli phase1
.\.venv\Scripts\python.exe -m pytest
```

The downloader invokes the installed official CLI:

```text
kaggle competitions download -c hull-tactical-market-prediction -p data/raw
```

It validates the full ZIP before extraction, preserves all official files, and creates `data/raw/download_manifest.json` with hashes. `phase1` reuses a hash-verified acquisition when possible. Separate commands are available:

```powershell
.\.venv\Scripts\python.exe -m src.cli download
.\.venv\Scripts\python.exe -m src.cli audit
```

The audit generates `reports/data_audit.md`, `docs/data_dictionary.md`, `artifacts/data_summary.json` and `artifacts/phase1_status.json`. Exit codes: 0 = requested action succeeded; 1 = observed schema/time validation failed; 2 = required data unavailable. Expected-schema deviations are reported for investigation; source data is never edited to match expectations. Generated reports may expose column names and aggregate statistics, so review competition terms before publication.

If Windows restricts its default temporary directory, point `TEMP` and `TMP` to a project-local `work/tmp` directory before installation. `requirements-lock.txt` captures all installed dependency versions for this Phase 1 environment; use `pip install -r requirements-lock.txt` to replay them.

## Results

| Deliverable | Current status |
| --- | --- |
| Project structure, configuration and documentation | Created |
| Official downloader and reproducible audit code | Implemented |
| Official train/test and evaluation/API files | Downloaded from the official competition; 13 file hashes verified, including 11 API/transport files |
| Actual data audit and observed dictionary | Generated from train 9,048×98 and test 10×99; no blocking schema/time errors |
| EDA, alpha analysis, training, regimes and backtests | Not started; Phases 2–6 |
| Empirical answer to the research questions | Not yet available |

Data-quality findings: substantial and time-varying historical feature missingness, absence of a separate MOM family, D6's -1/0 binary encoding, and complete overlap of the provided test dates with training. The tiny test sample also has constant nonmissing D1/D2/D4/D6/S7 fields; this is not evidence of constancy in the full history. The downloaded API code provides inference transport, not a verified financial scoring formula. No official score is claimed.

Unit tests use tiny, explicitly artificial fixtures in temporary directories. They verify software behavior and are not research datasets or evidence of market predictability.

## Limitations

Anonymous fields limit economic interpretation. Feature publication lags and target construction require verification. The absence of a calendar mapping limits calendar/regime claims. Nonstationarity, feature search, overlapping horizons, tail risk, leverage financing and transaction costs can invalidate apparent performance. No profitable strategy, live feasibility, official-score reproduction or model superiority has been established. This is a research framework, not investment advice.
