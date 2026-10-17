# Regime-Aware Alpha Forecasting and Dynamic S&P 500 Allocation

An independent quant research project combining a future anonymous-feature alpha research layer with future Hull-style dynamic S&P 500 allocation. “Jane Street-style” describes the intended research organization; it does not claim proprietary methods, data or endorsement. Only the official Hull Tactical dataset is permitted.

**Phase 1 — Official Data Acquisition & Audit: Completed.** Official acquisition, safe ZIP extraction, SHA-256 provenance, data audit/dictionary and credential safeguards are preserved.

**Phase 2 — Data Processing & EDA: Completed.** The previous Phase 2 research implementation and results were withdrawn. The rebuilt phase contains development-only descriptive EDA, a conservative preprocessing framework and a fixed competition reservation.

> No alpha discovery, predictive feature selection, model fitting, allocation optimization, or backtesting has been performed in the rebuilt Phase 2.

## Official data and fixed boundary

The only source is [Kaggle — Hull Tactical - Market Prediction](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data). Existing local files are reused; this rebuild made no data download. All 13 manifest-listed files, including 11 API/transport files, and the original ZIP pass SHA-256 verification.

| Area / file | date_id | Rows | Columns | Role |
| --- | --- | --- | --- | --- |
| Full train.csv | 0–9047 | 9,048 | 98 | Provenance, header and ID inventory |
| Development Area | 0–8867 | 8,868 | 98 | Current data processing and EDA; future research |
| Competition Simulation | 8868–9047 | 180 | 98 | Locked: count, IDs and schema metadata only |
| Official test.csv | 8980–8989 | 10 | 99 | Inference schema / API compatibility only |

There is no partition gap or overlap. Simulation values are excluded from all EDA, preprocessing choices, feature engineering, future selection and tuning. Its dtypes are not inferred from values; the same train header establishes column compatibility. Official test.csv overlaps the locked period and is **not an independent model evaluation set**. Only its header and IDs are inspected in this phase; it is never used for training, tuning or prediction evaluation.

Training columns are date_id, 94 anonymous features (M=18, E=20, I=9, P=13, V=13, S=12, D=9) and three outcomes: forward_returns, risk_free_rate, market_forward_excess_returns. There is **no independent MOM family**. Official test adds is_scored and three lagged fields in place of contemporaneous outcomes. Exact numbered-feature economic meanings and release timing are not inferred.

The [original Phase 1 audit](reports/data_audit.md), [data dictionary](docs/data_dictionary.md) and original summary remain unchanged. They predate the new firewall and include historical full-file inspection. The new lock governs subsequent use; it cannot make earlier inspection disappear. Their full-file statistics are not inputs to the rebuilt EDA or processing decisions.

## Actual development-only EDA findings

- No duplicate IDs or full rows, no infinite cells, and ordered unique date_id 0–8867. All 94 features and all three outcomes are retained.
- 85 features contain 137,675 missing cells: **16.52%** of feature cells. No feature is wholly missing or constant over observed development values. Long starting gaps are substantial: E7 has 6,969 leading missing rows, V10 6,049 and S3 5,733.
- Family missing fractions differ: M 25.84%, S 20.52%, V 20.10%; D has none. D6 is binary with values -1/0; its encoding is preserved.
- Member-level distributions and scales vary. E8 has sample skewness 6.09 and excess kurtosis 90.40. Extremes are retained. Anonymous fields are not assigned economic causes.
- Adjacent chronological feature blocks show distribution changes: median empirical CDF distance **0.3413**, 95th percentile **0.9921**, among 2,747 sufficiently observed pairs. These describe X distributions, not formal structural breaks or regimes. The last block has only 48 rows.
- **7 X–X pairs** have absolute Pearson correlation at least 0.95 (pairwise-complete, at least 126 observations); no features are removed. No X–Y correlation is computed.
- Target-only analysis: mean 0.00004252, median 0.00024783, sample std 0.01055681, excess kurtosis 2.2026. Lag-1 target autocorrelation is -0.0437. These retrospective descriptions do not establish predictability and are never joined to X.

See the complete [Phase 2 report](reports/phase2_data_processing_eda.md) and executed [notebook 02](notebooks/02_data_processing_eda.ipynb). Full per-feature statistics, missing intervals, family tables, temporal summaries, X–X pair counts and separate target-only artifacts are generated locally.

## Processing and leakage controls

The default `data/processed/development.csv` is an identity export: all 8,868 rows and 98 columns, original values and NaNs remain. No feature drop, outcome imputation, scaling, outlier removal or global historical mean/median fill occurs.

Optional software-tested candidates are a strictly prior expanding median and contemporaneous X missing indicators. They are not applied to the default export. When no past observation exists, the median candidate leaves NaN. A future fixed training-window median must be fitted separately at each training boundary and used only on subsequent rows; it is not used to fill the historical training period from its own future.

Trailing descriptive statistics end at the displayed row. Each drift comparison uses the preceding block and current block available at that endpoint. Full-development descriptive aggregates are not fitted transforms. Target rolling summaries describe realized labels retrospectively and are not prediction-time features. Feature EDA rejects outcome columns; target EDA accepts only its named target. No regime is defined from outcomes.

`configs/split_policy.json` keeps the firewall active even if ignored artifacts are removed. `artifacts/split_definition.json` binds exact ranges and source hashes; changed source data, schema or boundaries are rejected. Under this policy `phase1` / `audit` verify integrity and ID/schema metadata, preserve the historical audit byte-for-byte and cannot fall back to downloading or full-file value statistics.

## Future research design — not executed

The intended chain is Hull features → alpha research → return forecasting → regime analysis → signal → 0–200% S&P 500 exposure → risk and portfolio evaluation. These research questions remain unanswered: whether stable out-of-sample alpha exists, which families are stable, whether regimes matter, whether ML improves stability, and whether better forecasts improve investment outcomes.

Future development uses **expanding-window walk-forward within 0–8867**. Fold count, initial window, validation lengths, internal test length, models, tuning and timing/purge rules remain undecided. The next research stage must define them, then plan an internal holdout and strategy freeze before any separately authorized 180-row competition simulation. No validation folds are executed now. See [methodology](docs/methodology.md), [split design](docs/validation_design.md) and [research questions](docs/research_questions.md).

## Reproduce using existing data

Python 3.12+. From this repository, use the existing virtual environment:

```powershell
.\.venv\Scripts\python.exe -m src.cli check-env
.\.venv\Scripts\python.exe -m src.cli phase1
.\.venv\Scripts\python.exe -m src.cli phase2
.\.venv\Scripts\python.exe -m pytest
```

For a fresh environment install `requirements.txt`, or `requirements-lock.txt` for the captured dependency closure. On Linux/macOS use `.venv/bin/python`. Open notebook 02 with that environment and Run All; it executes the full data-only pipeline from the existing raw files and saves every artifact. It does not merely display precomputed research results. Notebook 01 calls the protected Phase 1 audit. Notebooks 03–08 are deferred plans without results.

Preserved Phase 1 acquisition tooling remains available for separately authorized initial acquisition (`python -m src.cli download`), safe extraction, hash verification and blocked-input reporting. This task never calls the downloader. If provenance or preserved audit files fail verification, stop and investigate; do not replace inputs silently. Existing Kaggle credentials remain local; never paste or commit their values.

## Files and outputs

| Path | Purpose |
| --- | --- |
| src/data/ | Preserved official acquisition/audit; conservative candidate preprocessing |
| src/validation/time_split.py | Identifier-only fixed reservation and bounded development loader |
| src/research/ | Separate X and target EDA, figures, generated reporting and orchestration |
| configs/research.yaml | Descriptive settings; no model/fold parameters |
| configs/split_policy.json | Tracked split firewall and preserved Phase 1 checksums |
| reports/phase2_data_processing_eda.md | Rebuilt report with six descriptive figures |
| artifacts/split_definition.json | Exact split, timestamp, source binding and data-use policies |
| artifacts/phase2_data_summary.json | Development quality, scope, processing and execution metadata |
| artifacts/missingness_summary.csv | All 94 features, missing counts and contiguous-gap summaries |
| artifacts/feature_statistics.csv | All 94 features, moments, quantiles and observed dtypes |
| notebooks/02_data_processing_eda.ipynb | Actually executed, reproducible data-only workflow |
| tests/ | Preserved Phase 1 checks plus boundary, firewall and causal-processing checks |

Raw/processed data, ZIPs, artifacts, credentials, caches and local delivery copies are ignored by Git. Reports/notebooks contain reviewed aggregates; observation-level and processed files remain local. The MIT license applies to project code/documentation, not official competition assets.

Anonymous definitions, publication timing, target construction, scoring details and a calendar mapping remain unresolved for future research. No alpha, forecast superiority or portfolio performance is claimed.
