# Methodology

## Phase 1 — preserved acquisition and audit

The only source is the official Hull Tactical Kaggle competition. The implemented downloader uses the official CLI, validates credentials by presence without logging values, checks the ZIP for corrupt members/path traversal/symlinks/duplicate members, preserves the archive and official API files, and records SHA-256 provenance. Hashes verify consistency with recorded local acquisition, not a publisher signature. The strict loader rejects empty CSVs and duplicate headers. The original audit includes schema, dtypes, chronology, missingness and an observed dictionary. All Phase 1 tests, source data and outputs are retained.

The completed Phase 1 full-file value audit predates the new reservation. It remains historical evidence and is not recomputed or consumed for new processing choices. Under configs/split_policy.json, phase1 and audit instead verify the official manifest/ZIP, protected historical output hashes, headers and identifiers. They write separate integrity receipts and do not auto-download on failure. Without a split policy, the original initial-acquisition/audit workflow remains available; unit tests exercise it using artificial temporary fixtures.

## Rebuilt Phase 2 — boundary before values

The superseded alpha/model implementation and result artifacts were withdrawn. This phase is Data Processing + EDA + Research Split Design only. It uses existing files and performs no acquisition, predictive correlation, alpha screening, model fitting, tuning, regimes, signals, allocation or portfolio evaluation.

Train has 9048 ordered rows (0..9047). Development is exactly 0..8867 (8868 rows). Competition Simulation is exactly 8868..9047 (180 rows), with no gap or overlap. The boundary is a user-fixed design, not selected from outcomes.

Before loading values, read only the raw header and date_id column, verify counts/order/uniqueness/ranges and persist a source-hash-bound split. The value-bearing loader uses nrows=8868, never constructing a simulation dataframe. Simulation checks cover only count, IDs and a common header. Its value-derived dtypes, duplicates, feature/target statistics and missingness are deliberately uninspected. The tracked policy survives deletion of ignored output artifacts and rejects changed boundaries or raw files.

Official test.csv is exclusively for inference schema/API/column compatibility and future interface testing. Its IDs overlap the locked period; this phase reads only IDs and the header. It is never training, tuning, an independent holdout, or model-prediction evaluation data.

## Development quality and missingness

Retain original order and all 94 X columns plus date_id and three outcomes. Check numeric dtypes, duplicate rows/IDs, chronological order, missing and infinite values. Reject invalid inputs rather than sorting/coercing/replacing them. Features are anonymous; use observed M/E/I/P/V/S/D prefixes, never invent MOM or infer numbered-feature economics. Preserve D6's -1/0 encoding.

For every X column report missing count/fraction, first/last observation, every contiguous missing interval, leading/trailing runs and longest gap. Summarize feature/family missingness in chronological 252-row blocks, marking the shorter final block. Outcome presence/missing counts are quality checks; missing returns are never filled.

## Feature-only descriptive EDA

Feature entry points accept X only and reject outcomes/metadata. For all features report mean, sample std (ddof=1), median, quantiles (1/5/25/75/95/99%), min/max, sample skewness and excess kurtosis. List value counts for fields with at most 20 unique observed values, identifying binary and discrete fields. All samples and unusual observations are retained. Family summaries take medians of member statistics, not pooled comparisons of incompatible units.

Trailing means/std use windows ending at the displayed row (252 rows, 126 minimum observations), sampled every 21 rows for display. Block drift compares adjacent chronological blocks using empirical CDF distance, mean change scaled by previous std and current/previous std. Require at least 20 observations in each block; undefined denominators stay undefined. This compares data available by each block end and is not a transformation, regime assignment, significance test or formal structural-break result. Future perturbations must not change earlier summaries.

X–X Pearson correlations use pairwise complete observations and at least 126 pairs; save observation counts. List distinct pairs with absolute correlation at least 0.95 without dropping any field. Family summaries use median/max absolute distinct-pair correlations; same-family self-correlations are excluded. Pairwise-complete matrices need not be positive semidefinite. Coverage, temporal trends, dependent observations and nonstationarity constrain interpretation. No Pearson/Spearman X–Y, IC/rank IC, mutual information, predictive importance or target-driven selection is calculated.

## Target-only EDA

A separate function accepts only development market_forward_excess_returns and IDs. Describe its unchanged distribution, moments/quantiles/extremes, trailing mean/std and lag-1..20 autocorrelation with pair counts. Extreme-observation details remain ignored local artifacts. No X is passed or joined. Retain supplied row alignment; do not shift, reconstruct, smooth, clip or fill the label. Target rolling statistics include the current realized label and are retrospective descriptions, never forecast-time predictors or regime inputs.

## Processing framework

Default processing is identity/retain NaN: all 8868 rows and 98 columns retain original values. No selection, normalization, scaling, global fill or outlier removal is performed.

Candidate designs only: native missing-value support; a median fitted inside a future historical training window and applied to later rows; strictly prior expanding median; current X missing indicators. The optional implemented expanding helper uses expanding median shifted by one row, leaves initial missing values unfilled and accepts X only. Indicators use only current missingness. They are tested but not applied to the default processed dataset. A full-development mean/median cannot fill its own historical periods; a future window transform must be fitted afresh at each boundary. No candidate is chosen using targets or predictive results.

Full-development descriptive tables are offline summaries, not causal online features or fitted transforms. All exported time-indexed rolling/block statistics end at or before their displayed timestamp; no centered window, backfill or future reference is allowed. date_id establishes ordering, not calendar dates or verified trading-day continuity.

## Phase 2.5 — research protocol frozen (current version: phase2.5-v2)

The [research protocol](research_protocol.md) now verifies the official one-day labels, five-year mean/MAD=4 target definition, one-day lagged field availability, baselines, multiplicity and timing rules. The [official metric](official_metric_notes.md) has been reproduced from Kaggle's published code and checked on synthetic cases. Its 1.20 volatility ratio is a verified penalty threshold. Forecast market_forward_excess_returns; the final scoring layer instead uses allocations, forward_returns and risk_free_rate. It never scores the forecasting target's IC directly.

Future expanding-window research is restricted to 0..7859. Internal Holdout is 7860..8867 (1008 rows); the final 180-row simulation remains locked. Initial training is 5087 rows, followed by 11 validation blocks of 252 rows, step 252, purge 1 and embargo 0. These parameters are fixed in configs/validation.yaml; no validation research or model fitting has run. Model families and hyperparameters remain unselected.

Future imputers/scalers/selectors/regime thresholds use historical training windows only and stay frozen through each validation block. X-only availability gates late-starting inputs; no field is forced into early folds. Primary Alpha metric: **block-wise Spearman Rank IC** (X and y ranked inside each validation block, ranked blocks concatenated without re-ranking). Secondary robustness metric: **training-window 1%/99% winsorized Pearson IC** (X only; bounds from the training window, reused on validation; never a discovery gate). Newey–West HAC lag 5 supplies the p-value; the paired moving-block bootstrap on the ranked series supplies confidence intervals only. Two pre-registered hypothesis families are each tested with BH FDR 0.05: 97 single features and **7 predefined family-composite hypotheses** (M/E/I/P/V/S/D, training-only members, directions and percentile maps, equal weights). A **104-hypothesis global BH sensitivity analysis** and BY sensitivity are reported alongside and never replace family results. v2 was defined before any real feature–target research, motivated only by X heavy tails, drift and availability structure seen in Phase 2 EDA. Baselines are the purged historical training mean and allocation=1.0; zero forecast/cash are sanity checks. Deflated Sharpe is reserved for later model/strategy search, not single-feature screening. Details and limitations are in the frozen protocol.

Selection and strategy freeze must precede any separately authorized Internal Holdout evaluation and final competition simulation. The internal holdout has already been described in historical Phase 2 EDA, so its new reservation is prospective, not historically untouched. No holdout/final values were used to choose Phase 2.5 parameters.

Earlier Phase 1 and withdrawn analysis cannot be erased from human/research history. This is a prospective firewall, not a claim of an historically pristine independent test. No new predictive claim is made. Future independent-performance reporting must disclose this limitation.

## Reproducibility

python -m src.cli phase1 verifies the preserved acquisition and audit. python -m src.cli phase2 verifies integrity, validates the reservation, reads only development values, runs separated descriptive EDA, exports identity-processed data, all tables and six figures, and renders the report from actual results. The summary records input/config/source hashes and dependency versions. Notebook 02 executes that same workflow from raw files and saves artifacts, then displays aggregate evidence. Tests include adversarial simulation payloads, target/X separation and future-perturbation invariance. No fake data supplies research conclusions; artificial fixtures are confined to tests.

No alpha discovery, predictive feature selection, model fitting, allocation optimization, or backtesting has been performed in the rebuilt Phase 2.
