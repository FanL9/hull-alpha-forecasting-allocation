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

## Future research design — not executed

Use expanding-window walk-forward solely within 0..8867 for future alpha selection, model comparison and tuning. Fold count, initial length, validation length, model, hyperparameters, internal holdout length and outcome-availability/purge policy remain undecided. Confirm release timing/label horizon before defining those gaps. Never shuffle.

Future imputers/scalers/selectors/regime thresholds are fitted only on each historical training window and frozen for its later evaluation interval. Nested chronological tuning, if adopted later, must preserve that boundary. After this design is agreed, specify an internal chronological holdout within development, a strategy freeze and a separately authorized 180-row competition simulation. Freeze code, data hashes, preprocessing, feature list, models, sizing/risk rules and any sequential update policy before that simulation. This phase executes no folds or models.

Earlier Phase 1 and withdrawn analysis cannot be erased from human/research history. This is a prospective firewall, not a claim of an historically pristine independent test. No new predictive claim is made. Future independent-performance reporting must disclose this limitation.

## Reproducibility

python -m src.cli phase1 verifies the preserved acquisition and audit. python -m src.cli phase2 verifies integrity, validates the reservation, reads only development values, runs separated descriptive EDA, exports identity-processed data, all tables and six figures, and renders the report from actual results. The summary records input/config/source hashes and dependency versions. Notebook 02 executes that same workflow from raw files and saves artifacts, then displays aggregate evidence. Tests include adversarial simulation payloads, target/X separation and future-perturbation invariance. No fake data supplies research conclusions; artificial fixtures are confined to tests.

No alpha discovery, predictive feature selection, model fitting, allocation optimization, or backtesting has been performed in the rebuilt Phase 2.
