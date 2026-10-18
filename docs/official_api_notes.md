# Official interface and field verification

Phase 1 acquisition and API inspection are preserved. Phase 2.5 adds verified field definitions from the [official data page](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data) and the [official metric](https://www.kaggle.com/code/metric/hull-competition-sharpe). See [research protocol](research_protocol.md) for the definitive timing/target rules and [metric notes](official_metric_notes.md) for formulas and source hashes.

## Preserved acquisition

The existing official archive contains train/test CSVs and 11 API/transport files. Its SHA-256 is `654f6e604e8cc76dca9649b3685b8738842ef46e78a3ccef93f0faf2dd962208`; all original bytes and the acquisition manifest remain unchanged. The archive does not contain the financial scoring formula. Phase 2.5 obtained that formula separately from Kaggle's official `metric` account; this supersedes the earlier pending-verification status without changing raw files.

## Inference behavior

The inspected `data/raw/kaggle_evaluation/default_gateway.py` loads test.csv, groups batches in original order and passes each batch to the predictor. It initially uses batch_id and falls back to the first sorted column, date_id in this file. It configures a five-minute response timeout; its competition-specific validation hook adds no rules. The supplied infrastructure was inspected, not executed against market data.

The official description explicitly lists the three `lagged_*` outcomes in inference data with a one-day lag. They are **competition-available lagged outcome features**, permitted as future candidates when reconstructed causally and when fold-level coverage qualifies. They are not permanently forbidden. Current outcomes, is_scored and date_id remain outside predictors. No lagged-feature alpha test has been run.

The default Phase 1 anonymous-name filter still returns anonymous fields only, preserving its original API. Future candidate construction uses the explicit policy in `src/features/availability.py` to include verified lagged fields. This distinction preserves 94 anonymous inputs while allowing a 97-field future candidate universe.

## Mock test and data boundaries

The local test.csv has 10 IDs, 8980–8989, overlapping the locked simulation period. It is used only for schema/interface compatibility. Its feature and lagged-outcome values are not read for protocol decisions, research, training or evaluation. The official description also states that the public training-phase test copied a historical 180-ID tail; that fact is not evidence of independent predictive performance.

Development remains 0–8867 and Competition Simulation remains 8868–9047. Phase 2.5 adds the prospective Internal Holdout inside development without redoing EDA. Exact observation publication cutoffs, tradable intraday execution and reconstruction details not published by the host are not guessed; the published one-day field definitions and scoring arithmetic are verified.
