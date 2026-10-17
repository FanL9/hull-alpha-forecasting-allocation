# Official acquisition and API source review

Review date: 2026-10-16. Scope: file inventory and read-only source inspection, without executing the supplied API or submitting predictions.

## Provenance

Source: [official Kaggle Hull Tactical competition data](https://www.kaggle.com/competitions/hull-tactical-market-prediction/data), downloaded using the official CLI with locally configured authentication. Archive: `hull-tactical-market-prediction.zip`, 3,032,317 bytes. SHA-256:

```text
654f6e604e8cc76dca9649b3685b8738842ef46e78a3ccef93f0faf2dd962208
```

The complete original ZIP and its 13 extracted files remain in ignored `data/raw/`. The local manifest records acquisition time, CLI version, byte counts and SHA-256 hashes. Both required CSVs and every manifest-listed API file have been reverified. This records acquisition provenance and later integrity; a locally generated hash manifest is not a publisher signature.

## Actual files

- `train.csv`: 9,048 rows × 98 columns.
- `test.csv`: 10 rows × 99 columns.
- 11 files under `kaggle_evaluation/`: gateway and inference entry points, transport/templates, protobuf definitions and generated communication code.

No financial scoring function defining the volatility penalty or the competition's Sharpe-like calculation was found in these downloaded Python files. File names containing “evaluation” are not evidence that the archive contains the financial metric implementation. The 1.20 configuration reference remains an unverified requirement until the official metric definition is obtained.

## Verified gateway behavior

The source file `data/raw/kaggle_evaluation/default_gateway.py` reads `test.csv` and groups batches in original identifier order. It initially looks for `batch_id`; when that field is absent it uses the first CSV column and checks that it is sorted. In the supplied file, that first column is `date_id`. The gateway configures a five-minute response timeout. Its competition-specific validation hook contains no additional logic.

`default_inference_server.py` connects the generic inference server to this gateway. The remaining inspected files provide communication infrastructure. Phase 1 does not execute this infrastructure or install its optional Polars/gRPC runtime dependencies, since neither is needed for CSV auditing.

## Observed schema and research implications

Training contains `date_id`, 94 anonymous features and three outcomes. Test contains the same 94 feature names, `date_id`, `is_scored`, `lagged_forward_returns`, `lagged_risk_free_rate` and `lagged_market_forward_excess_returns`.

Training identifiers are 0–9,047; test identifiers are 8,980–8,989. All 10 test identifiers overlap training. Both files are ordered and have unique identifiers internally. These facts establish that this supplied test file cannot supply an independent future research holdout. They do not establish hidden evaluation dates or future API content.

The gateway passes batches through, but its source does not establish the precise financial definitions, timing and publication lags of every field. Current predictor eligibility therefore continues to exclude all contemporaneous outcome fields, all `lagged_*` fields, `is_scored` and unclassified fields. Verify official descriptions before relaxing any exclusion or constructing longer-horizon labels.

The research holdout will be reserved from the chronological tail of labelled training history before development target exploration. No final cut or empirical model evaluation has yet been performed.
