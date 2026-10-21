# Phase 5 Real Research Results

The single authorized real Phase 5 run completed successfully on 2026-10-21.
No model, protocol, seed, ensemble weight or selection rule was changed after
observing these results. This records the frozen development-study outcome.

## Execution and method

- Pipeline: `phase5-deep-alpha-pipeline-v1`, commit `4ce84be875c9a25a37615833924eebdbabe1fa49`.
- Protocol: `protocol-v3.1`, commit `4ce62ab8f5a1dde62b2d682b1e5b1ae008e71f7b`.
- Parsed research history: date_id 0–7859; target `market_forward_excess_returns`.
- All 11 frozen expanding outer folds and five neural seeds were retained.
- Ridge, GRU and TCN used the frozen 63 × 16 representative-input design.
- A: Raw-Control; B: Engineered; C: equal-weight GRU/TCN; D: frozen normalized A/B/C combination.
- Reference-local selection, scaling, purged early stopping, refit, OOF normalization
  and residualization followed the [frozen execution design](../docs/phase5_pipeline.md).
- Exact A/B forecast continuity with sealed Phase 4 was required before evaluation.
- Start: `2026-10-21T08:30:49.495703+00:00`; finish: `2026-10-21T09:40:54.835718+00:00`.
- Actual wall time: **70.09 minutes**, four workers,
  one numerical thread each. There was one CLI invocation and no automatic retry.

## Observed results

Primary statistics are equal-weight means of the 11 outer-fold Rank ICs.

| Pipeline | Mean OOS Rank IC | Positive folds | Worst fold |
| --- | ---: | ---: | ---: |
| A | 0.076468 | 10/11 | -0.004938 |
| B | 0.062949 | 11/11 | 0.001311 |
| C | 0.055368 | 9/11 | -0.026770 |
| D | 0.077398 | 11/11 | 0.003244 |
| GRU | 0.050306 | 9/11 | -0.022357 |
| Ridge | 0.052075 | 8/11 | -0.022621 |
| TCN | 0.052357 | 10/11 | -0.004020 |

The standalone Ridge/GRU/TCN rows are diagnostics; the formal primary candidates
remain A/B/C/D. Constant historical-mean and zero forecasts have undefined Rank IC
under the frozen variance rule; their blank entries are not zero-IC estimates.

## Frozen development selection

**Selected: A.** D has the highest mean IC, but D − A is only **0.000930**.
A and D both satisfy the predeclared near-best tolerance of 0.01; the frozen
simplicity preference therefore chooses A. This is development selection only:
the holdout remains locked. No replacement winner or new tuning was introduced.

C − A is **−0.021100**. GRU − Ridge is **−0.001768**, and TCN − Ridge is
**+0.000282**. The observed mean differences do not establish a clear nonlinear
improvement. These are descriptive results, not new significance tests or
portfolio-profitability claims. Full per-fold differences and residual diagnostics
are archived rather than used to change the frozen selection rule.

The TCN interpretation applies to this exact global-average-pooling architecture.
Its synthetic weaknesses were known before the authorized real run. These results
do not test alternative pooling designs or the whole class of temporal convolutions.

## Boundaries and evidence

Only the research prefix was parsed for model research. Internal Holdout 7860–8867
and Final Simulation 8868–9047 were not parsed or unlocked. Source-file byte hashing
for the frozen data lock does not constitute parsing held-out X/y. The historical
Phase 2 EDA disclosure remains as previously documented.

The canonical execution state is COMPLETE; the single-run reservation remains
consumed. Original artifacts, state and receipts are preserved. All 7 published
output hashes, the sealed-prediction hash and all 877 completed task-envelope and
receipt hashes were checked before archiving. Cache tensors stay local; this archive
is an immutable research record, not a portable resume workspace.

The original frozen release validation remains **479 passed, 0 failed, 0 skipped**.
This results-only commit changes no executable files; no model or test rerun was
used to produce the archive. No allocation, backtest or held-out evaluation occurred.

- [Summary](phase5_real_run/phase5_summary.csv)
- [Fold results](phase5_real_run/phase5_folds.csv)
- [Predictions](phase5_real_run/phase5_predictions.csv)
- [Residual diagnostics](phase5_real_run/phase5_incremental.csv)
- [Paired fold differences](phase5_real_run/phase5_differences.json)
- [Frozen selection](phase5_real_run/phase5_primary_selection.json)
- [Original run manifest](phase5_real_run/phase5_run_manifest.json)
- [Snapshot seal](phase5_real_run/snapshot_seal.json)
