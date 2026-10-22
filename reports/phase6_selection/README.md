# Phase 6A Real Allocation Selection

Completed on 2026-10-22 using frozen Protocol v4 and `phase6-allocation-pipeline-v1`
(commit `11a70fe7b7977606159dc8d0e694eff65268f164`). One real CLI invocation, 7.96
seconds; no retraining, model reselection or additional allocation search.

## Inputs and method

Only A's sealed research OOS forecasts on date_id 5088–7859 were used: 2772 rows,
11 blocks of 252. The bounded history loader parsed through 7859 for the frozen
lagged-volatility state and the same dates' forward returns/risk-free rate.
The raw-file byte hash check was performed as required by the existing data lock;
Internal Holdout and Final Simulation values were not parsed or evaluated.

The registered inventory was ten mappings plus `constant_1`. The unmodified
Hull official score selects admissible rules (overall above baseline and at least
6/11 block wins), then the first rule within the strict 0.05 near-best margin in
the frozen simplicity order. Costs and descriptive bootstrap intervals do not
change selection. No new rule was introduced after observing these results.

## Results

| Candidate | Official score | Blocks above baseline |
| --- | ---: | ---: |
| constant_1 | 0.863809 | - |
| linear_k0.10 | 0.865756 | 9 |
| linear_k0.20 | 0.858741 | 9 |
| linear_k0.35 | 0.768116 | 5 |
| linear_k0.50 | 0.677611 | 5 |
| linear_k0.75 | 0.616148 | 5 |
| volscaled_k0.10 | 0.967532 | 6 |
| volscaled_k0.20 | 0.988066 | 6 |
| volscaled_k0.35 | 1.007087 | 6 |
| volscaled_k0.50 | 1.011263 | 8 |
| volscaled_k0.75 | 0.906561 | 8 |

**Selected: `volscaled_k0.10`**, official score **0.967532**,
versus **0.863809** for `constant_1`; block wins:
**6/11**. The highest-scoring rule was
`volscaled_k0.50` (1.011263); its advantage over the
selected rule is 0.043730,
which is below 0.05, so the frozen simplicity order chooses k=0.10.

These are official score units, not percentage returns or evidence of a
statistically established allocation advantage. This is development selection;
the mapping still needs the separately authorized, frozen holdout evaluation.

## Frozen mapping state

```text
mu_A    = -0.03914753913955494
sigma_A = 0.20598592027915236  (ddof=1; 2772 research OOS forecasts)
z_t = (A_forecast_t - mu_A) / sigma_A
c_t = clip(std_252(lagged_returns) / std_21(lagged_returns), 0.5, 1.5)
position_t = clip(c_t * (1 + 0.10 * z_t), 0, 2)
```

Undefined volatility ratio uses 1.0; missing forecast maps to position 1.0.
The frozen mean/std and selected mapping are retained for future holdout/final
execution. Selected mean exposure was 1.143874; its
observed range was [0.424283, 1.827065].
Cost sensitivities: 1 bp score 0.959713; 5 bp score
0.928433. Costs did not influence the chosen rule.

## Verification and boundaries

Protocol and pipeline tags were published before selection. Both preflights and
all 25 Phase 6/protocol tests passed. The first test invocation encountered sandbox
temporary-directory setup errors; unchanged tests passed with a workspace-local
temporary directory. No real selection was attempted before the passing checks.
This task did not rerun the older neural-network suite or alter any frozen file.

The canonical state is `completed` and continues to block another real attempt.
All four published output hashes match the canonical state and manifest. Outputs
and state were copied byte for byte; their hashes are recorded in
[snapshot_seal.json](snapshot_seal.json). Original canonical files remain intact.

- [Candidate scores, costs and descriptive intervals](phase6_candidate_scores.csv)
- [Block scores](phase6_block_scores.csv)
- [Daily positions](phase6_daily_positions.csv)
- [Selected mapping and standardization](phase6_selection.json)
- [Original run manifest](phase6_run_manifest.json)
- [Canonical state snapshot](phase6_selection_state.json)

Internal Holdout 7860–8867 and Final Simulation 8868–9047 remain locked. No held-out
model refit, allocation evaluation or final competition simulation was performed.
