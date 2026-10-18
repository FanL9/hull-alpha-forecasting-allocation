# Phase 3 Alpha Research Pipeline

## Purpose

Implement the pre-registered Alpha tests in [phase2.5-v2](research_protocol.md) without fitting prediction models or strategies. Synthetic results verify software and provide no evidence about market Alpha.

## Data boundary

Research accepts consecutive date_id **0–7859** only. Internal Holdout 7860–8867 and Final Simulation 8868–9047 are excluded.
Synthetic development reads configuration/schema metadata and artificial frames, never official market rows.
The future real entry requires explicit `--run-real-phase3`, first verifies the frozen protocol and raw-file hashes, then uses the existing `load_research_prefix` with a 7860-row limit. Hashing streams file bytes; it does not parse held-out observations.

## Pipeline

Frozen 11 folds → training-only availability → 97 single features and 7 independent composites → block-wise Rank IC → NW significance and bootstrap intervals → separate 97/7 BH and BY → global-104 sensitivity.

- Single features include all 94 documented anonymous fields and the 3 causally reconstructed lagged fields; unavailable hypotheses remain in their families with p=1.
- Composites always run, using the frozen training Spearman directions, empirical percentile maps, equal weights and member-count rules.
- Training-fitted winsor bounds affect X only; winsorized Pearson and fold stability are descriptive. Validation pair/coverage checks determine testability, never training feature eligibility.
- NaN gaps retain time positions. Bootstrap uses the already block-ranked sequence; concatenated ranks are never globally re-ranked.

## Formal decision rule

`PASS`: testable and its own family's BH passes. `FAIL`: testable and its own family's BH fails. `NOT_TESTABLE`: frozen minimum requirements or estimable HAC uncertainty are absent, with p=1.
BY, global-104, bootstrap intervals and stability diagnostics never override the family-specific decision. A BH-only discovery receives a descriptive dependence-sensitivity note.

## Outputs

Each run creates five CSV tables plus two JSON files:

- `single_feature_results.csv`, `family_composite_results.csv`
- `fold_level_ic.csv`, `feature_availability.csv`, `multiple_testing_summary.csv`
- `fold_transform_audit.json`: training members, directions, percentile-reference hashes and composite winsor bounds.
- `phase3_run_manifest.json`: Git/code/protocol/data identity, timestamp, boundaries, folds, inference settings, fixed hypothesis counts and output hashes.

The default synthetic destination is `artifacts/phase3_synthetic/`; the future real destination is `artifacts/phase3/`. Existing nonempty output directories are preserved. Artifacts stay out of Git.

```powershell
.\.venv\Scripts\python.exe -m src.cli phase3 --synthetic
.\.venv\Scripts\python.exe -m pytest
```

`phase3` without an explicit mode stops. Synthetic frames require a marker, and the full test suite blocks access to this checkout's real data directory. Real-entry ordering is tested with doubles only.

## Current status

Pipeline implemented and tested on synthetic data.
Real Alpha Research not yet executed.

Full suite: **195 passed, 0 failed, 0 skipped**. [Synthetic tests](../tests/test_phase3_alpha_pipeline.py) cover signal signs/noise, tails/drift, missing and late inputs, composite rules, training isolation, multiplicity, output provenance and execution guards.
