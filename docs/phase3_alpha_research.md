# Phase 3 Alpha Research Pipeline

## Purpose

Implement the pre-registered Alpha tests in [phase2.5-v2](research_protocol.md) without fitting prediction models or strategies. Synthetic results verify software and provide no evidence about market Alpha.

## Data boundary

Research accepts consecutive date_id **0–7859** only. Internal Holdout 7860–8867 and Final Simulation 8868–9047 are excluded.
Synthetic development reads configuration/schema metadata and artificial frames, never official market rows.
The future real entry requires explicit authorization, a clean Git worktree and HEAD at the annotated `phase3-pipeline-v1` tag. Execution preflight reads metadata only; protocol/data verification must then succeed before the existing `load_research_prefix` parses at most 7860 rows. Hashing streams file bytes; it does not parse held-out observations.
The execution tag is separate from the unchanged Phase 2.5-v2 statistical lock.

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
- `phase3_run_manifest.json`: run_type, Git commit/tag/dirty state, aggregate pipeline code hash, per-file hashes, protocol/data identity, timestamp, boundaries, folds, inference settings, fixed hypothesis counts and output hashes.

The default synthetic destination is `artifacts/phase3_synthetic/`; real output is fixed to canonical `artifacts/phase3/`. Its real manifest blocks reruns even if another output path is supplied. An atomic `artifacts/phase3_execution_state.json` claim blocks concurrent or interrupted reruns before any research values are parsed. Existing evidence is preserved and no force-rerun option is provided. Artifacts stay out of Git.

```powershell
.\.venv\Scripts\python.exe -m src.cli phase3 --synthetic
.\.venv\Scripts\python.exe -m src.cli phase3-check
.\.venv\Scripts\python.exe -m pytest
```

`phase3-check` verifies protocol metadata and execution readiness without reading market data. It requires the freeze commit/tag to exist. `phase3` without an explicit mode stops. Synthetic frames require a marker; tests and calibration workers block this checkout's real data directory. Real-entry behavior is tested with doubles only.

## Current status

Pipeline implemented and tested on synthetic data.
Real Alpha Research completed once at `phase3-pipeline-v1`; see the [sealed results](../reports/phase3_alpha_research_results.md). No Phase 4 work has started.

The [synthetic engine tests](../tests/test_phase3_alpha_pipeline.py), [execution guards](../tests/test_phase3_execution.py) and [calibration panel](../tests/test_phase3_calibration.py) cover pipeline behavior, single-run/Git restrictions and statistical sanity. Calibration executes five IID null seeds and three Gaussian-copula power seeds, with 97/7/104 hypotheses and the unchanged 2000-draw bootstrap in every run. Weak signals are never required to pass BH; this finite panel does not certify nominal FDR.

Full suite: **224 passed, 0 failed, 0 skipped**. Results: [synthetic calibration report](../reports/phase3_pipeline_calibration.md).
