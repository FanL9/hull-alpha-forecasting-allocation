# Phase 4A: engineered alpha implementation

Authority: [Protocol v3.1](protocol_v3_1_engineered_alpha_fair_comparison.md),
commit `4ce62ab8f5a1dde62b2d682b1e5b1ae008e71f7b`, annotated tag `protocol-v3.1`.
The old v3 and Phase 2.5-v2 rules/results remain immutable. This implementation
does not train the separately specified Phase 5 models or construct allocations.

## Implementation

- `src/features/engineered_alpha.py`: causal Level, Delta5/21, Z252 and observed-value Rank63;
  generalized frozen eligibility; reference-fitted signed alpha families and a separate unsigned X-only V regime.
- `src/research/phase4_engine.py`: one `NestedSelector` for A97 and B499; inner same-sign ranking,
  exact pairwise-complete training Spearman redundancy, fixed equal weights and paired outer evaluation.
- `src/research/phase4_synthetic.py` and `phase4_calibration.py`: the prospective 20 null seeds and
  three fixed seeds for each raw/transformed/interaction plant. Every run uses all 11 outer folds.
- `src/research/phase4_execution.py`: protocol/code identity, canonical atomic reservation,
  read-only preflight, clean-worktree/tag checks and future bounded real entry.

Two screening blocks are rebuilt inside every fitting reference. Descriptive
Pearson clipping uses actual earlier inner-OOF forecasts from complete selectors,
including their own two screening blocks. It does not use an outer-fitted
selector to forecast earlier training rows. No new p-values/FDR are computed.

Every selected score uses direction × (2 × training mid-ECDF − 1). The denominator
is the fixed selected count; missing mapped contributions become zero, subject
to at least max(2, ceil(selected/2)) observed members. Fewer than two selected
candidates produces the explicit no-alpha constant-zero state, whose IC is
undefined. Undefined folds are never dropped to improve a primary mean.

Lagged outcomes are reconstructed with offset −1. A rolling representation may
use X through its current decision row; all fitting maps/thresholds use only the
purged prefix. Earlier validation outcomes can enter later rows only through
the official available lag fields; they never refit that block's selector.

## Synthetic-only commands

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_phase4_features.py tests/test_phase4_engine.py tests/test_phase4_execution.py
.\.venv\Scripts\python.exe -m src.research.phase4 --synthetic --case raw --seed 11 --output work/phase4A/example
.\.venv\Scripts\python.exe -m src.research.phase4 --calibrate --workers 3 --workspace work/phase4A/calibration --output reports/phase4_pipeline_synthetic_calibration
```

Destinations must not already exist. Synthetic CLI/API cannot write canonical
`artifacts/` or `data/`. Workers deny all access to the project's market-data
directory. Only artificial arrays and frozen metadata are used.

The [calibration report](../reports/phase4_pipeline_synthetic_calibration.md)
and machine-readable evidence distinguish synthetic calibration from research.
Calibration is a software investigation, not evidence of real alpha efficacy.
No seed, K, transformation, threshold or window is retuned from the panel.

## Future real-run preflight

```powershell
.\.venv\Scripts\python.exe -m src.research.phase4 --check
```

This metadata-only command requires a clean checkout exactly at annotated
`phase4-engineered-pipeline-v1`, verifies protocol tags/locks and the code hashes
in that tag, checks frozen data-identity metadata/boundaries and unused state.
It never hashes/parses actual market files or runs selection. A successful
preflight is not authorization to execute real research.

The future real entry uses only the existing bounded `load_research_prefix`
after guards and an exclusive canonical reservation. State is always anchored
at `artifacts/phase4_execution_state.json`, output at `artifacts/phase4/`.
Any prior reservation (including failed/interrupted) or real manifest blocks a
second attempt. Alternative destinations are rejected before loading data.
There is no reset/force/rerun CLI. Real execution is intentionally left to a
separate user-authorized task after the pipeline freeze.

Internal Holdout 7860–8867 and Final Simulation 8868–9047 are not loaded by Phase 4.
Historical descriptive EDA/prior Phase 3/post-hoc observation caveats remain as
disclosed in v3; these are not retrospectively relabelled as untouched history.
