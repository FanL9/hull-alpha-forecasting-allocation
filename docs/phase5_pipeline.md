# Phase 5 nonlinear alpha execution freeze

Protocol [v3.1](protocol_v3_1_engineered_alpha_fair_comparison.md) and its inherited
[v3 sections 7–13](protocol_v3_engineered_and_deep_alpha.md) are authoritative.
This execution layer adds no statistical candidates, allocation or holdout access.
The Phase 4 A/B algorithm is reused unchanged with complete earlier reference
fits. Eventual real A/B outer forecasts must equal the archived Phase 4 forecasts
byte-for-value before Phase 5 evaluation proceeds. Phase 4 artifacts stay sealed.

## Environment

Use Python 3.12.6 and the versions in `requirements-phase5.txt`. The original
`requirements.txt` remains part of the immutable Phase 4 implementation.
Install the official CPU wheel first on Windows/Linux:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-phase5.txt
```

The [official PyTorch instructions](https://pytorch.org/get-started/previous-versions/)
provide the CPU index. `torch-cpu` is not a package requirement. Mac installations
can install `torch==2.10.0` from PyPI for development; the frozen Windows AMD64 CPU
environment is required for the single real run. Changing environments requires
separate review and synthetic validation, never a silent reproducibility fallback.
`configs/phase5_environment.json` binds the tested execution environment.

## Commands

```powershell
python -m src.research.phase5 --check
python -m src.research.phase5 --synthetic --case noise --seed 101 --output work/my_phase5 --workers 4
python -m src.research.phase5 --calibrate --output work/phase5_calibration --workers 4
```

Read-only preflight checks frozen metadata, annotated tags/code, environment,
clean Git state, original data hash bindings, protocol locks, protected Phase 3/4
bytes and canonical execution state. It does not read market CSV rows, train,
or reserve a run. Actual source byte verification happens only inside a separately
authorized real entry. Hashing bytes is not CSV parsing. No test/calibration opens
the project's actual `data/` tree.

The implemented `--run-real-phase5` entry is NOT authorized in Phase 5A. Its output
and task store are canonical `artifacts/phase5/`; the reservation and independent
anchor are `artifacts/phase5_execution_state.json` and
`artifacts/phase5_reservation.json`. Neither `--output` nor another project root
can create a second real attempt. There is no force/ignore-hash option.

## Deterministic task DAG

Eleven outer reference endpoints, their twelve distinct inner references, and
thirteen additional references for D's honest inner normalization produce 36
neural fitting contexts. Each has GRU/TCN × five seeds × stopping/refit stages:
720 neural stage tasks. All models use the same reference-specific representatives
and 63 × 16 design, including zero padding. Earlier selectors are complete fits;
no outer-selected representatives are reused for earlier OOF predictions.

The parent constructs physically truncated fitting label series. Worker capsules
contain only the stopping weight prefix, internal stopping block, full legal
refit labels, and target-free forecast sequences. A fold's later outcome may be
legal training history for a later fold. This is fold-local sealing, not a global
ban on all overlapping historical outcomes. Official lagged outcomes remain legal
past X. Forecasts and all normalization/residual maps are sealed before metric
attachment; no forecast function accepts its own subsequent block target.

Every task identity binds protocol, source-code commit/hash, model configuration,
data/split hashes, model, seed, reference/scenario/outer consumers, input/scaler
identity, train/purge/validation bounds and selected epoch count. Numerical arrays
are serialized with explicit dtype/shape and base64 bytes in sorted strict JSON.
No wall time, timestamp, pathname or completion order enters research payloads.
All five seeds are mandatory. Nonfinite loss is a recorded unavailable seed;
it cannot be retried with another seed, dropped, or replaced by a best seed.

One process per independent seed uses one Torch/BLAS thread, CPU float32 and
deterministic algorithms. Worker count is an engineering scheduling option only.
At most that many sequence capsules are materialized concurrently. Architecture,
optimization and early-stopping rules do not change with worker count.

## State machine and resume

`UNUSED → PREPARED → TRAINING → TRAINING_COMPLETE → EVALUATING → COMPLETE`.
An OS-backed coordinator lock serializes run-state transitions and per-task locks
prevent duplicate simultaneous writes; locks are released on process death.

- PREPARED/TRAINING/FAILED_PRE_EVAL: same identity may resume. A completed task
  is reused only after envelope, identity, output SHA and independent receipt
  validation. Missing/corrupt completed artifacts are an error, never a cache miss.
- An interrupted task with no completed artifact reruns from the original seed,
  never from partial optimizer/model checkpoints. Atomic temporary files are not
  treated as results. An orphaned complete envelope is validated before receipt
  recovery. Writes flush/fsync, atomically rename, then commit the receipt.
- TRAINING_COMPLETE/EVALUATING/FAILED_POST_EVAL: only replay evaluation/output
  publication from the complete sealed prediction artifact is allowed. No model
  fitting, epoch reselection or task replacement is permitted after this boundary.
- COMPLETE: refuse execution; read-only checks remain available.
- Missing state with an existing anchor, or missing anchor with state/output,
  refuses execution. A changed identity refuses every resume state.

An independent completion journal binds every completed task. For real execution
it lives at `artifacts/phase5_task_receipts.json`, outside the result/cache tree.
Deleting the output/cache tree cannot remove this authority or regain training.
After training begins, a missing journal is fatal. Read-only preflight verifies
orphans without recovering receipts or writing the journal; recovery is allowed
only within the locked resume path. The sealed state also binds its full completed
inventory before any evaluation.

The canonical reservation survives failures. Resume continues ONE statistical
attempt. It does not authorize parameter changes, another output directory or
manual removal of poor-looking completed tasks. A user able to rewrite Git,
reservation and all evidence can of course destroy an audit trail; this pipeline
rejects inconsistent evidence rather than claiming tamper-proof external storage.

## Synthetic evidence and freeze

The prospective panel definitions and numerical engineering criteria are recorded
in `configs/phase5_calibration.json` before calibration. Architectures cannot be
tuned from those results. The two-epoch compute budget is accepted only by the
synthetic software E2E test path, never real execution or full calibration.

Reports: [calibration](../reports/phase5_pipeline_synthetic_calibration.md) and
[runtime](../reports/phase5_runtime_benchmark.md). No real Phase 5 market X-Y
results were observed during implementation. Internal Holdout and Final Simulation
remain prospectively locked; the historical Phase 2 EDA disclosure is unchanged.
