# Phase 4 incident: v1 real entry failed before parsing data rows

## What happened

The single authorized real Phase 4 attempt at annotated tag `phase4-engineered-pipeline-v1`
(commit `720d06015c55bbcd6300d265f3252f46f9d20574`, Protocol `protocol-v3.1` / `4ce62ab8f5a1dde62b2d682b1e5b1ae008e71f7b`)
passed preflight, reserved the canonical state and then stopped with
`KeyError: 'train_columns'` in `src/validation/walk_forward.py::load_research_prefix`.

`run_real` passed `configs/split_policy.json` unchanged to the bounded loader. That
loader also compares the CSV header with `split_plan["train_columns"]`, a field the
tracked policy does not contain. The Phase 3 and post-hoc entries add it with
`{**policy, "train_columns": read_header(train)}`; the Phase 4 entry omitted that line.
Synthetic tests replaced the loader with doubles, so the omission was not exercised.

## Data access

The loader checks, in order, the outer reservation, the train file's SHA-256 and then
`split_plan["train_columns"]`. The error was raised at that third check, before the
`pd.read_csv(nrows=7860)` call. Only file bytes (for hashing) were read; no X–Y data
row was parsed. `run_real` sets `real_xy_read=true` only after a successful load, so
the v1 state records `real_xy_read=false`. No Phase 4 output directory, IC, selected
feature or prediction was produced. Internal Holdout 7860–8867 and Final Simulation
8868–9047 were not parsed or used.

The v1 state file (`artifacts/phase4_execution_state.json`, ignored by Git) is kept
unchanged on the executing machine. It was not reset, edited or deleted.

## Remediation (v1.1)

Protocol v3 §9 requires a runtime fix to be reproduced synthetically and frozen
again before real re-execution. `phase4-engineered-pipeline-v1.1` therefore:

1. Adds the same `train_columns` header step used by Phase 3 (`load_research_history`).
2. Adds a regression test reproducing the v1 `KeyError` with the real split-policy
   structure, and an integration test running the unpatched bounded loader on an
   artificial 9048-row file, proving that exactly one `read_csv(nrows=7860)` occurs
   and no held-out sentinel value is parsed.
3. Uses its own state file `artifacts/phase4_execution_state_v1.1.json`. Preflight
   requires the v1 record to exist and to show a real, `failed_or_interrupted`,
   `KeyError` attempt at the v1 tag commit with `real_xy_read=false`; otherwise it
   refuses. The v1 record's bytes and SHA-256 are embedded in the v1.1 state and run
   manifest, so the remediation is auditable and cannot be used for silent reruns.

Only `src/research/phase4_execution.py` (entry/guards), its tests and documentation
change. Candidate construction, the nested selector, evaluation, the synthetic
calibration evidence and Protocol v3.1 are unchanged.

When the v1.1 real run completes, commit a byte-identical copy of the v1 state file
next to this report together with the sealed v1.1 results.
