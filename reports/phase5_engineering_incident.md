# Phase 5A synthetic engineering incident

No market data were accessed. No real Phase 5 reservation was created.

The first full-suite run reported **466 passed, 6 errors**. All six errors arose
from one shared E2E fixture failure: Windows temporarily denied `os.replace`
while another process read the completion journal. The initial engine-only E2E
had passed before the independent journal was added. The failing source hash was
`e16bec1063cd11a0218269f0970744e96ed24ffba164616e692d322b0d4a925a`.

The three first full-budget synthetic panel jobs were stopped before any final
panel IC summary was inspected. Their incomplete evidence remains under ignored
`work/phase5A/calibration/`; it is not silently removed or relabelled a pass.
The traceback/XML remains at `work/phase5A/full-tests.xml`. An earlier manual
synthetic command also failed on a nonexistent artificial source path before
reservation/loading; the three proper artificial sources were then generated.

Remediation retries only the same fsynced temporary bytes for transient
`PermissionError`, up to five seconds, before the atomic rename. It never trains
a task again to retry publication. Persistent denial and every other error remain
fatal. Deterministic tests cover both transient and persistent denial. Model
architecture, initialization, optimization, generators, seeds and prospective
calibration criteria are unchanged.

Because executable identity changed, the old partial calibration cache is NOT
reused under the new identity. All panels and the full suite are rerun in fresh
`work/phase5A/calibration_v2/` and `work/phase5A/full-suite-v2/` directories. Only
those complete, separately identified results may support the execution freeze.

A subsequent static review found that the calibration report wrapper's shortcut
for already-completed panels did not revalidate their identity/output hashes.
It did not affect model fitting, but could reuse a stale report after a source
change. The wrapper now checks source bytes, code/config/data identity, state,
anchor, complete task inventory, sealed predictions, manifest and every output.
It also rejects E2E smoke-budget evidence as production calibration. Orphaned real
completion journals alone now block a fresh reservation. Tests cover these cases.
No full-panel IC summaries were inspected before these checks were added.

The unfinished v2 validation was stopped conservatively. The final executable
version is validated afresh in `work/phase5A/calibration_v3/` and
`work/phase5A/full-suite-v3/`. All earlier evidence remains present. No architecture,
optimizer, seed, generator, eligibility rule or calibration criterion changed.

The v3 full suite finished with **469 passed, 8 setup errors**, all from the same
shared E2E fixture. It exposed the corresponding Windows read-side sharing denial
while another process replaced the journal. This is an infrastructure failure,
not a model or statistical failure. Bounded access-denial retry now covers reads
as well as rename; malformed JSON, failed checksums and persistent denial are
still fatal. A four-process, 120-task journal stress test exercises the actual
task store independently of neural training. Final evidence is produced in fresh
`work/phase5A/calibration_v4/` and `work/phase5A/full-suite-v4/` directories.

The new stress test also exercised a longer pytest temporary path and found
Windows path-length failure when the UUID was appended to a 64-character task
filename. Temporary artifacts now use short UUID sibling names on the same
filesystem. Final content-addressed names and serialized research bytes do not
change; incomplete temporary files still cannot count as completed tasks.

Final full-suite evidence uses `work/phase5A/full-suite-v4b/` and
`work/phase5A/full-tests-v4b.xml`. The earlier v4 invocation was stopped to reduce
CPU contention, without a source change or a reported test failure; its partial
evidence remains present. The final candidate implementation hash is
`bf26dca9c36312707c9fd8e27bb32f6a0682c78e202f144be09ee4333cf63bcb`.
Full-budget v4 calibration jobs were also paused/resumed to control contention.
This exercises the same run identities: completed stages remain immutable and
incomplete stages restart from their original initialization. No seed is removed,
and no model, generator, prospective criterion or source identity changes.
