# D4-RECOVERY-IO-001 — Restore from the verified boundary

2026-09-10. ACCEPTED. Contract change: NO. Applies equally to recovery of either
formal variant in `stage4_formal_8fb878a48ba4`.

The Vanilla worker stopped at 2,424 groups / 303 committed windows / 606 Adam
steps. The original recovery scanned every historical checkpoint payload, and
the worker would repeat that scan on startup. Each checkpoint is approximately
1.4 GB. Scanning hundreds of GB before every restart unnecessarily delays GPU
work. The autonomous engineering policy explicitly permits changes to resume
mechanisms that preserve experiment semantics. Treating the source freeze as a
reason to leave this operational problem unresolved was an incorrect decision.

## Decision and implementation plan

Use `scripts/stage4_resume_runtime.py` as an explicit operational worker entry
point. It installs `scripts/stage4_resume_boundary.py:restore_boundary` in place
of `online.restore`, then executes the original worker. This is a documented
runtime restore override, not a claim that the entire runtime is unchanged.
The original 49 frozen execution files, their source guards, actor subprocesses,
checkpoint transaction implementation, RNG restoration, scientific configuration,
monitor schedule, and full raw acceptance verifier remain unchanged.

The boundary restore checks every committed window's consecutive index and full
state lineage, hashes all referenced commit artifacts (including selection,
update, checkpoint marker and policy-sync receipts), checks every historical
checkpoint file's existence/size, and matches the recorded policy digest and
step count. It fully hashes **every file of the checkpoint actually loaded** and
binds its controller to the run, configuration, selection, update, and previous
state. It ignores a stale convenience pointer. An uncommitted tail does not add
budget; a committed window after a gap fails closed. The original recovery code
still handles adoption/rollback and paid raw-rollout conservation.

Each invocation writes `runs/<run_id>/boundary_restore_audits/<timestamp>.json`
with the exact commits, restored state, implementation hash, scope, bytes checked
and elapsed time. Each worker launch records and validates the wrapper and
override hashes in its `operational_runtime.json`. Both formal variants are
eligible for this same restore policy. Fresh Dynamic initialization still starts
from the original SFT without history, under the frozen queue.

## Trade-off and acceptance

Historical same-size payload corruption is deferred to the full raw verifier;
it is **not** ruled out by a successful boundary audit. This is acceptable for
resuming from a separately fully verified boundary, but cannot support a Stage4
VERIFIED claim. Tests explicitly demonstrate both that limitation and detection
by the original full restore. Final `verify_stage4.py` continues hashing every
historical checkpoint; none of its acceptance checks are skipped or cached.

No existing per-file persistent hash cache was available from the interrupted
scan. Its partial progress is not represented as completed verified evidence.
The assistant-owned read-only scan was stopped with its PID, open file, I/O
counters and reason retained in `formal_recovery_optimization_transition_001.json`.
No training worker or other project's GPU task was terminated.

Boundary checks read approximately one full checkpoint plus the small historical
commit artifacts per invocation, instead of all historical payloads. There is no
stat-only hash cache for the boundary; it is fully rechecked after waiting for
the GPU and again when the worker starts. The supervisor waits while another
project owns the GPU, then launches the original run and attaches the persistent
queue. An external process can still race the GPU-idle check; existing memory
guards remain authoritative. No unrelated process is killed automatically.

## Validation and follow-up

Run CPU tests for exact state/checkpoint equivalence to the old full restore,
boundary corruption, missing historical files, corrupt metadata, broken state
lineage, commit gaps, stale pointers, preserved inflight raw data, and fresh SFT
initialization. Re-run existing recovery/boundary/transaction tests. Then check
the real 303-window commit chain and checkpoint 0302, retain its measured audit,
and leave the detached supervisor waiting for or using the GPU. The first resumed
commit must still prove 2,432 groups / 304 windows / 608 steps, identical retained
rollout data and native optimizer/RNG continuity before claiming resumed training.

Scientific training comparisons and budgets are unchanged. Recovery waiting and
I/O time remain real system costs; they must not be erased from wall-time analysis.
