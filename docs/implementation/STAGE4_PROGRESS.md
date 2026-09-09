# Stage4 progress and continuation

Updated2026-09-09 22:56 Asia/Shanghai. **PILOT_PASS; Stage4 incomplete.** Stage0 VERIFIED,
Stage1–3 DONE, Stage5–6 NOT_STARTED. Formal progress: Vanilla0/5000,
Dynamic0/5000 accepted mixed. READY_FOR_STAGE5=NO.

| Gate | Actual evidence |
| --- | --- |
| Matched real8×4 optimization diagnostic | PASS; three fresh SFT/native Adam conditions, shared trajectories and old logprobs |
| Vanilla smoke |32 training groups,4 windows,8 optimizer steps; raw verifier PASS |
| Dynamic smoke |32 accepted mixed,80 generated groups,4 windows,8 steps; raw verifier PASS |
| Real committed-boundary resume |Actual SIGTERM/new parent and native optimizer continuity PASS for both variants |
| Native final reload preflight |Vanilla smoke checkpoint reloaded in a fresh process, Adam step8,558 finite response logprobs |
| Pilot |Both fresh512-group runs completed64 windows/128 optimizer steps and passed raw verification |
| Formal |Not prepared or launched; both625-window budgets still required |

Pilot pair prepared and detached queue launched at2026-09-09 07:52UTC
(15:52 Asia/Shanghai), from shared code commit`11be2ed`:

- Vanilla:`s4_pilot_vanilla_20260909T075203_95a6e9`, initially RUNNING with
  worker PID2174146.
- Dynamic:`s4_pilot_dynamic_20260909T075204_3e449e`, PREPARED and queued.
- Queue PID2174073; logs under
  `/data/WSH/medical-post-train-artifacts/stage4-pilot-queue/`.

These are launch-time observations; inspect current PID/heartbeat before acting.
Pilot progress never increments the two formal counters in project state.

Completion update: Vanilla raw verification PASS at18:22; Dynamic PASS at22:10.
Queue status is`BOTH_PILOTS_RAW_VERIFIED` and the pilot queue has ended. No
formal training is currently running. Analysis and remaining gates are in
`STAGE4_PILOT_REVIEW.md`; exactly0 formal groups have been trained.

Runtime is native verl FSDP2/GSPO plus native vLLM rollout. The shared smoke/
pilot candidate is LR1e-6, mini4 prompts, one epoch, G4 and8 groups/window.
The formal config is not frozen. Both pilots must complete before its decision.
Two failed early runs and the initial verifier/test failures remain documented.
Cross-process seeded sampling is not fully bitwise reproducible:29/32 first
smoke responses matched token IDs, despite identical initial parameter digest.

## Inspect before acting

Read `experiments/stage4/pilot_queue_status.json` and `pilot_pair.json` when
present. The detached queue runs Vanilla then Dynamic, each with an actual
pause/SIGTERM/new-process check. It requires full raw verification before
advancing to the second pilot. Queue completion does not launch formal runs.

```bash
.venv-train/bin/python scripts/run_stage4.py inspect --run <exact-run-path>
```

If the owner is alive and heartbeat/progress move, leave it running. After an
observed exit, inspect stderr/failure/last checkpoint first. A retained incomplete
actor transaction can be audited and archived before restoring the same run:

```bash
.venv-train/bin/python scripts/run_stage4.py recover --run <exact-run-path>
.venv-train/bin/python scripts/run_stage4.py launch --run <exact-run-path>
```

`recover` requires all recorded parent processes dead and GPU idle. It preserves
raw rollout, archives orphan actor work, and never adds uncommitted budget.
Unreturned generation/validation cost cannot be fabricated; that case fails
closed and needs investigation. A completed pilot/formal cannot be replaced by
a fresh run just because a session ended.

## Remaining gates before formal launch

Both fresh512-group pilots, monitor512 evaluations and the initial stability/
cost/paired-case review are complete. No persistent collapse was observed.
Perform the additional checkpoint transaction fault
injections required by `CHECKPOINT_AND_RESUME.md`; the two committed-boundary
resume tests do not claim those additional crash timings were tested.

Then freeze one formal pair config and one code commit, declare both fresh SFT
run manifests, and execute exactly5000 groups per variant. Complete raw verifier,
validation/cost curves, native reloads, cases, report, interview narrative and
Stage5 readiness. Final commit/push and clean worktree follow those gates.

Bulk path: `/data/WSH/medical-post-train-artifacts/runs/`. Git holds compact
manifests, receipts, curves and cases. Clone alone does not restore bulk data.

## Formal entry gates — completed2026-09-09 15:57UTC

All three real transaction injections PASS (`recovery_fault_injections.json`): A temp-before-rename rollback, B renamed native checkpoint adoption without optimizer replay, C applied-but-uncommitted optimizer rollback. Final effective steps6 in each24-group diagnostic; physical steps8/6/7. All paid raw output and orphan work retained. Detailed numerical replay limitation and formal freeze decision are in `STAGE4_FORMAL_EXECUTION.md` and D4-FORMAL-FREEZE.

Final preflight suite:134 PASS. Shared formal candidate and first-crossing1M-token monitor protocol are recorded. Next operation is clean-commit freeze and preparation of two fresh-SFT5000-group runs. At this entry-gate snapshot no formal worker has launched and both formal counters remain0. Stage5/6 remain NOT_STARTED, READY_FOR_STAGE5=NO.
