# Stage4 progress and continuation

Updated2026-09-09. **SMOKE_PASS; Stage4 incomplete.** Stage0 VERIFIED,
Stage1–3 DONE, Stage5–6 NOT_STARTED. Formal progress: Vanilla0/5000,
Dynamic0/5000 accepted mixed. READY_FOR_STAGE5=NO.

| Gate | Actual evidence |
| --- | --- |
| Matched real8×4 optimization diagnostic | PASS; three fresh SFT/native Adam conditions, shared trajectories and old logprobs |
| Vanilla smoke |32 training groups,4 windows,8 optimizer steps; raw verifier PASS |
| Dynamic smoke |32 accepted mixed,80 generated groups,4 windows,8 steps; raw verifier PASS |
| Real committed-boundary resume |Actual SIGTERM/new parent and native optimizer continuity PASS for both variants |
| Native final reload preflight |Vanilla smoke checkpoint reloaded in a fresh process, Adam step8,558 finite response logprobs |
| Pilot |Fresh512 groups per variant required; exact active IDs in `experiments/stage4/pilot_pair.json` once prepared |
| Formal |Not prepared or launched; both625-window budgets still required |

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

Finish both fresh512-group pilots and monitor512 evaluations; review genuine
clipping, gradients, entropy, format, length, mixed subtypes and resource use.
Capture actual cases and investigate persistent pathology without choosing by
test or selection1024. Perform the additional checkpoint transaction fault
injections required by `CHECKPOINT_AND_RESUME.md`; the two committed-boundary
resume tests do not claim those additional crash timings were tested.

Then freeze one formal pair config and one code commit, declare both fresh SFT
run manifests, and execute exactly5000 groups per variant. Complete raw verifier,
validation/cost curves, native reloads, cases, report, interview narrative and
Stage5 readiness. Final commit/push and clean worktree follow those gates.

Bulk path: `/data/WSH/medical-post-train-artifacts/runs/`. Git holds compact
manifests, receipts, curves and cases. Clone alone does not restore bulk data.
