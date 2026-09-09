# Stage4 progress and continuation

Snapshot2026-09-10 00:01 Asia/Shanghai: **FULL_RUNNING; incomplete.** Stage0 VERIFIED, Stage1–3 DONE, Stage5–6 NOT_STARTED. READY_FOR_STAGE5=NO.

| Formal run | Actual state at launch snapshot | Required budget |
| --- | --- | --- |
| Vanilla `s4_formal_vanilla_20260909T155928_24aec5` | RUNNING; initial monitor512, 0 committed groups |5000 groups /625 windows /1250 optimizer steps |
| Dynamic `s4_formal_dynamic_20260909T155929_18b61a` | PREPARED in the same persistent queue;0 groups |5000 accepted mixed groups /625 windows /1250 optimizer steps |

Both initialize from the original Stage1 SFT SHA`1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64`, with fresh Adam/scheduler/stream. Neither continues a pilot. The unique pair is `stage4_formal_8fb878a48ba4`, frozen at clean code commit`540b777bdd9b3ae8d0f8d32d7b90a567c41ac67e`. The complete config/input/source/model/validation hashes are in `experiments/stage4/formal_pair.json`.

The initial worker PID is2297533; queue PID2297463. These are launch observations, not a substitute for current PID/heartbeat inspection. Queue logs: `/data/WSH/medical-post-train-artifacts/stage4-formal-queue/stage4_formal_8fb878a48ba4/attempt_001/`. Bulk runs are under `/data/WSH/medical-post-train-artifacts/runs/`.

## Current-state commands

```bash
cat experiments/stage4/formal_queue_status.json
cat experiments/stage4/formal_queue_heartbeat.json
.venv-train/bin/python scripts/run_stage4.py inspect --run /data/WSH/medical-post-train-artifacts/runs/s4_formal_vanilla_20260909T155928_24aec5
.venv-train/bin/python scripts/run_stage4.py inspect --run /data/WSH/medical-post-train-artifacts/runs/s4_formal_dynamic_20260909T155929_18b61a
nvidia-smi
```

If the worker is alive, continue monitoring; do not launch a competitor. If dead, inspect the attempt failure, latest committed checkpoint, incomplete actor transaction and raw reservations. `run_stage4.py recover --run ...` preserves orphan costs and performs the tested rollback/adoption. Restart the formal queue only after the audited recovery; it reconciles the immutable commit chain and cached pointer before launching. Unknown unreturned generation/validation cost cannot be silently retried. Do not change the shared scientific config between variants.

The detached queue runs Vanilla5000 → final native reload/full raw verification → Dynamic5000 → final reload/full raw verification → measured analysis/plots. It has no elapsed-time success cutoff. It stops for failures or diagnostic pauses without relaxing budgets. After both raw verifiers, manual cases, retrospective, interview/resume evidence and full stage verifier still remain; no automatic DONE or Stage5 launch occurs.

## Completed entry evidence

- Matched real8×4 optimizer diagnostic; LR1e-6 selected for stability and nonzero second-mini GSPO clip activity.
- Both32-group online smokes and both512-group fresh pilots passed raw verification and real process/native resume. Pilot analysis is in `STAGE4_PILOT_REVIEW.md`; pilot performance is not the formal result.
- Three real SIGKILL transaction recoveries PASS at2026-09-09 15:57UTC. A24 groups/6 effective steps/8 physical steps; B24 mixed groups/6 effective/6 physical; C24 groups/6 effective/7 physical. B adopted the original durable checkpoint with zero optimizer replay; A/C retained orphan work and reused persisted rollout.
- All134 preflight tests PASS. Fault-tested runtime bytes match the frozen formal code.
- Fault A restored exact input identities/RNG and reproduced old arrays exactly, but floating-point optimizer replay was not bitwise identical. Relative adapter L2 difference2.4558e-5 and second-mini differences are retained in `fault_a_replay_numerics.json`. No bitwise GPU backward claim is made.

## Validation, costs and remaining work

Shared monitor512 at training groups0,512,...,4608,5000 and each first committed crossing of the common1M,2M,... training-rollout-token grid. The primary axis is physical prompt tokens once per encounter plus all output tokens, including rejects and overflow. Validation/control costs are separate; actual overshoot is retained, never interpolated.

Measured checkpoint size1,409,125,569 bytes; freeze preflight free3,411,659,776,000 bytes versus2,044,921,839,775 bytes projected with reserve. Full native checkpoints remain available every window. Prior pilot/raw evidence is retained. Pilot-based training-phase estimates are about18h Vanilla and30h Dynamic; each monitor512 evaluation measured roughly11minutes in pilots. The complete pair is conditionally about50–60h, potentially longer if amplification rises. This is not a stopping rule.

Stage4 DONE still requires both actual5000-group budgets, all monitor points, raw verifiers, final reloads, full cost/exposure/case analysis, `docs/stage_reports/04_gspo_training.md`, interview/resume evidence and Stage5 handoff. Selection1024 and all test evaluation remain forbidden in this task.
