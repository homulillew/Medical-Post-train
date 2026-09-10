# Stage4 progress and continuation

## Update 2026-09-10 14:46 Asia/Shanghai — training resumed

The boundary recovery succeeded after the other GPU task released memory. Worker2533414 and queue2533541 are active. The first recovered window committed2432 groups /304 windows /608 steps at14:24, with native optimizer/RNG continuity and exactly preserved inflight rollout verified in `experiments/stage4/vanilla_formal_resume_verified_001.json`. Current queue snapshot is2496/5000 groups,312 windows,624 optimizer steps. Dynamic remains prepared0/5000. Earlier failure, partial scan and GPU wait evidence are retained. The Stage5 dataset preparation used CPU only and left all49 frozen execution files unchanged; no project-model test evaluation ran.

## Update 2026-09-10 14:10 Asia/Shanghai — boundary verified, waiting for GPU

The recovery I/O fix passed 22 tests and the real 303-window metadata chain plus full checkpoint 0302 verification in **17.8356 seconds**. It hashed 202,982,445 bytes of commit artifacts and 1,409,397,133 bytes of boundary payloads. Historical payload contents are deferred to the unchanged full raw acceptance verifier; this is not a full-history PASS.

Detached supervisor **2529625** now reports `WAITING_FOR_GPU_RELEASE`. Another project's answer-quality evaluation PID2524217 holds approximately5GB on the shared GPU. No foreign process was terminated. The supervisor automatically recovers and starts the original run after GPU release, using an explicit operational restore wrapper around the unchanged frozen worker. All49 frozen execution files and scientific settings remain byte-identical; the runtime restore override is recorded, not hidden. The first resumed commit must still be verified before claiming resumed GPU training. Vanilla remains2424/5000, Dynamic0/5000.

The old assistant-owned read-only history scan PID2516672 was stopped with I/O/state evidence retained. It had no persistent per-file verification cache; its partial scan is not counted as a completed audit. Current logs: `/data/WSH/medical-post-train-artifacts/stage4-formal-recovery/incident_001_attempt_002/`. Evidence: `experiments/stage4/resume_boundary_verification_001.json`, `formal_recovery_launch_002.json`, `formal_recovery_status.json`. Design/trade-offs: [recovery I/O decision](STAGE4_RECOVERY_IO_DECISION.md).

## Update2026-09-10 13:41 Asia/Shanghai — recovery in progress

Vanilla has committed **2424/5000 groups (48.48%)**,303 policy windows and606 optimizer steps. Dynamic remains PREPARED at0/5000. Vanilla stopped at10:32 after the next rollout/reward batch, before actor launch, on `Unexpected GPU residue before actor load`. The original failure and queue failure remain retained. The303 committed windows each measured1.608GiB sleep residue; the failure log suddenly reported5.76GiB at engine level. The source of extra allocation is NOT_ESTABLISHED (external allocation and transient runtime retention are hypotheses). Current GPU was idle at inspection.

The uncommitted window303 contains8 complete groups and8053 output tokens, with no actor launch or optimizer step. It will be reused exactly. Independent detached recovery supervisor PID2516672 is checking the full historical checkpoint chain using the unchanged frozen recovery code. It then launches the original run and reattaches the unchanged formal queue, and verifies the first resumed window's native state, budget and raw-data identity. Cold storage checks read hundreds of GB and are still running; training has **not yet resumed** at this snapshot. Inspect `experiments/stage4/formal_recovery_status.json`; logs are under `/data/WSH/medical-post-train-artifacts/stage4-formal-recovery/incident_001/`. Successful recovery will write `vanilla_formal_resume_verified_001.json`. The helper stops and retains a failure if recovery checks or the new worker fail.

Latest completed monitor512:2048 groups,62.890625% versus initial SFT55.078125%. Earlier measured points include1536 groups64.6484375%; no checkpoint is selected from these measurements. All seven scheduled measurements through256 windows are complete, including first1M/2M token crossings at windows104/207. Current committed training rollout tokens:324153 physical prompt +2593112 output =2917265. Inflight costs remain additional paid work until committed.

Across303 committed windows, mean second-mini clip34.73597%, maximum75%; sequence ratios0.961165–1.029966; recorded optimizer scalars finite. Last32 windows average entropy0.714923 and response length261.79 tokens, largest window P95428.35. No persistent clipping/length pause was triggered. Mean measured training phase103.535s/window projects about9.26h remaining Vanilla training, excluding remaining validation, checksum recovery and verification. Dynamic runtime remains conditional on its actual amplification. Stage4 is incomplete, READY_FOR_STAGE5=NO, Stage5/6 NOT_STARTED.

Evidence: `vanilla_formal_incident_001.json`, original attempt failure, immutable commits/update logs and validation summaries. No scientific config or frozen runtime module was changed. The added supervisor is operational orchestration of existing recovery/launch/queue entry points.

## Retained launch snapshot

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
