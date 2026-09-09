# Stage4 formal execution and recovery plan

This is a prospective execution protocol, not a formal stage report. Stage4 remains PILOT_PASS until a formal worker starts. The two 5000-group results, final analysis and Stage5 handoff are still required.

## Scientific configuration

The candidate in `configs/stages/s4_formal_shared.json` inherits the verified pilot: original Stage1 Qwen3-8B SFT, rank32/alpha64 LoRA, unchanged Stage2 gated hybrid reward/BGE-M3, G4, temperature0.6, response cap1024, 8 training groups/window, mini4, one epoch, LR1e-6 constant, GSPO clip0.0003/0.0004, gradient norm clip1. No KL, critic, reference, entropy bonus or quantization is introduced. Accuracy-only filtering/refill is the sole intended intervention.

Each fresh formal run has exactly 5000 training groups, 20000 training trajectories, 625 committed policy windows and 1250 optimizer steps. Diagnostic and pilot checkpoints are excluded from initialization. Native optimizer state must be empty at window0 and match the previous checkpoint thereafter.

## Entry gates and commands

1. Audit remote/main and the existing pair before preparing anything. The initial audit is `experiments/stage4/formal_entry_audit.json`.
2. `scripts/fault_test_stage4.py prepare` allocates three unique DIAGNOSTIC runs. The detached `run` supervisor performs actual SIGKILL and native actor recovery. Target24 groups per diagnostic is an explicit recovery-test budget; none contributes to formal5000.
3. A: kill after temp files and marker are durable, before rename. Preserve temporary evidence and orphan optimizer events, return to previous native checkpoint, and reuse the exact persisted rollout.
4. B: kill after rename, before actor result/window publication. Validate the full checkpoint tree and source references, load its exact adapter/Adam/scheduler/RNG in a new native actor, add zero optimizer steps, sync the adopted adapter to vLLM, then publish one logical window.
5. C: kill after the first real optimizer step of window1, before a valid new checkpoint. Effective state stays at 8 groups/1 window/2 steps; retain that orphan step in physical cost and resume the prior checkpoint.
6. All diagnostics continue to 24 groups/3 windows/6 effective steps. Physical optimizer counts should be A8, B6, C7. Raw generation must not repeat and all native identities/cursors must be verified. Save receipts, PIDs, checkpoint tree, costs and source manifests.
7. Record D4-FORMAL-FREEZE only after all three real gates PASS. Commit reviewed code, protocol and evidence. `scripts/prepare_stage4_formal.py freeze` refuses a dirty workspace or an existing formal pair, verifies input files and fault-tested runtime bytes, then prepares both manifests at the same clean source commit.
8. Detach `scripts/continue_stage4_formal.py`. It runs Vanilla, waits for final native reload and complete artifact sealing, replays the raw verifier, then launches the already frozen Dynamic run. It never changes scientific config based on Vanilla validation.

Creating the first manifest necessarily makes the workspace operationally dirty before creating the second. `formal_pair.json` records the clean state immediately before either manifest, identical source hashes, and the same source commit. This is distinguished explicitly from the raw dirty-state snapshots in run manifests.

## Validation and token accounting

Use the frozen Stage2 monitor512 only. Common group schedule: 0,512,1024,1536,2048,2560,3072,3584,4096,4608,5000. The primary generated-token grid is every positive integer multiple of 1000000, deterministically continued until each run ends. The grid therefore covers the actual Vanilla range without guessing its final token count or inspecting scores.

The token denominator is physical training prompt tokens once per encounter plus all generated output tokens, including rejected and overflow responses. Logical prompt tokens multiplied by G are secondary. Validation and sync/control tokens are separately retained and excluded from the training-token axis. At the first committed window crossing a threshold, save its actual counter/window/policy and evaluate immediately. Multiple crossed thresholds and any group milestone share one monitor evaluation. Resume replays this same boundary before further training. Only thresholds reached by both variants support the shared-threshold comparison; actual token overshoot is always shown. No interpolation or later checkpoint substitution is allowed.

## Monitoring and pauses

Finite reward components, advantages, old/current logprobs, ratios, losses, gradient norms and entropy are mandatory. Fail closed on nonfinite values. The shared health protocol pauses after eight consecutive second-mini objective clip fractions strictly above0.9, or three consecutive nonoverlapping16-window blocks whose actual generated-response P95 is at least90% of the1024 cap. One high window does not trigger persistent-collapse claims. Pauses retain the committed checkpoint and require a documented diagnosis before any further rollout. These checks never change reward, LR, sampling eligibility or formal budget. Entropy remains a measured diagnostic without a bonus.

Dynamic keeps the inherited maximum32 generation batches per window. Starvation is BLOCKED with all paid raw output retained. No all-correct/all-wrong acceptance or budget reduction is authorized.

## Ownership, recovery and resources

The launcher checks previous owner PIDs, a per-run launch lock and shared GPU flock. Workers and the formal queue have separate durable heartbeats. A new session first inspects `formal_queue_status.json`, each active run's `status.json`, heartbeat, owner PID, GPU memory and `checkpoint.json`. A live owner must not be duplicated. For a dead owner, inspect failure, reservations/raw returns and incomplete actor/checkpoint/sync before invoking `run_stage4.py recover` and `launch`.

Unknown unreturned generation or validation cost fails closed; no zero-cost retry is invented. `transactions.cost_ledger` reconstructs physical generation, archived optimizer events and control work separately from the effective committed counters. Orphan work remains in its recovery archive. Unrecorded crash-tail timing stays unknown.

Native checkpoints are retained every window because the one-GPU actor process reloads them between windows. The freeze preflight uses measured checkpoint size ×1250 ×1.1 plus100GiB reserve, against actual free disk. Prior pilot evidence is never removed. The pilot-based18h Vanilla/30h Dynamic training-phase projection excludes extra formal validation and possible amplification changes; 50–60h pair is a planning estimate, never a stop condition.

## Completion and handoff

After both runs reach5000, reconstruct every group/reward/advantage/native loss/checkpoint/sync and every group/token-triggered monitor point. Retain all measured cost components, exposure histograms and repeated-prompt histories; population-level changes alone are not proof that individual questions were learned. Create formal curves, manual good/adverse cases, the retrospective `docs/stage_reports/04_gspo_training.md`, interview/resume evidence and checkpoint handoff. Stage4 can become DONE only after the full verifier and all documented scientific/report gates pass.

No Stage5 selection1024, CMExam/CMB test, checkpoint selection or Stage6 execution is permitted in this task. READY_FOR_STAGE5 stays NO until the complete formal evidence and handoff are verified.

## Observed replay precision limitation

Fault A restored the prior adapter/Adam/scheduler/RNG identities and reproduced all saved old arrays exactly, but the discarded and recomputed output adapters were not byte-identical. Measured adapter relative L2 difference was2.4558e-5, maximum element difference2.2851e-6. The second-mini logprob differences reached0.38448 and sequence-ratio differences0.008412; therefore this is not described as bitwise update reproducibility. GPU backward/reduction nondeterminism is a hypothesis, not an established root cause. The full measurement is `experiments/stage4/fault_a_replay_numerics.json`. Effective recovery starts from the trusted prior state; orphan output is retained and excluded from the authoritative lineage. This also motivates adopting the exact durable checkpoint in Fault B rather than recomputing it.

The formal queue also runs `scripts/audit_stage4_boundary.py` before launching a PREPARED/INTERRUPTED owner. It verifies the immutable synced commit chain and repairs a stale convenience pointer with an audit receipt if a prior process died between window commit and pointer publication. It never treats an unsynced actor checkpoint as an effective commit; that still follows the tested adoption/recovery mechanism. This covers the final-window boundary, where there may be no later update to refresh a stale pointer before final native reload.
