# Stage 4 execution plan

2026-09-09. Authorized by the owner's Stage4 request (127 numbered sections).
Stage1–3 DONE; fresh Stage3 raw verifier PASS in
`experiments/stage4/prerequisite-stage3.json`. Stage5–6 remain NOT_STARTED.
This is a prospective implementation plan, not a completed stage report.

## Scientific contract and gates

Retain Qwen3-8B revision b968826d9c46dd6066d109eabc6255188de91218,
final SFT SHA1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64,
rank32/alpha64/dropout0/seven projections, BF16 base and FP32 trainable master
weights. Inherit all six frozen Stage2 execution modules and reward manifest.
Reward remains 0.8acc+0.15acc*sem+0.05format; BGE is weak correctness-gated
reference-alignment shaping. Accuracy alone controls Dynamic eligibility.
G4, temperature0.6, top_p1/top_k-1, response cap1024. No new loss/reward terms,
critic, reference worker, KL, entropy bonus, quantization or test usage.

Ordered acceptance: fresh8x4 train-only optimization diagnostic, at most three
initial LR candidates (1e-5,3e-6,1e-6), identical trajectories/rewards/order/old
logprobs/SFT start; Vanilla32 and Dynamic32 mixed online smoke; fresh512-group
pilots for each including actual termination/new-process resume; freeze common
formal config and code; fresh Vanilla5000 and Dynamic5000 accepted; raw verifier,
complete report/cases/curves/readiness; commit and push main. Each formal has
625 windows of8 and, with mini4/one epoch,1250 optimizer steps. Pilot cannot
become formal by continuation. Negative comparisons do not prevent completion.

## Architecture and evidence transactions

`rl/actor.py`: native verl0.9 FSDP2 LoRA engine, native group-relative advantage
and native GSPO loss; wrapper supplies exact persisted microbatch/token evidence.
`rl/rollout.py`: inherited native vLLM and frozen BGE scorer, policy identity and
hash guards, raw outputs committed before scoring. Engine sleeps while the
separate actor process owns GPU, then wakes and loads a unique new adapter ID.
`rl/controller.py`: pair-shared deterministic cyclic15k stream, eight valid
Vanilla or eight mixed Dynamic groups/window, bounded32 refill batches; all
overflow expires with its generating policy. Seeds derive from pair/domain,
encounter and prompt (not variant). Record generated/training exposure separately.
`scripts/run_stage4.py`: durable prepare/launch/inspect/resume. Detached worker,
GPU flock, process identity, heartbeat, owned-worker cleanup, finite failures.

Window order: reserve requests → immutable raw → frozen reward/vectors → select
eight → freeze ALL32 actor old logprobs → native advantages → two optimizer
steps → native optimizer/scheduler/RNG checkpoint and portable LoRA → COMMITTED
marker → vLLM new-ID sync/probe → next window. Persist each window checkpoint
initially (stronger than64-window recovery cadence). Every64 and final retain
validation checkpoint. Checkpoints use hashes, atomic rename and fsync; do not
advance effective training counters before commit. Resume replays committed
lineage and durable generation costs; unreturned in-flight generation has unknown
cost and fails closed, never gets zero-cost silent replacement. Partial optimizer
transactions cannot be counted. Actual crash/failure evidence always retained.

Validation: same frozen CMExam monitor512, initial/each64 windows/final; greedy
n1/cap1024 and identical template/parser. Keep selection1024 for Stage5. Every
point binds cumulative generated prompt/output/total tokens and policy windows;
evaluation/control tokens remain separate from training rollout cost. No early
stopping, no opportunistic change between formal variants.

## D4-001: installed API audit and initial diagnostic choices

ACCEPTED for diagnostic only; contract change NO. Source inspected locally:
`verl/trainer/ppo/core_algos.py`, `workers/engine/fsdp/transformer_impl.py`,
`workers/config/optimizer.py`, and `vllm/v1/sample/sampler.py`.
Official release reference: https://github.com/volcengine/verl/blob/v0.9.0/verl/trainer/ppo/core_algos.py
Installed package hashes, rather than a rolling documentation page, define runtime.

Native advantage uses sample std (`torch.std`, correction1) and epsilon1e-6,
not Stage3 descriptive population std. Native GSPO uses masked sequence mean
log-ratio, clipped objective, seq-mean-token-mean aggregation and its native
stop-gradient formulation; never substitute token PPO. Set exact ActorConfig
`clip_ratio_low=.0003`, `clip_ratio_high=.0004`. Record both sign-aware objective
clip activity and any-bound ratio exceedance; those are different quantities.

vLLM default raw_logprobs precede temperature. Actor recomputes old logprobs
before any update, using the generating adapter and same temperature0.6 as
current logprobs. Use FP32 log-softmax and explicit response/EOS masks. This
wrapper's FP32 division is recorded explicitly (native padded engine casts
temperature to logits dtype). Diagnostic compares raw-temperature1 teacher
forcing to returned vLLM raw logprobs, and inspects temperature0.6 distribution
separately. Never use raw temperature1 rollout scores as temperature0.6 old.
Predeclared initial parity investigation limits: mean absolute token difference
<=.03, P99<=.25, max absolute sequence-mean difference<=.02; failure requires
investigation, not retroactive tolerance widening. Old/current first-mini equality
has stricter1e-6 ratio guard. Native dropout is zero. No importance correction
weight or reference worker introduced; record residual cross-engine precision.

Use AdamW betas(.9,.999), eps1e-8, weight_decay0, gradclip1, constant LR/no
warmup for bounded diagnostic. Pilot choice follows measured drift/clipping,
not reward or validation winner. Save composed upstream config and actual
engine/optimizer dataclasses, and verify no hidden reference/critic/KL paths.

## Resource and risk assessment

Observed idle RTX5880 Ada46068MiB,15MiB used; RAM97GiB available, disk3.3TiB.
Stage3 initial generation projection: Vanilla5.45h, Dynamic10.57h, excluding
actor/old-logprob/switch/checkpoints/validation and policy-dependent acceptance.
Real diagnostic will calibrate those missing phases. Each window full LoRA +
Adam snapshot may cost about1.05GB;1250 formal windows plus pilots about1.5TB.
Preflight free disk>=100GiB reserve and bound total snapshot projection; no
deletion of prior experiments. No formal launch until real pilot resource gate.

Measured correction after smoke: a complete per-window checkpoint contains
both native and portable LoRA copies plus Adam state:1,409,053,142bytes,
about1.41GB/1.31GiB per window.
Budget roughly1.8TiB for1250 formal windows and128 pilot windows, plus raw
evidence, validation and a100GiB free-space reserve. The original1.05GB/window
estimate above was low; preflight must use actual checkpoint bytes and current
free disk. Do not remove earlier run evidence to satisfy the resource gate.

P0: stale adapter sync, wrong temperature/mask, old computed after update,
nonfinite gradients, mixed-policy refill, checkpoint corruption. Assert online,
preserve failure and stop. FSDP optimizer may skip nonfinite gradients upstream;
wrapper must turn this into FAILED, never silently continue. Positive sync
controls at early updates, first resume update, and periodic checkpoints compare
matched probe logprobs/outputs and repeats, beyond API acknowledgement.
R22: first-mini ratio1 is expected; later-mini clipping must activate without
persistent extreme saturation. R23–25 semantic/source limitations remain.
R12: raw costs and effective training counts follow separate ledgers. R27:
release BGE GPU allocations before actor, verify engine lifecycle and owned PID.

## Completion evidence map

Raw verifier replays stream, response tokens/decode, parser and saved BGE vectors,
acc eligibility, old/advantages/current logprobs/sequence ratios/loss, finite
gradients, actual optimizer state and LoRA lineage, sync/resume receipts and
all1250 formal windows. Match pair hashes except sampling_mode, enforce exactly
5000/20000 each and no test/selection use. Retain per-window and cumulative
cost/distribution/format/length/optimization/system metrics and validation curves.
Automatically mine extremes and real repeated-prompt changes; review full cases,
write `04_gspo_training.md` only from actual results. READY_FOR_STAGE5 stays NO
until both formal experiments and every mandatory evidence gate pass.
