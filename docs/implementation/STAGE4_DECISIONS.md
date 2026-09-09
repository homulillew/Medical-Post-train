# Stage 4 decisions and observations

## D4-002 — Conservative shared optimizer candidate after real matched diagnostic

2026-09-09. ACCEPTED for smoke/pilot; formal NOT FROZEN. Contract change NO.
Real newly generated train-only8x4 trajectories are retained in
`s4_optimization_diagnostic_20260909T065945_66b040`. That run FAILED before
any optimizer step when native config contained non-JSON torch dtype/tokenizer.
New run `s4_optimization_diagnostic_20260909T070249_a3a605` reuses that exact
immutable batch, explicitly attributes its original generation cost once, and
runs all three LR candidates from the same SFT with fresh native optimizers.
Original failure/config/source archive and32 responses are retained. Config
serialization now explicitly describes runtime object types/paths and dtype.

Raw verifier `experiments/stage4/diagnostic_verification.json` passed: native
advantage/loss reconstructed from saved tensors, all three old/mask/reward/advantage
arrays exactly equal, all32 old logprobs persisted before the first optimizer
step, every first-mini ratio exactly1, and each complete checkpoint hash checked.

| LR | second-mini ratio min/max | objective clip fraction | any-bound exceedance | mean absolute token drift | parameter L2 delta |
| --- | --- | --- | --- | --- | --- |
| 1e-5 | .698170 /1.005568 | .1875 | .8125 | .0489714 | .151061 |
| 3e-6 | .933328 /1.006385 | .1875 | .9375 | .0296531 | .0456183 |
| 1e-6 | .994234 /1.010573 | .2500 | .9375 | .0255196 | .0151302 |

Select LR1e-6, mini4 prompts, one epoch, clip_low.0003/high.0004, AdamW
betas.9/.999 eps1e-8 weight_decay0, clipgrad1, constant LR/no warmup for both
variants' smoke and pilots. Lower LR reduces observed parameter and mean drift,
avoids the .698 extreme, and retains objective clip activity. This is not a
validation/reward winner selection. There is no independent Dynamic optimizer.

Objective clipping requires both ratio direction and advantage sign; any-bound
exceedance counts both directions. A .9375 exceedance does not mean93.75% of
trajectories have their improving objective clipped. Both metrics remain visible.
BF16 forward quantization means individual ratio movements are not monotone in
LR; even1e-6 has many ratios outside the tight interval. This is an open pilot
stability/precision limitation, not a claim that smaller LR solves every risk.
One batch and six diagnostic steps cannot prove long-run stability.

## O4-001 — Actor/rollout raw parity and temperature boundary

All32 trajectories: temperature1 vLLM raw vs actor teacher-forced mean absolute
token difference .01935215, P99 .10998935, max .28433657; maximum absolute signed
sequence-mean difference .00550873. This passes the predeclared investigation
limits (.03/.25/.02 respectively) but is not bitwise cross-engine parity. Both
optimization old/current use actor FP32 temperature.6 log-softmax; all32 old
values freeze before any optimizer step. No returned vLLM temperature1 scores
enter the importance ratio. Native FSDP2 and native GSPO loss execute; no full
Ray trainer, reference, critic, KL or entropy bonus is instantiated.

## O4-002 — Historical verifier lifecycle limitation

Fresh Stage1/2 raw rechecks pass every scientific data/training/reload/case gate,
but retain FAIL because each original stage verifier demands all later stages
NOT_STARTED. Those assertions are incompatible with legitimate Stage2/3 DONE.
The original FAIL receipts remain unmodified. Separate
`scripts/audit_stage4_prerequisites.py` requires exactly those expected failed
gates, rehashes every contract/base/SFT file, requires existing PASS receipts and
Stage1–3 DONE/Stage5–6 untouched, then issues a scoped predecessor handoff PASS.
Fresh full Stage3 raw verifier already passed all9 gates before Stage4 work.

## Recovery implementation boundary

Online smoke saves every window's native LoRA/Adam/scheduler/extra checkpoint,
explicit Python/NumPy/CPU/CUDA RNG, and portable adapter before committing budget.
Actor reload must match trainable digest, optimizer-state digest/steps and
scheduler. The controller deliberately pauses after two windows for actual
SIGTERM and a fresh parent process; both variants must repeat this check.
This section describes the required implementation, not a claim of passed
online smoke or resume. Results will be appended only after verification.

## O4-003 — Completed Vanilla smoke and numerical verifier investigation

`s4_smoke_vanilla_20260909T071222_ef95d2` completed32 training groups,
128 trajectories,4 policy windows and8 optimizer steps. It generated32693
output tokens and4056 physical prompt tokens. An external observer actually
terminated PID2149881 after window2 and observed a new parent PID2152015
restore identical controller/stream state. Subsequent native actors restored
Adam, scheduler, trainable digest and explicit RNG. Raw verification passed;
an additional independent check recomputed every positive adapter-sync delta
and repeat control from the saved prompt logprobs. Ratios ranged
.991764307–1.008340478; mean minibatch objective clip fraction was.15625.
This is smoke evidence only; both formal budgets remain zero.

The first raw verifier attempt failed its CPU/GPU advantage allclose check.
For small-variance all-correct hybrid-reward groups, the largest measured CPU
versus GPU discrepancy was5.67436e-5 in window0 and5.53131e-5 in window3.
FP64 normalization gave maximum GPU discrepancy2.28775e-5 and2.34190e-5
respectively. FP32 reduction order is amplified by dividing by a small std.
No rewards or training tensors were changed. The verifier now independently
checks FP64 sample-std normalization with the forward-rounding bound
`max(1e-6, 8*eps_float32*max_abs_reward/(sample_std64+1e-6))`, requires zero
advantages for constant groups, and exact scalar-to-token mask broadcasting.
Native GSPO is then recomputed with those validated saved GPU advantages.
This changes the verifier's numerical oracle, not the training algorithm or
the predeclared actor/vLLM parity thresholds.

An earlier Vanilla run `s4_smoke_vanilla_20260909T070935_4a09cb` failed before
the first optimizer step because an orchestration record contained a Path
object. Explicit path serialization fixed the issue. Its raw generation,
error, configuration and artifact manifest remain retained under its own ID.

## O4-004 — Shared first encounters reproduce identity, not every sampled token

First8 smoke encounters share prompts, request seeds, initial adapter SHA,
native trainable-parameter digest and engine configuration. Of32 responses,
29 have identical token IDs and28 have identical raw logprobs across the two
fresh vLLM processes. The initial strict equality check therefore failed.
`experiments/stage4/smoke_first_window_matching.json` retains every match flag.
Stage3 already observed analogous cross-process stochastic variation
(106/128 exact responses). Its numerical/scheduling cause remains unresolved.
This is a common-randomness limitation, not evidence of different SFT weights.
No custom sampler is introduced to force identical text. The pair must retain
the same runtime, native request seeds and initial identity controls, and
report actual first-encounter matching rates. Persisted-checkpoint resume
reuses saved raw outputs; it does not depend on regenerating identical text.

## O4-005 — Completed Dynamic smoke and the pilot handoff

`s4_smoke_dynamic_20260909T071807_adad34` completed32 accepted mixed groups,
128 training trajectories,4 policy windows and8 optimizer steps. It generated
80 groups/320 responses,80039 output tokens and10536 physical prompt tokens.
Output-token dispositions:32148 selected,25442 rejected all-correct,
14654 rejected all-wrong,7795 expired eligible overflow. Amplification2.5.
Actual parent termination/restart changed PID2165870 to2168220 after16
training groups. Raw selection, loss, optimizer lineage and positive sync
controls passed independent verification. Ratios ranged.981803656–1.010007143;
mean minibatch objective clip fraction.2421875. These are four-window smoke
measurements, not evidence of long-run stability or Dynamic superiority.

Manual full-response review of train27714 is retained in
`smoke_manual_format_case.json`: all four visible final answers are D, but
one duplicates an answer tag inside think and is unparseable under the frozen
parser. This produces an accepted3/4 group without demonstrating disagreement
about the visible final option. Medical explanation claims were not assessed.

Pilot handoff adds fixed monitor512 evaluation at fresh SFT and window64,
native final-checkpoint reload in another process, source checks on the runtime
launcher, duplicate-owner protection, and a sequential detached pilot queue.
Incomplete actor rollback preserves raw rollout and archives orphan optimizer
artifacts; completed actor checkpoints may be reused before redoing sync.
This recovery extension is distinct from the two physically tested committed-
boundary resumes. Unreturned generation/validation costs remain unresolved and
cannot be silently replaced with zero. Frozen reward/math/sampling are unchanged.

Smoke phase-time projection (generation, reward, actor process, sleep/wake and
sync): Vanilla105.7s/window; Dynamic approximately161s/window. This suggests
about1.9h and2.9h for64-window pilots, or18.3h and28h for625-window formal
runs, excluding validation, initialization and recovery overhead. Small-smoke
Dynamic acceptance is uncertain and must be recalibrated after the pilots.
