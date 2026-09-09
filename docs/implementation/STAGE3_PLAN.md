# Stage 3 execution plan and decisions

Date: 2026-09-09. Status: implementation authorized; formal requires real smoke and restart PASS. Contract change: NO.

## Inputs and scope

Stage 0 VERIFIED, Stage 1/2 DONE at entry. Fresh prerequisite receipt: `experiments/stage3/prerequisite-stage2.json`. Inherit `configs/stages/s2_formal_1024.json` model, engine, sampling, six execution hashes, candidate pool, initialization and reward manifest verbatim. Stage 3 is fixed-policy generate/reward/classify/filter/refill integration with zero optimizer updates; Stage 4–6 remain NOT_STARTED. No held-out/test access, reward tuning, difficulty pruning, reference model, advantage or loss computation.

## Upstream audit and decision D3-001

The [official verl DAPO recipe](https://verl.readthedocs.io/en/latest/algo/dapo.html#dynamic-sampling-with-group-filtering) specifies equal-accuracy group rejection followed by repeated generation until enough eligible groups or a finite generation-batch limit. Its training runner also executes actor optimization, which is outside this stage. Installed verl 0.9.0 does not expose that recipe's filter in `verl/trainer` (source search recorded during audit). Therefore reuse the proven vLLM generation, parser, semantic encoder, hybrid reward and group statistics modules, with a thin independently tested durable sampler controller. Do not transplant a trainer or add other DAPO components. Upstream main APIs are not assumed interchangeable with installed 0.9.0.

## Environment and resource plan

One idle RTX 5880 Ada, 46,068 MiB; torch 2.11.0, vLLM 0.24.0, Transformers 5.5.3, PEFT 0.18.1, verl 0.9.0, sentence-transformers 5.3.0, Python 3.12 in `.venv-train`. Durable bulk `/data/WSH/medical-post-train-artifacts/runs`, about 3.3 TiB free. Stage 2 measured about 31.8 GB generation peak and 4.0 GB separate BGE peak. Plan keep vLLM resident and load BGE on GPU after policy identity/sleep-wake controls; expected combined <40 GB, measured in smoke. No optimizer/activations for training. If coexistence fails, preserve failure and use sequential scorer unloading in a new engineering run without changing reward. Stage 2 forecast: ~482 groups / 1928 responses / ~458k output tokens, ~0.70 h generation at 182 tokens/s. These are estimates, never stopping budgets. Smoke32 + restart overhead ~5–10 min; formal expected <1 h; raw/vector evidence <1 GB. Actual wall, token costs and peak memory replace forecasts in final report.

## Concrete implementation map

- `sampling/dynamic.py`: complete G4 validity independent of acc-only eligibility; deterministic cyclic full-pool stream; encounter IDs, policy barrier, state counters, exact-target overflow and starvation. 16 patterns plus reward/length adversarial and invalid/resume/wrap tests.
- `sampling/stage3.py`, `scripts/run_stage3.py`: unique run manifests, immutable source/config hashes, source archive, per-process environment/commands/logs/identity controls; 16-prompt batches, max128 formal batches. Stream domain `stage3:formal` with seed42 and epoch-specific SHA ordering of all15k; smoke independent `stage3:smoke`. Exposure is descriptive, never affects sampling. Prompt IDs may overlap, fresh encounters and generation always required; report Stage2 and smoke overlap.
- Durable batch reservation -> raw response file -> scored rows/vectors -> classification -> immutable commit/state_after -> replaceable checkpoint. Resume replays commits, verifies cached state, reuses persisted raw batch if scoring interrupted; an interrupted generation without a raw result is explicitly unresolved/FAILED, never silently retried with fabricated zero cost.
- Smoke32: pause after first committed16, record next IDs/counts/costs, external SIGTERM to owned process group, new PID same run and source resumes at next reservation. Pass only after32 and preserved state/unique IDs/ledger checks.
- Formal fresh run: exactly256 accepted mixed, extra last-batch mixed tagged overflow_eligible; retain all wrong/correct/invalid/overflow trajectories. Stop at target or max128 batches; starvation => BLOCKED. Any actual invalid group ends FAILED with evidence retained.
- `sampling/analysis.py`, analysis CLI: counters, token/length/parser/reward/correctness variance, 0..4 counts, accepted1/2/3, mixed subtypes, cardinality and exposure ledgers, elapsed generation/worker time, policy controls separately accounted. >=50 full trajectories reviewed by agent; 2–5 full frontier cases, category coverage including NOT_OBSERVED.
- `scripts/verify_stage3.py` through `verify_stage.py --stage 3`: recompute from raw outputs, token IDs, frozen parser/reward and saved BGE vectors, replay eligibility/counters/stream/checkpoints; check frozen manifests, real restart, tests, raw conservation and reports. Seal all bulk evidence after final process exit.
- `docs/stage_reports/03_dynamic_sampling.md`, compute budget, resume evidence, readiness, project state, git commit and push. DONE only after all evidence and verifier PASS.

Commands: `.venv-train/bin/python scripts/run_stage3.py prepare --mode smoke`; `launch --run <bulk>`; `terminate --run <bulk>` at pause; `launch --run <same>`; independent `prepare --mode formal` then `launch`; analysis and `.venv-train/bin/python scripts/verify_stage.py --stage 3 --output experiments/stage3/verification-final.json`.

## Risks / acceptance gates

- vLLM identity drift: inherited split_k=1 and batch-invariant env, base/SFT/negative/repeat/sleep-wake checks each process; stop if identity fails.
- BGE coexistence OOM: smoke memory evidence first, preserve failed run. No truncation/reward substitution.
- Eligibility contamination: pure function uses only binary acc after separate validity; exhaustive16 and randomized extraneous-field tests. Unparseable remains0; mixed subtype never drives decision.
- Data pruning/cursor drift: deterministic full-pool cyclic permutation, unique encounter indexes, checkpoint replay and epoch/policy tests; no blacklist state.
- Lost/duplicated costs: immutable raw and commit transactions, fsync, physical restart smoke and verifier recomputation. Interrupted in-flight generation has unknown cost and fails closed.
- High amplification: finite128x16 bound; never lower256 budget. Overflow costs retained.
- Stage 2 comparison: different independent slice, batch size16 versus4 affects throughput; no causal accuracy/training or hardware-utilization claims. Reward/model/cap/temperature unchanged.

All three mandatory future baselines and 5000-group full RL budgets remain intact. Stage4 batch/LR/clipping decisions are deferred to Stage4 smoke/pilot.

## Decision D3-002: retain terminal smoke evidence-writer failure

First smoke `s3_smoke_20260909T051456_a53a8c` really generated32 groups and14 mixed, including real SIGTERM/new-PID resume. Its final evidence writer raised `TypeError: dict() got multiple values for keyword argument run_id` after durable batch commits. Cause: Stage3 summary already supplies run_id whereas the copied run writer supplied it again. Fix only metadata merging, covered by a regression test. The run remains FAILED with raw groups, process logs, source archive and recovered descriptive costs; it does not satisfy the smoke gate. A fresh32-prompt smoke with the same science and deterministic stream repeats the real restart before formal. This is an engineering failure, not a model/reward change. Extra smoke/reload cost is retained.

## Decision D3-003: physical exit gate and engine shutdown

The first failed worker's Python process remained alive with its vLLM child and BGE allocations after summary TypeError. A second smoke `s3_smoke_20260909T052138_e6717b` consequently failed at engine startup: only12.09GiB free versus28.85GiB required, with zero generation requests. Preserve both failures. Cleanup only the verified owned process group. Add prelaunch GPU-empty assertion, explicit installed vLLM `engine_core.shutdown(timeout=30)` at successful generation end, and owned-group SIGTERM after persisting any worker exception. The third fresh smoke repeats the complete pipeline/restart with these lifecycle fixes. No memory-utilization, model or reward parameter is changed. Failure startup time is retained separately; no unknown token tail is treated as zero.

## D3-004: accepted smoke and stochastic replay boundary

`s3_smoke_20260909T052408_5114dc` passed real32x4 smoke plus real16-group termination/new-process resume and all four raw-verification gates. It generated13 mixed,11 all-wrong,8 all-correct,0 invalid. Successful explicit shutdown released GPU to15MiB. All98 repository tests passed. Formal inherits exactly its execution/controller hashes and freezes a separate `stage3:formal` stream/counters.

Comparing the earlier failed-at-summary32-group smoke with this successful run under the same prompt/seed/scientific settings:106/128 raw trajectories are byte-identical and2/32 acc vectors differ. The root numerical/scheduling cause was not established. Therefore seed/stream determinism and stable policy identity must not be described as bitwise stochastic vLLM regeneration determinism. This does not invalidate persisted-state resume: no previously committed group is regenerated. Save paired evidence in `experiments/stage3/cross_process_sampling_observation.json`; do not tune policy/reward on these observations.
