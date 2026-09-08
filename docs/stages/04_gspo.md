# Stage 4 — Full LoRA GSPO Training

## Why

Execute the project's two primary online-RL experiments under controlled conditions and retain enough evidence to separate policy-learning effects from sampling-cost effects.

## Primary experiments

Both runs start from the **same verified Stage 1 SFT adapter**.

### Run A — Vanilla GSPO

- Dynamic Sampling: OFF
- Group-relative advantage: ON
- GSPO policy loss: ON
- All generated groups may contribute according to the trainer/reward semantics.

### Run B — Dynamic GSPO

- Dynamic Sampling: ON
- Acceptance metric: binary correctness (`R_acc`)
- All-correct/all-wrong groups rejected and refilled
- Same SFT initialization and principal GSPO/reward settings as Run A

The sampling intervention is the main intended difference.

## Initial planned configuration

These are starting values to verify against current `verl`/Qwen3 implementation and hardware:

- model: Qwen3-8B;
- precision: BF16 base where feasible;
- PEFT: LoRA, planned `r=32`, `alpha=32`;
- group size: 4;
- accepted prompts/update target: 8;
- accepted trajectories/update target: 32;
- actor learning rate: ~1e-5 starting point for LoRA;
- advantage estimator: GRPO-style group-relative;
- policy loss: GSPO;
- policy epochs over rollout: 1;
- initial GSPO clip low/high: 3e-4 / 4e-4 if compatible with the current implementation;
- KL reward/loss: planned off;
- entropy coefficient: planned 0, while entropy remains a diagnostic;
- gradient clipping: planned 1.0;
- max response length: planned ~512;
- vLLM/model execution organized to fit one 48 GB GPU.

The implementation agent must inspect the current upstream APIs and may adapt exact config names. Material algorithmic changes require a decision record.

## Reward

Use the verified Stage 2 hybrid reward version for both primary runs. Dynamic acceptance remains based only on `R_acc`.

## Smoke run

A smoke run may be very small (for example a few dozen groups/updates) and should verify:

- rollout -> reward -> advantage -> GSPO loss -> backward -> optimizer step;
- LoRA parameters update;
- old-policy/log-prob/importance-ratio path is sane;
- vLLM/policy weight or adapter synchronization works;
- checkpoint creation;
- structured metrics;
- no immediate OOM/NaN.

**Smoke completion is not Stage 4 completion.**

## Resume test

Before trusting a long formal run, perform a real interrupted-resume check:

1. train for a limited interval;
2. save checkpoint;
3. terminate normally/intentionally;
4. restore;
5. continue;
6. verify optimizer/scheduler/global counters/accepted-group count do not silently reset.

Document the result.

## Pilot run

A pilot of roughly **500–1,000 accepted groups** is allowed and encouraged to validate stability, memory, reward dynamics, response length, and learning-rate reasonableness.

A pilot is not a substitute for the primary full run even if the trend is already visible.

## Mandatory formal budget

Each primary run must reach at least:

**5,000 accepted training groups.**

With `G=4`, this corresponds to at least **20,000 accepted trajectories** per primary run.

For the initial `8 accepted prompts/update` design, this is approximately 625 policy-update batches, but **accepted-group count is the authoritative minimum**, not the approximate step count.

For Dynamic GSPO, rejected refill groups do not count toward the 5,000 accepted-group target, but their rollout tokens/cost must be recorded.

For Vanilla GSPO, define and record the comparable training-group budget before the run so that evaluation can compare both update budget and generated-token budget fairly.

## No early-success stopping

The formal run must not be declared complete because:

- reward increased;
- accuracy appears saturated;
- the loss is stable;
- several checkpoints exist;
- hundreds of steps have completed;
- the agent believes the conclusion is already obvious.

If adaptive early stopping is ever introduced as a formal research change, it requires a contract decision before interpreting the run.

## Valid reasons to invalidate/stop a run

Examples:

- persistent NaN/Inf;
- corrupted reward/parser labels;
- wrong model/adapter initialization;
- serious implementation bug invalidating collected trajectories;
- unrecoverable OOM after documented reasonable mitigation;
- checkpoint state corruption;
- discovered train/test leakage.

Mark the run invalid/failed, preserve evidence, fix, and rerun. Do not lower the formal budget after the fact.

## Required runtime metrics

Retain at least:

- accepted groups;
- generated groups/responses;
- generated output tokens (and prompt tokens if available);
- policy updates;
- total/accuracy/semantic/format reward;
- policy entropy or closest meaningful entropy metric;
- clip fraction;
- gradient norm;
- response length;
- all-correct/mixed/all-wrong ratios;
- sampling amplification for Dynamic run;
- validation accuracy at planned checkpoints;
- wall-clock time;
- peak VRAM/major memory notes.

## Checkpoint policy

Save recoverable checkpoints often enough that a long single-GPU run can survive interruption. Initial suggestion: every ~500 accepted groups or a similarly safe wall-clock interval.

A resumable checkpoint should retain, where applicable:

- LoRA adapter;
- optimizer/scheduler;
- global/policy-update counters;
- accepted/generated group counters;
- generated-token counters;
- random/trainer state needed for a valid resume;
- source run ID/config.

Final/best adapters must be hash/manifest tracked.

## Acceptance criteria

Stage 4 can reach `FULL_PASS` only when:

- Vanilla primary formal run reaches its declared >=5,000 training-group budget;
- Dynamic primary formal run reaches >=5,000 **accepted mixed** groups;
- both start from the same verified SFT initialization;
- reward/version and principal GSPO settings are matched except documented necessary differences;
- neither run contains unresolved NaN/corruption;
- formal structured logs and generated-token counters exist;
- final/best adapters reload and generate correctly;
- resume behavior was validated;
- primary run manifests explicitly state any confounders.

A performance improvement is not required for acceptance.

## Case-study capture

Automatically/manually retain examples of:

- reward hacking;
- response-length explosion or collapse;
- format degradation;
- entropy/correctness-contrast collapse;
- SFT failure corrected after RL;
- SFT correct response regressed after RL;
- strong mixed groups near the policy frontier;
- unusual high/low reward trajectories;
- major OOM/performance/recovery decisions.

## Autonomous exploration

The agent may use pilots/diagnostics to adjust:

- micro-batching/gradient accumulation;
- vLLM memory utilization/cache handling;
- checkpoint interval;
- generation throughput settings;
- LoRA target modules;
- learning rate within a justified range;
- max response length;
- generation temperature shared by both primary formal runs.

If exploration materially changes the primary algorithm/reward/group size/formal budget, create a project-contract decision before launching final formal runs.

## Stage report must answer

1. Why GSPO instead of vanilla GRPO/PPO-style token-level optimization for this project?
2. How are group-relative advantage and sequence-level GSPO loss separated conceptually?
3. What does the old policy represent, and how was it handled without a separate permanent reference model?
4. How did one 48 GB GPU accommodate rollout and training?
5. What was the actual generated-token/update cost of each run?
6. How did entropy, clipping, response length, and group distributions evolve?
7. What did Dynamic Sampling improve or hurt before final evaluation?
8. Which failed/pilot configurations informed the final run?

## Interview-story deliverable

Produce a real engineering/research narrative covering online rollout, group advantage, GSPO sequence-level optimization, single-GPU memory trade-offs, resume/recovery, and at least one training-dynamics case backed by logs.
