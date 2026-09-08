# Project Charter

## 1. Project mission

Build a reproducible, evidence-rich **full post-training pipeline for a medical LLM on one 48 GB GPU**, using Qwen3-8B as the planned backbone, while studying a focused research question rather than maximizing benchmark score at any cost.

The planned pipeline is:

`Medical SFT -> verifiable rollout/reward -> DAPO-style Dynamic Sampling -> LoRA GSPO -> controlled evaluation -> vLLM serving`

## 2. Primary research question

Under a constrained single-GPU budget, what happens when DAPO-style Dynamic Sampling is introduced into GSPO-based group-relative online RL for medical reasoning?

The project must characterize at least four effects:

1. **Capability:** Does the final policy improve held-out medical reasoning accuracy?
2. **Update quality:** Does filtering groups without correctness contrast change the usefulness of policy updates?
3. **Rollout cost:** How much extra sampling/refill is required to obtain accepted groups?
4. **Training dynamics:** How do all-correct, mixed, and all-wrong group distributions evolve as the policy changes?

The project must not collapse this question into a single final benchmark number.

## 3. Planned backbone and resource envelope

Primary target:

- Backbone: Qwen3-8B
- Training precision: BF16 where feasible
- Parameter-efficient training: LoRA, planned rank 32
- Hardware assumption: one 48 GB CUDA GPU
- SFT target: 20,000 selected examples, one complete epoch
- RL candidate pool: 15,000 CMExam prompts
- Primary group size: 4
- Formal GSPO target: 5,000 accepted groups per primary run unless the project contract is explicitly amended before the run

Exact engineering choices may change when hardware/library evidence requires it, but material changes require a decision record.

## 4. Mandatory primary comparison

The final study must retain three principal model states:

1. **Medical SFT**
2. **Vanilla GSPO** initialized from the Medical SFT checkpoint
3. **Dynamic GSPO** initialized from the same Medical SFT checkpoint, using DAPO-style accuracy-group filtering/refill

For the Vanilla vs Dynamic comparison, hold constant as much as reasonably possible:

- initial checkpoint;
- reward definition;
- model/LoRA architecture;
- group size;
- GSPO policy loss settings;
- optimizer family and nominal learning rate;
- generation policy unless a documented experiment establishes a necessary shared change.

Dynamic Sampling should be the principal intervention, not one change among many.

## 5. Planned reward hypothesis

The initial reward design is a correctness-gated hybrid reward:

`R = 0.8 * R_acc + 0.15 * R_acc * R_sem + 0.05 * R_format`

Where:

- `R_acc`: verifiable ground-truth answer correctness;
- `R_sem`: semantic alignment between model reasoning and the reference explanation, initially planned via MedEmbed;
- `R_format`: response-format validity.

The gating is intentional: semantic similarity must not rescue an incorrect final answer.

This formula is a starting research design, not an immutable implementation constant. A material change requires a decision record and a controlled justification. Dynamic Sampling must use correctness/accuracy as its group-filtering metric, not the total hybrid reward, unless the project contract is explicitly revised.

## 6. Scope boundaries

The following are explicitly out of scope for the mandatory project path:

- RAG;
- tool-use/agent training;
- DPO/ORPO/PPO benchmark suites;
- process reward models;
- separate learned reward model training;
- OPD/distillation;
- multi-GPU FSDP claims;
- quantization research;
- TensorRT deployment;
- speculative decoding.

They may appear only as optional explorations after mandatory work is protected.

## 7. Scientific integrity rules

- A negative or null result is a valid final result if the protocol is completed.
- Test data must never guide checkpoint selection or hyperparameter tuning.
- Failed experiments and regressions must be retained, not erased.
- Report generated-token cost and update budget separately.
- Do not claim Dynamic Sampling reduces rollout compute unless measured generated-token evidence demonstrates it.
- Do not describe a short smoke/pilot run as formal RL training.
- Resume interrupted full runs where valid checkpoints allow it; do not silently replace incomplete formal runs with shorter substitutes.

## 8. Success criteria

The project succeeds when it produces a reproducible body of evidence, not merely working code. At minimum:

- a complete SFT run;
- validated reward/rollout pipeline;
- validated Dynamic Sampling refill behavior;
- full Vanilla and Dynamic GSPO runs;
- controlled held-out evaluation;
- efficiency/training-dynamics analysis;
- deployable final LoRA adapter;
- serving benchmark;
- retained cases, observations, decisions, and interview-ready stage reports.

The strongest outcome is a clear causal story supported by real artifacts. The project remains complete if Dynamic Sampling fails to improve the final score, provided the mandatory experiments and evidence are complete.
