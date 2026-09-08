# Final Interview Story — Medical Post-training Project

This document is produced only after the project has real Stage 1–6 evidence. It must not invent missing results.

## 1. One-sentence project definition

State the domain, model scale, full post-training scope, central research problem, and resource constraint.

## 2. 30-second answer

Use:

**Problem -> key design -> experiment -> measured result -> lesson.**

Avoid listing every library.

## 3. Two-minute project narrative

Cover:

1. why the project was needed;
2. why Qwen3-8B / single-48-GB design;
3. SFT data strategy;
4. correctness-gated hybrid reward;
5. Dynamic Sampling problem and measured group distribution;
6. why GSPO;
7. controlled Vanilla vs Dynamic comparison;
8. final capability/efficiency result;
9. deployment closure.

## 4. Three strongest technical hooks

### Hook A — Reward design

- original problem:
- key case proving the problem exists:
- final reward design:
- measured effect/limitation:

### Hook B — Dynamic Sampling

- zero/no-correctness-contrast problem:
- actual mixed/all-correct/all-wrong distribution:
- measured sampling amplification:
- update-vs-rollout efficiency result:

### Hook C — GSPO/system

- why sequence-level GSPO:
- single-GPU rollout/training architecture:
- hardest memory/resume issue:
- measured training cost/stability:

## 5. Hardest debugging story

Use STAR-like structure but stay technical:

- symptom;
- evidence;
- hypotheses tested;
- root cause;
- fix;
- validation;
- what changed in the final design.

Link run/decision/case IDs.

## 6. Most surprising experimental result

Separate observation from explanation and include alternative explanations considered.

## 7. Negative result / limitation

Describe a real thing that did not work or did not improve, and why that increased understanding.

## 8. Exact final numbers

| Claim | Value | Run/artifact source | Conditions/caveat |
|---|---:|---|---|
| SFT CMExam accuracy | | | |
| Vanilla GSPO accuracy | | | |
| Dynamic GSPO accuracy | | | |
| Dynamic vs Vanilla delta/CI | | | |
| Sampling amplification | | | |
| Generated-token comparison | | | |
| TTFT/TPOT/throughput | | | |

No number should appear without evidence.

## 9. Likely deep-dive questions

Prepare evidence-grounded answers to:

- Why not simply use GRPO/DAPO?
- What exactly does GSPO change relative to GRPO?
- What remains GRPO-style in your training?
- Why does all-correct/all-wrong matter?
- Does Dynamic Sampling actually save compute?
- Why gate semantic reward with correctness?
- Why is embedding similarity not medical correctness?
- How did you avoid data leakage?
- How did you fairly compare Dynamic vs Vanilla?
- How did one 48 GB GPU handle rollout and training?
- How did checkpoint/resume work?
- What was the most important bad case?
- Which result would make you abandon the method?

## 10. What you would do next

Prioritize 2–3 evidence-motivated extensions rather than naming unrelated algorithms.

## 11. Claims to avoid

List claims the evidence does not support, for example:

- “beats commercial medical assistants” without a controlled protocol;
- “Dynamic Sampling reduces rollout compute” when generated tokens increased;
- “clinically reliable” based on exam benchmarks;
- “distributed FSDP training” on one GPU;
- “proposed a new algorithm” when using/adapting published methods.
