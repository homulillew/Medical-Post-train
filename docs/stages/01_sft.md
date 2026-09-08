# Stage 1 — Medical SFT

## Why

Create a strong, reproducible medical-domain initialization for all later RL experiments without spending most of the project budget on SFT.

## Scientific question

Can a compact, high-quality mixed medical SFT set adapt Qwen3-8B to medical instruction following and reasoning while preserving enough unsolved capability frontier for later group-relative RL?

## Required inputs

Planned sources:

- 10,000 Chinese reasoning examples from `FreedomIntelligence/medical-o1-reasoning-SFT`;
- 10,000 Chinese medical instruction/QA examples from `FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K`.

Primary backbone: Qwen3-8B.

Planned PEFT: BF16 base + LoRA, initial target `r=32`, `alpha=32`.

The implementation agent should verify current model/dataset revisions and licenses before execution.

## Mandatory data work

Produce a final SFT train set of **20,000 examples** plus a separate validation set.

At minimum perform and report:

- exact duplicate removal;
- question-level/near duplicate handling between sources;
- empty/corrupted/non-medical obvious-quality filtering;
- token-length filtering/inspection;
- train/validation overlap checks;
- tokenizer statistics using the actual Qwen3 tokenizer.

Required statistics include sample count, total tokens, mean, P50, P90, P95, and max after final preprocessing.

## Mandatory training experiment

Formal SFT must train on the **entire 20,000-example planned train set for one complete epoch** (or an explicitly documented equivalent if packing changes sample counting).

A `max_steps` smoke/pilot run is not a substitute.

## Smoke criteria

A small run, typically 100–500 examples / a few optimizer steps, should verify:

- model/tokenizer load;
- chat/reasoning format;
- finite loss;
- LoRA parameters receive gradients and update;
- checkpoint save/reload;
- sample generation after reload.

## Pilot criteria

Optional but encouraged when memory/performance settings are uncertain. A pilot should establish stable throughput/VRAM and reasonable sequence handling before the full epoch.

## Full-run acceptance criteria

Stage 1 may reach `FULL_PASS` only if:

- final train count is 20,000;
- planned one full epoch completed (>=99% of intended examples/tokens);
- no unresolved NaN/Inf/invalid-loss issue;
- final LoRA adapter exists and reloads;
- exact training config, dataset manifest, code revision, environment, and raw metrics are retained;
- a validation evaluation and generation sanity check exist;
- training did not accidentally consume Stage 2 RL/test data.

The stage does **not** require a specific accuracy improvement to pass.

## Required evidence

Retain at least:

- data selection/split manifests;
- dedup/quality report;
- token statistics;
- SFT formal run record;
- train/validation loss curve/data;
- final adapter manifest/hash;
- a small set of pre/post-SFT generation cases;
- any material implementation/performance decision records.

## Bad/case-study capture

Capture:

- malformed or extreme-length source samples;
- examples removed by dedup/quality filters;
- examples where SFT clearly improves medical response structure/reasoning;
- examples showing undesirable verbosity/format tendencies after SFT;
- any memory/performance failure that changes the final training configuration.

## Autonomous exploration

The implementation agent may freely explore:

- packing vs non-packing;
- LoRA target modules;
- learning rate within a reasonable LoRA range;
- gradient checkpointing/dynamic batching;
- sequence-length cap;
- data mixing/shuffling implementation;
- throughput optimizations.

Material changes to the 20k formal dataset, model backbone, or primary PEFT design require a decision record.

## Stage report must answer

1. Why these two data sources and this scale?
2. What did cleaning/dedup remove, quantitatively?
3. What were the actual token statistics and compute cost?
4. What was the main memory/throughput bottleneck on 48 GB?
5. How do responses differ before vs after SFT?
6. What SFT artifact becomes the immutable initialization for Stage 4 comparisons?
7. What would be changed with more compute/data?

## Interview-story deliverable

Produce a stage report with:

- 30-second summary;
- 2-minute narrative;
- deep-dive notes on data quality, LoRA training, and single-GPU trade-offs;
- 2–5 representative cases backed by real artifacts.
