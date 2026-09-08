# Implementation Plan

## 1. Environment audit

- GPU model / count / VRAM:
- CPU / RAM:
- available disk / durable artifact path:
- CUDA / driver:
- Python:
- existing package constraints:

## 2. Upstream revisions to pin

- Qwen3-8B revision:
- tokenizer revision:
- Transformers:
- PEFT:
- verl:
- vLLM:
- torch:
- MedEmbed model/revision:
- dataset revisions:

Explain why each version/revision is chosen and identify any known compatibility risk.

## 3. Proposed repository architecture

Describe modules/directories for:

- data preparation;
- SFT;
- reward/parser;
- rollout;
- Dynamic Sampling;
- GSPO;
- evaluation;
- serving;
- run metadata/logging;
- artifact manifests;
- verification/tests.

## 4. Stage-by-stage implementation map

For each Stage 1–6 list:

- required code/config;
- smoke command;
- pilot command if applicable;
- formal command;
- planned output paths;
- acceptance/verification mechanism;
- expected dependencies on previous artifacts.

## 5. Data plan

For each source:

- exact upstream identifier/revision;
- fields used;
- sampling/filtering strategy;
- split/dedup strategy;
- estimated sample/token counts;
- local storage estimate;
- license/provenance notes.

## 6. Single-GPU memory/compute plan

Estimate separately:

- SFT model/optimizer/activation memory;
- vLLM rollout memory;
- GSPO actor/backward memory;
- how rollout/training share one 48 GB GPU;
- expected micro-batching/checkpointing/caching choices;
- expected GPU hours for smoke/pilot/formal runs.

## 7. Checkpoint/resume plan

Explain:

- what is saved;
- checkpoint cadence;
- how counters are restored;
- how adapter/model state is synchronized with rollout;
- how resume integrity will be tested before formal Stage 4.

## 8. Experiment evidence plan

Describe how every run will retain:

- unique run ID/class;
- config/command;
- environment/code revision;
- metrics/counters;
- stdout/stderr;
- artifact manifests;
- cases/observations.

## 9. Formal comparison controls

State how Vanilla GSPO vs Dynamic GSPO will be kept comparable and what the exact intended difference is.

## 10. Risks and fallbacks

For each major risk list evidence/trigger and fallback. Include at least:

- BF16 LoRA SFT OOM;
- vLLM/verl single-GPU coexistence issue;
- Qwen3/verl API incompatibility;
- semantic reward too expensive or uninformative;
- RL pool too easy/hard;
- low Dynamic Sampling acceptance;
- long-response runaway;
- checkpoint/resume failure;
- insufficient storage/time.

## 11. Exploration candidates

List optional diagnostics worth trying, expected cost, and what decision each would inform.

## 12. Contract deviations requested

List any proposed change to the project hard constraints. If none, state `None`.

Do not start expensive formal training until this plan is coherent and major compatibility/resource risks have been tested cheaply.
