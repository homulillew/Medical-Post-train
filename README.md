# Medical-Post-train

A docs-first, evidence-driven project for full post-training of a medical LLM on a single 48 GB GPU.

## Current runtime status

Stage 0 has real Qwen3-8B BF16 LoRA, checkpoint/resume, native verl FSDP2/GSPO,
and vLLM adapter/sleep evidence. The accepted rollout configuration explicitly
uses native sampling and batch invariance after default LoRA numerical failures.
Stage 1–6 remain **NOT_STARTED**; no formal training or test evaluation has run.

- [Stage 0 report](docs/stage_reports/00_runtime_compatibility.md)
- [Runtime installation and CLI](env/README.md)
- [Selected evidence](experiments/stage0/selected_runs.json)
- [Failed and successful run inventory](experiments/stage0/run_inventory.json)
- [Measured and conditional compute budget](docs/implementation/COMPUTE_BUDGET.md)

Run `python scripts/verify_stage0.py --bulk-hashes` in the analysis runtime to
check local evidence. Bulk models/checkpoints/raw responses are outside Git;
their manifests retain exact paths, sizes and hashes. A clone alone does not
restore those local artifacts.

## Research objective

Build and evaluate a complete pipeline around **Qwen3-8B**:

1. Medical LoRA SFT
2. Verifiable medical RL rollout with correctness-gated hybrid reward
3. DAPO-style Dynamic Sampling
4. LoRA GSPO online RL
5. Controlled evaluation and ablation
6. vLLM serving and inference benchmarking

The central research question is not simply whether RL improves a medical benchmark. It is whether **DAPO-style Dynamic Sampling combined with GSPO changes the quality of training updates, rollout cost, and final medical reasoning capability under a constrained single-GPU budget**.

## Execution philosophy

This repository is intentionally governed by stage gates. Successful code execution, a smoke test, or a short partial RL run is **not** project completion.

Each stage progresses through:

`NOT_STARTED -> IMPLEMENTED -> SMOKE_PASS -> PILOT_PASS -> FULL_RUNNING -> FULL_PASS -> VERIFIED -> DONE`

A stage becomes `DONE` only after its mandatory full-run budget, evidence retention, acceptance checks, and stage report are complete.

Implementation details are deliberately left open to the executing agent. The project fixes scientific questions, mandatory baselines, minimum experiment budgets, evidence requirements, and test-set integrity; engineering choices may be explored and improved when documented.

## Documentation map

- [`AGENTS.md`](AGENTS.md): operating rules for coding/research agents
- [`docs/PROJECT_CHARTER.md`](docs/PROJECT_CHARTER.md): scientific scope and project contract
- [`docs/DEFINITION_OF_DONE.md`](docs/DEFINITION_OF_DONE.md): stage-state semantics and completion rules
- [`docs/AUTONOMY_AND_EXPLORATION.md`](docs/AUTONOMY_AND_EXPLORATION.md): hard constraints vs autonomous exploration
- [`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md): run identity, reproducibility, and evidence standards
- [`docs/CASE_STUDY_PROTOCOL.md`](docs/CASE_STUDY_PROTOCOL.md): bad/good/boundary-case capture
- [`docs/ARTIFACT_RETENTION.md`](docs/ARTIFACT_RETENTION.md): what is stored in Git vs external/local artifacts
- [`docs/stages/`](docs/stages/): mandatory stage specifications
- [`docs/templates/`](docs/templates/): run, decision, stage, and interview-story templates

## Planned stages

| Stage | Name | Mandatory outcome |
|---|---|---|
| 1 | Medical SFT | Qwen3-8B medical LoRA SFT on the full planned dataset |
| 2 | Reward & rollout | 15k RL pool + 1k-prompt / 4k-response profiling run |
| 3 | Dynamic Sampling | Accuracy-based group filtering + real refill integration |
| 4 | GSPO | Full Vanilla and Dynamic GSPO runs from the same SFT checkpoint |
| 5 | Evaluation | CMExam + external CMB controlled comparison and sampling analysis |
| 6 | Serving | Final LoRA deployed with vLLM and reproducible serving benchmark |

## Important

This project is designed to preserve **negative results, failed hypotheses, bad cases, configuration changes, full raw metrics, and decision rationale**. A scientifically complete negative result is preferable to an incomplete positive-looking demo.

## Implementation handoff — 2026-09-08

The repository takeover and execution design is in
[`docs/implementation/IMPLEMENTATION_PLAN.md`](docs/implementation/IMPLEMENTATION_PLAN.md).
The audited machine has one **NVIDIA RTX 5880 Ada Generation** with 46,068 MiB visible VRAM.
The plan includes pinned upstream source references, dependency compatibility risks,
Stage 1–6 commands and module interfaces, compute scenarios, durable resume, and evidence gates.
Candidate dependencies have metadata resolution evidence; the training stack is not yet runtime validated.

[`project_state.json`](project_state.json) keeps all six stages **NOT_STARTED**.
The only GPU execution in takeover was a tiny BF16 environment diagnostic; no formal training,
full dataset download, profiling run, or model benchmark was performed.
Future training commands in the plan are interfaces to implement, not existing executables.

Check these planning artifacts with `python scripts/verify_planning.py` (requires `jsonschema`).
This checks the takeover documents, initial state, and contract invariants; it does not verify any Stage completion.
