# Medical-Post-train

A docs-first, evidence-driven project for full post-training of a medical LLM on a single 48 GB GPU.

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
