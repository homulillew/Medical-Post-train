# Medical-Post-train

A docs-first, evidence-driven project for full post-training of a medical LLM on a single 48 GB GPU.

## Current runtime status

Stage 0 has real Qwen3-8B BF16 LoRA, checkpoint/resume, native verl FSDP2/GSPO,
and vLLM adapter/sleep evidence. The accepted rollout configuration explicitly
uses native sampling and batch invariance after default LoRA numerical failures.
Stage 1 is **DONE**: Qwen3-8B BF16 + r32/alpha64 LoRA trained on all 20,000 unique
examples for one epoch, with 100% coverage and all ten verifier gates passing.
Stage 2 is **FULL_PASS**, pending final verification: the 15k CMExam pool,
200-pair semantic diagnostic, independent 50×4 smoke, and complete 1,000×4
formal rollout with scoring are retained. No policy optimization is performed. Stage 3–6 remain **NOT_STARTED**.
No examination test scoring has run. Current stage status is recorded in
[`project_state.json`](project_state.json).

- [Stage 0 report](docs/stage_reports/00_runtime_compatibility.md)
- [Stage 1 report and interview evidence](docs/stage_reports/01_medical_sft.md)
- [Stage 1 verification receipt](experiments/stage1/verification-final.json)
- [Fixed SFT initialization and hashes](experiments/stage1/initialization_manifest.json)
- [Runtime installation and CLI](env/README.md)
- [Selected evidence](experiments/stage0/selected_runs.json)
- [Failed and successful run inventory](experiments/stage0/run_inventory.json)
- [Measured and conditional compute budget](docs/implementation/COMPUTE_BUDGET.md)

The Stage 0 validator applies to its historical stage-isolation checkpoint;
its archived PASS receipt is retained. Stage 1's archived PASS also retains its
historical isolation boundary. The Stage 2 verifier checks current evidence and
requires the full profiling budget before passing. Bulk models/checkpoints/raw responses are outside Git;
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

## Historical implementation handoff — 2026-09-08

The repository takeover and execution design is in
[`docs/implementation/IMPLEMENTATION_PLAN.md`](docs/implementation/IMPLEMENTATION_PLAN.md).
The audited machine has one **NVIDIA RTX 5880 Ada Generation** with 46,068 MiB visible VRAM.
The plan includes pinned upstream source references, dependency compatibility risks,
Stage 1–6 commands and module interfaces, compute scenarios, durable resume, and evidence gates.
At the original takeover commit, dependencies had metadata resolution evidence only.

At that takeover checkpoint all six stages were **NOT_STARTED**.
The only GPU execution in takeover was a tiny BF16 environment diagnostic; no formal training,
full dataset download, profiling run, or model benchmark was performed.
Future training commands in the plan are interfaces to implement, not existing executables.

Check these planning artifacts with `python scripts/verify_planning.py` (requires `jsonschema`).
This checks the takeover documents, initial state, and contract invariants; it does not verify any Stage completion.

## Current execution

Stage 0 runtime compatibility is [VERIFIED](docs/stage_reports/00_runtime_compatibility.md).
Stage 1 medical SFT is **DONE**. The complete 1,000-example validation NLL changed
from 2.055825 to 1.346031; 50 fixed validation prompts all produced closed SFT
think/answer blocks with no truncation at cap1024. These are source-validation
loss and structural measurements, not medical accuracy. Source reasoning errors
and generated-answer disagreements remain documented in the report.

Current progress is authoritative in
[`project_state.json`](project_state.json), with explicit runs in
[`experiments/stage1/selected_runs.json`](experiments/stage1/selected_runs.json).
Stage 2 full profiling and scoring have completed; selected runs are in
[`experiments/stage2/selected_runs.json`](experiments/stage2/selected_runs.json).
Stage 3–6 have not started. Historical planning/Stage 0/Stage 1 validators intentionally check their original stage-isolation boundaries; their archived receipts are preserved.

Use the independent training environment and real Stage 1 entry points:

```bash
.venv-train/bin/python scripts/verify_stage1_data.py --run experiments/stage1/s1_data_20260908T144854_bac3a7 --output /tmp/new-data-verification.json
.venv-train/bin/python scripts/run_stage1.py prepare --config configs/stages/s1_smoke.json
.venv-train/bin/python scripts/run_stage1.py prepare --config configs/stages/s1_pilot.json
.venv-train/bin/python scripts/run_stage1.py inspect --run <absolute-bulk-run-directory>
.venv-train/bin/python scripts/run_stage1.py resume --run <absolute-bulk-run-directory> --checkpoint <verified-checkpoint-directory>
.venv-train/bin/python scripts/verify_stage.py --stage 1
```

Each prepare creates a new run and detached worker; resume continues the same run with a new attempt and verified optimizer/scheduler/RNG/sample cursor. Formal preparation additionally requires committed configuration and successful smoke/pilot receipts. Do not relaunch smoke/pilot to resume an existing formal run. The original `mpt sft --mode smoke` remains a **Stage 0 synthetic capacity probe**, not the real medical SFT entry point above.

The frozen 20k corpus, token cache, full metrics, generation outputs and checkpoints live under `/data/WSH/medical-post-train-artifacts/`; Git keeps compact evidence manifests and reports. See [Stage 1 execution design](docs/implementation/STAGE1_EXECUTION.md) and [decisions](docs/implementation/STAGE1_DECISIONS.md).
