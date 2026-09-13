# Medical-Post-train

A docs-first, evidence-driven project for full post-training of a medical LLM on a single 48 GB GPU.

## Current runtime status

Stage 0 has real Qwen3-8B BF16 LoRA, checkpoint/resume, native verl FSDP2/GSPO,
and vLLM adapter/sleep evidence. The accepted rollout configuration explicitly
uses native sampling and batch invariance after default LoRA numerical failures.
Stage 1 is **DONE**: Qwen3-8B BF16 + r32/alpha64 LoRA trained on all 20,000 unique
examples for one epoch, with 100% coverage and all ten verifier gates passing.
Stage 2 is **DONE**: 15k CMExam candidates, a 200-pair semantic diagnostic,
50×4 smoke, and full 1,000×4 profiling with 80 complete qualitative reviews
and all eight verifier gates passing. Measured train-sample accuracy is 51.775%,
mixed groups 53.1%, and truncation 0%. Stage2 performed no policy optimization.
Stage 3 is **DONE**:496 newly generated groups,256 accepted mixed and1 retained overflow,
with1.9375× measured amplification and all nine verifier gates passing.
Stage 4 has reached **FULL_PASS**, with both formal runs at **5000/5000 groups**,
625 windows and 1250 optimizer steps. Both original full raw verifiers and final
native reloads passed. Stage-level acceptance and narrative/manual-review
deliverables remain incomplete; Stage 5 readiness remains **NO**.

The [postformal analysis](experiments/stage4/postformal_analysis_v1.json) rebuilds
all frozen group milestones and first committed shared token crossings, with
[CSV/figure sources](experiments/stage4/postformal_data_v1/manifest.json).
Final Dynamic minus Vanilla monitor512 accuracy is +0.9765625 percentage points;
the paired exploratory 95% bootstrap interval is [-2.54, +4.49] points and exact
McNemar p=0.661. The largest observed group-grid gap is at 4096 groups (+4.10 points),
not confined to early training. Dynamic wins only the 4M crossing among shared
1M–5M token thresholds. These are single-seed validation observations, not test claims.
Dynamic generated 14,872 unique prompts (no reuse in the actual exposure map),
with 2.9744× group amplification and 2.92454× rollout-token cost relative to Vanilla.
Mean response length is 259.57 versus 264.59 tokens, so added cost comes primarily
from more groups rather than longer responses.

A separately [preregistered frontier diagnostic](experiments/stage4/frontier_diagnostic_v1/preregistration.json)
is running on exactly 1000 clean validation-reserve prompts, three fixed final
policies and G=4 (12,000 responses, **zero optimizer updates**). The frozen set has
zero cluster overlap with monitor512, selection1024, RL pool, SFT train/validation,
and sealed test clusters. Diagnostic generation and verification must finish before
mechanism conclusions are made. [Machine-readable handoff](experiments/stage4/postformal_handoff_current.json).
The Random-3× source schedule is retained; its full protocol awaits the truncated
owner instruction. No auxiliary training has started. Final narrative authorship
belongs to ChatGPT/user; machine evidence is not a substitute for completed review.

A GPU-release assertion stopped training at3216 groups on September10 17:56;
the overnight failure and stale project status are retained. The saved next
checkpoint was adopted without optimizer replay, and a fresh window passed.
The persistent queue remains active. See the
[incident, recovery and progress analysis](docs/implementation/STAGE4_GPU_RELEASE_RECOVERY.md).
Operational wrappers restore the audited boundary and wait at most60 seconds for
real actor GPU release, preserving the original4GiB threshold. The full acceptance
audit remains mandatory. The49 frozen execution files and scientific config
remain unchanged; complete inflight rollout and failure evidence are retained.
At the final 5000-group checkpoints, fixed monitor512 accuracy is **67.58% for
Vanilla** and **68.55% for Dynamic**: a **0.98 percentage-point gain (5 questions)**.
Dynamic consumed **17,443,344** rollout prompt-plus-output tokens versus Vanilla's
**5,964,478** (**2.925×**). These counters exclude validation and optimizer compute;
the ratio is not a GPU-hour or monetary-cost ratio. This single-seed validation
result does not establish a compute-efficiency advantage or a held-out test gain.
Final-budget checkpoints are reported without selecting an earlier validation peak.
Readiness for Stage 5 remains **NO**.
Stage 5–6 remain **NOT_STARTED**.
Stage5 evaluation datasets are **prepared and frozen**: CMExam official6811
(6809 scorable), CMB2000, clinical74 cases/208 questions, retention200, and a
111-question source-risk slice. The DeepSeek Flash external reference probe
completed all1100 requests with search/tools disabled: CMExam458/512 (89.45%),
CMB250/280 (89.29%), and308 open-ended responses retained for independent judging.
Calculated probe cost was USD1.8637;11 capped responses had empty visible answers
and remain in the denominator. See the
[external API report](docs/implementation/STAGE5_EXTERNAL_DEEPSEEK_FLASH.md),
[dataset preparation report](docs/implementation/STAGE5_DATASET_PREPARATION_REPORT.md)
and [frozen manifest](experiments/stage5/dataset_freeze_v1.json).
The subsequent anonymous V4 Pro open-QA judge was stopped at the user's budget
request after69 clinical answers /26 cases. [Partial scores](docs/implementation/STAGE5_OPEN_QA_JUDGE_PARTIAL_RESULTS.md)
are retained; the retention-set judge and real human audit remain incomplete.
No project-model final-test scoring has run. Current stage status is recorded in
[`project_state.json`](project_state.json).

- [Stage 0 report](docs/stage_reports/00_runtime_compatibility.md)
- [Stage 1 report and interview evidence](docs/stage_reports/01_medical_sft.md)
- [Stage 1 verification receipt](experiments/stage1/verification-final.json)
- [Stage 2 report and interview evidence](docs/stage_reports/02_reward_rollout.md)
- [Stage 2 final verification](experiments/stage2/verification-final.json)
- [Stage 3 inherited readiness and frozen inputs](experiments/stage2/readiness.json)
- [Stage 3 report and interview evidence](docs/stage_reports/03_dynamic_sampling.md)
- [Stage 3 verification receipt](experiments/stage3/verification-final.json)
- [Stage 4 sampler readiness](experiments/stage3/readiness.json)
- [Stage 4 current progress and commands](docs/implementation/STAGE4_PROGRESS.md)
- [Stage 4 measured decisions and limitations](docs/implementation/STAGE4_DECISIONS.md)
- [Stage 4 frozen formal pair](experiments/stage4/formal_pair.json)
- [Stage 4 real transaction fault verification](experiments/stage4/recovery_fault_injections.json)
- [Stage 4 completed pilot analysis](docs/implementation/STAGE4_PILOT_REVIEW.md)
- [Stage 3 execution plan and lifecycle decisions](docs/implementation/STAGE3_PLAN.md)
- Current refill progress: `python scripts/stage3_status.py`
- [Fixed SFT initialization and hashes](experiments/stage1/initialization_manifest.json)
- [Runtime installation and CLI](env/README.md)
- [Selected evidence](experiments/stage0/selected_runs.json)
- [Failed and successful run inventory](experiments/stage0/run_inventory.json)
- [Measured and conditional compute budget](docs/implementation/COMPUTE_BUDGET.md)

The Stage 0 validator applies to its historical stage-isolation checkpoint;
its archived PASS receipt is retained. Stage 1's archived PASS also retains its
historical isolation boundary. Stage 2 was reverified at Stage 3 entry; its isolation-boundary PASS is retained in
`experiments/stage3/prerequisite-stage2.json`. Stage 3 verification consumes those frozen inputs and independently checks new raw refill evidence. Bulk models/checkpoints/raw responses are outside Git;
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
Stage 2 is **DONE**, including full profiling, scoring, analysis and verification; selected runs are in
[`experiments/stage2/selected_runs.json`](experiments/stage2/selected_runs.json).
Stage 3 is **DONE**, Stage 4 is **FULL_PASS** with acceptance deliverables pending, and Stage 5–6 remain **NOT_STARTED**.
Historical planning/Stage 0/Stage 1 validators intentionally check their original stage-isolation boundaries; their archived receipts are preserved.

Use the independent training environment and real Stage 1 entry points:

```bash
.venv-train/bin/python scripts/verify_stage1_data.py --run experiments/stage1/s1_data_20260908T144854_bac3a7 --output /tmp/new-data-verification.json
.venv-train/bin/python scripts/run_stage1.py prepare --config configs/stages/s1_smoke.json
.venv-train/bin/python scripts/run_stage1.py prepare --config configs/stages/s1_pilot.json
.venv-train/bin/python scripts/run_stage1.py inspect --run <absolute-bulk-run-directory>
.venv-train/bin/python scripts/run_stage1.py resume --run <absolute-bulk-run-directory> --checkpoint <verified-checkpoint-directory>
.venv-train/bin/python scripts/verify_stage.py --stage 2 --output /tmp/stage2-recheck.json
```

Each prepare creates a new run and detached worker; resume continues the same run with a new attempt and verified optimizer/scheduler/RNG/sample cursor. Formal preparation additionally requires committed configuration and successful smoke/pilot receipts. Do not relaunch smoke/pilot to resume an existing formal run. The original `mpt sft --mode smoke` remains a **Stage 0 synthetic capacity probe**, not the real medical SFT entry point above.

The frozen 20k corpus, token cache, full metrics, generation outputs and checkpoints live under `/data/WSH/medical-post-train-artifacts/`; Git keeps compact evidence manifests and reports. See [Stage 1 execution design](docs/implementation/STAGE1_EXECUTION.md) and [decisions](docs/implementation/STAGE1_DECISIONS.md).
