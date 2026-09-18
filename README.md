# Medical-Post-train

A reproducible medical LLM post-training study on one 48 GB GPU:
Qwen3-8B medical SFT → Vanilla / Dynamic Sampling GSPO → held-out evaluation → vLLM serving.

## Current status — 2026-09-18

Stages 0–4 are complete. Stage4 full verification passed with an explicitly recorded
owner waiver for its manual documentation review; no clinician review is implied.
Both formal GSPO runs reached 5,000 groups, 625 windows and 1,250 optimizer steps.

Stage5 objective evaluation is verified under the **v3 answer parser**. All 35,236
final exam responses and 1,224 open-QA responses have been generated. Independent
open-QA scoring is partial (16 valid judgments from a 20-entry Atria API pilot),
and real human audit is pending, so **Stage5 is not DONE**.
Stage6 serving benchmarks have not started.

| Validation-selected model | CMExam (6809) | CMB (2000) |
|---|---:|---:|
| SFT | 61.48% | 55.80% |
| Vanilla GSPO (4608 groups) | 61.70% | 57.50% |
| Dynamic GSPO (5000 groups) | 62.04% | 57.65% |

All main paired confidence intervals cross zero. At equal 5000-group endpoints,
CMExam is 62.06% Vanilla vs 62.04% Dynamic; CMB is 57.30% vs 57.65%.
Dynamic used 2.92454× the training rollout prompt-plus-output tokens. The results
do not establish Dynamic superiority or a compute-efficiency advantage.

V3 corrects false answer-parser rejections after test outputs were observed.
Earlier scores and raw responses are retained; no checkpoint was reselected and
no valid response was regenerated. Treat v1/v2 score reports as historical.

## Active results and review

- [Current Stage5 report](docs/stage_reports/05_objective_evaluation_v3.md)
- [Current machine-readable results index](experiments/stage5/current_results_v3.json)
- [V3 objective verification](experiments/stage5/closure_v3_20260918/objective_verification_v3.json)
- [Results handoff](experiments/handoffs/stage5_results_to_chatgpt_v3.json)
- [Interview narrative](docs/stage_reports/05_interview_story_v3.md)
- [Open-QA review package manifest](experiments/stage5/closure_v3_20260918/review_packet.json)
- [Atria API pilot: results and failures](docs/stage_reports/05_atria_api_pilot_20260918.md)
- [Project state and remaining gates](project_state.json)

The anonymous review package contains 1347 judge entries including position flips.
Its current human-audit queue covers 84 items / 252 comparisons. Scores and reviewer
identities in the immutable templates remain blank. Actual API judgments are
stored separately: the owner-authorized pilot tested 20 entries, with 16 valid
scores and 4 unscored entries. API requests have stopped. Its only fully observed
position-flip pair disagreed; this small pilot cannot establish model rankings.
Human reviews remain zero. Safety flags are review candidates, not clinical judgments.

## Reproduction and storage

- [Project charter](docs/PROJECT_CHARTER.md), [completion criteria](docs/DEFINITION_OF_DONE.md)
- [Experiment protocol](docs/EXPERIMENT_PROTOCOL.md), [evaluation design](docs/implementation/STAGE5_EVALUATION_DESIGN.md)
- [Environment](env/README.md), [Stage4 report](docs/stage_reports/04_gspo.md)
- [Parser v3 decision](docs/decisions/STAGE5_PARSER_CORRECTION_V3.md)
- [Cleanup round 1](experiments/maintenance/disk_cleanup_20260918/summary.json), [round 2](experiments/maintenance/disk_cleanup_20260918_round2/summary.json)

Bulk weights and raw outputs live outside Git under
`/data/WSH/medical-post-train-artifacts`; a clone alone does not restore them.
Owner-authorized disk cleanup retained all adapters and 77 full training checkpoints,
while pruning obsolete intermediate optimizer/native states. Original manifests,
metrics and verification receipts remain, but historical full-file training checks
cannot be rerun unchanged on pruned nodes.

The [previous README](docs/history/README_before_stage5_v3_20260918.md) is retained as
historical context; its old stage-status statements are superseded by the above.
