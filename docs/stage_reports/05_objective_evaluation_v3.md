# Stage5 objective evaluation — v3 verified measurement

As of 2026-09-18: OBJECTIVE_EVAL_PASS (v3); OPEN_QA_JUDGE_PENDING. Stage5 remains FULL_RUNNING and Stage6 NOT_STARTED. This is the active exam report; v1/v2 reports are retained historical measurements.

All 35,236 unique final exam generations (four checkpoints × 8,809 questions) were rescored from original responses. No valid output was regenerated. The v3 verifier checked raw/response receipt hashes and all predictions, and independently reproduced summaries, secondary subsets, slices and paired statistics. Historical full token replay was linked through unchanged evidence.

## Primary models selected on validation only

| Model | Checkpoint groups | CMExam 6809 | CMB 2000 | CMExam clean 6732 | CMB medical 1929 |
|---|---:|---:|---:|---:|---:|
| sft | fixed SFT | 4186/6809 (61.48%) | 1116/2000 (55.80%) | 4129/6732 (61.33%) | 1075/1929 (55.73%) |
| selected_vanilla | 4608 | 4201/6809 (61.70%) | 1150/2000 (57.50%) | 4142/6732 (61.53%) | 1111/1929 (57.59%) |
| selected_dynamic | 5000 | 4224/6809 (62.04%) | 1153/2000 (57.65%) | 4167/6732 (61.90%) | 1112/1929 (57.65%) |

## Equal-update comparison

| Endpoint | CMExam | CMB |
|---|---:|---:|
| vanilla 5000 groups | 62.06% | 57.30% |
| dynamic 5000 groups | 62.04% | 57.65% |

Both formal runs used 5,000 groups, 625 policy windows and 1,250 optimizer steps. Checkpoint selection remains frozen at Vanilla 4608 and Dynamic 5000; final tests did not select a new checkpoint.

## Paired uncertainty

| Dataset / comparison | Delta (pp) | 95% paired bootstrap CI | Exact McNemar p |
|---|---:|---|---:|
| CMExam / primary: selected_vanilla_minus_sft | +0.220 | [-0.852, 1.278] | 0.7043 |
| CMExam / primary: selected_dynamic_minus_sft | +0.558 | [-0.485, 1.586] | 0.3110 |
| CMExam / primary: selected_dynamic_minus_selected_vanilla | +0.338 | [-0.705, 1.381] | 0.5441 |
| CMExam / scientific_endpoints: dynamic_minus_vanilla | -0.029 | [-1.072, 1.028] | 0.9783 |
| CMB / primary: selected_vanilla_minus_sft | +1.700 | [-0.400, 3.700] | 0.1156 |
| CMB / primary: selected_dynamic_minus_sft | +1.850 | [-0.200, 3.850] | 0.0799 |
| CMB / primary: selected_dynamic_minus_selected_vanilla | +0.150 | [-1.850, 2.150] | 0.9210 |
| CMB / scientific_endpoints: dynamic_minus_vanilla | +0.350 | [-1.650, 2.350] | 0.7689 |

Bootstrap: 10,000 paired prompt resamples, seed 20260914. Primary pairwise intervals cross zero. These data do not establish a Dynamic advantage, equivalence, or consistent RL improvement over SFT. Multiple comparisons/slices are exploratory and unadjusted; one training seed limits generalization.

## Fixed CMExam difficulty slices

| Model | Easy (4525) | Medium (1500) | Hard (784) |
|---|---:|---:|---:|
| sft | 69.79% | 49.40% | 36.61% |
| selected_vanilla | 69.61% | 51.80% | 34.95% |
| selected_dynamic | 69.92% | 51.33% | 36.99% |

Dynamic has a higher hard-slice point estimate than selected Vanilla, but a lower medium-slice point estimate. Do not infer a medium-difficulty mechanism or a subgroup win from point estimates. Complete fixed category membership, counts and interval results remain in the linked JSON tables.

## Parser correction and costs

Final exam prompts asked for “最终答案：A”, whereas selection prompts required letter-only answer tags. The original parser rejected many clear answers. V2 fixed marker-in-tag handling; v3 additionally handles repeated equal conclusions, option descriptions and ordinary option mentions. V3 was specified after observing test outputs, with tests and code pinned before its aggregate scores were computed. This is a disclosed measurement correction, not a pre-test frozen parser.

| Model | CMExam unparseable | CMB unparseable |
|---|---:|---:|
| sft | 4.45% | 4.25% |
| selected_vanilla | 5.74% | 7.45% |
| selected_dynamic | 6.43% | 7.45% |

Formal training generated 5,964,478 prompt-plus-output tokens for Vanilla and 17,443,344 for Dynamic (2.92454×). This ratio excludes validation and optimizer compute and is not a GPU-hour/currency ratio. The held-out results do not establish a rollout-compute efficiency advantage.

## Open QA, safety and remaining gates

The 74 clinical cases/208 questions and 200 retention prompts have 1,224 complete local responses. The frozen blind schedule has 1,224 pairs plus 123 position flips (1,347 judging entries). No independent judgments or human reviews are complete. There are no preference win-rates or rubric quality scores yet.

Safety is machine triage only: 111 source-defined items, 333 reused responses, four flagged responses across three items. These are review candidates, not confirmed unsafe answers; unflagged outputs are not certified safe. The human packet currently covers 84 items / 252 pairs: 82 frozen items plus additional flagged items. All future judge critical flags must also be reviewed.

Remaining Stage5 gates: independent scoring, position-consistency analysis, required real human audit, safety review/adjudication, and final integrated narrative. Stage4 owner waiver covered Stage4 manual documentation only; it does not waive Stage5 review. No paid API calls were made for this closure.

## Evidence

- [V3 verification](../../experiments/stage5/closure_v3_20260918/objective_verification_v3.json)
- [CMExam tables](../../experiments/stage5/parser_correction_v3/cmexam_final_results_v3.json) and [CMB tables](../../experiments/stage5/parser_correction_v3/cmb_final_results_v3.json)
- [V3 case index](../../experiments/stage5/closure_v3_20260918/case_candidates_v3.json): all gain/regression buckets retained, including unfavorable cases.
- [Blind-review package manifest](../../experiments/stage5/closure_v3_20260918/review_packet.json)
- [Historical v1 report](05_objective_evaluation.md) and [initial v3 snapshot](05_exam_rescoring_v3.md); do not use either as the current complete report.
- [Cleanup round 1](../../experiments/maintenance/disk_cleanup_20260918/summary.json) and [round 2](../../experiments/maintenance/disk_cleanup_20260918_round2/summary.json). Intermediate native training states were pruned after owner authorization. All adapters and 77 protected full checkpoints remain. Historical Stage4 full-file verification cannot be repeated unchanged for pruned nodes; original verification receipts and removal manifests are retained.
