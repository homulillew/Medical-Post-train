# Stage5 exam rescoring v3 — CMExam complete, CMB pending

2026-09-16. This report supersedes v1/v2 CMExam accuracy interpretation after the owner-authorized parser correction. Generation, checkpoint selection and training did not change. Stage5 remains FULL_RUNNING; open-QA judgments remain pending.

## Completed CMExam results

All four checkpoints have 6,809 retained responses. V3 extracts visible explicit answers, permits identical repeated conclusions and option descriptions, and rejects conflicting or missing answers. The correction was specified after observing test outputs; it is not a pre-test frozen protocol.

| Checkpoint | Correct / 6809 | Accuracy | Unparseable |
|---|---:|---:|---:|
| sft | 4186 | 61.48% | 4.45% |
| vanilla_4608 | 4201 | 61.70% | 5.74% |
| dynamic_5000 | 4224 | 62.04% | 6.43% |
| vanilla_5000 | 4226 | 62.06% | 5.40% |

Selected Dynamic minus selected Vanilla: +0.338 percentage points, paired bootstrap 95% CI [-0.705, 1.381], exact McNemar p=0.544. Equal-update Dynamic 5000 minus Vanilla 5000: -0.029 percentage points, CI [-1.072, 1.028], p=0.978. All primary pairwise intervals cross zero; no clear advantage is established. Bootstrap uses 10,000 resamples, seed 20260914.

The previous v2 Dynamic deficit is not robust to correcting the parser. This supports an evaluation-format explanation for much of that measured deficit; it does not establish equivalence or a Dynamic benefit. Residual unparseable responses remain counted incorrect, so these are end-to-end scores under the explicit v3 extraction rules.

V3 also rejects two previously v2-parsed outputs (one in each Vanilla checkpoint): a textual conclusion without an explicit letter precedes the final letter. These conservative failures are retained and recorded; no gold-label matching resolves them.

## Evidence and remaining work

- [Rule amendment](../decisions/STAGE5_PARSER_CORRECTION_V3.md).
- [CMExam tables, paired gains/regressions, fixed difficulty/category slices and secondary subsets](../../experiments/stage5/parser_correction_v3/cmexam_final_results_v3.json).
- [Pinned code and tests](../../experiments/stage5/parser_correction_v3/source_manifest.json): 115 tests passed.
- Per-item v3 predictions and raw/receipt provenance: `/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1/rescored_v3`.
- The CPU watcher automatically rescores future completed CMB jobs and emits `cmb_final_results_v3.json`. SFT CMB currently completed: 1116/2000 (55.80%); a full model comparison is pending.
- Full token/attempt lineage verification remains separate in the frozen objective pipeline. V3 replay checks raw/receipt hashes, adapter identity and v1 score reconstruction. Neither this report nor watcher completion declares full Stage5 verified.
- Final project reporting/handoff must use the v3 exam artifacts and retain this post-observation correction disclosure. Original pipeline-generated v1 scores/case mining remain archived legacy outputs.
