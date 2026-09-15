# Stage5 visible-answer rescoring v3

- Date: 2026-09-16
- Status: ACCEPTED
- Owner authorization: “重新评估”, following analysis of remaining v2 parse failures.
- Contract change: YES; explicit post-observation final-evaluation parsing correction.
- Run ID: `stage5_parser_correction_v3_20260916`, class EVALUATION.

V2 fixed a final marker inside an answer block but retained overly strict checks.
Equal repeated conclusions, letter-plus-option-description answers, ordinary
option mentions and numeric inequalities could still be rejected. This made the
v2 end-to-end score unsuitable as an unqualified measure of medical knowledge.

Before reading v3 aggregate accuracy, pin code and tests implementing these rules:

1. Only the visible final region is eligible. Reject malformed, nested or multiple
   answer/thinking blocks. Numeric comparison operators are content, not tags.
2. Extract option letters immediately following explicit answer or positive choice
   markers. Permit the option description after the letter; never infer a letter
   by matching medical meaning or the gold answer.
3. Canonicalize comma-separated, contiguous and conjunction-separated multi-select
   letters. Check against each item's legal options; reject duplicate/illegal letters.
4. Repeated explicit conclusions must agree. A heading with no answer immediately
   followed by another final marker contributes no candidate. Different positive
   conclusions, explicit alternatives and uncertain/retracted answers fail.
5. Ordinary mentions of options are not competing conclusions. Negative choices
   are not positive assertions. A bare option at the start is accepted only within
   the constrained standalone/letter-plus-description syntax in the tests.
6. Recoveries count as fallback accuracy, not strict-format compliance. Missing
   answers and truncated thinking remain failures. Do not increase generation caps.

The parser has no gold-label or checkpoint-identity input. Rescore every primary
and equal-update exam checkpoint with identical rules. Do not reselect checkpoints,
change prompts, regenerate answers or edit v1/v2 evidence. Test identities, labels,
decoding and budgets remain frozen. Pin the new code/tests before computing scores.

Outputs are additive under `stage5_project_v1/rescored_v3` and
`experiments/stage5/parser_correction_v3`. The v3 CPU watcher consumes only completed
jobs, verifies source/receipt/raw hashes and v1 score replay, emits predictions,
v1/v2/v3 summaries, parse reasons and paired tables using existing frozen slices
and 10,000 resamples with seed 20260914. Full token/attempt replay remains separately
required from the original generator's objective verifier. V3 rescoring completion
does not mean full Stage5 completion or a clinical/open-QA judgment.

The original pipeline still produces archived v1 reports; v2 also remains running
for historical comparison. Final reporting must use v3 for the revised exam
measurement and disclose v1/v2 differences and this post-observation correction.
V3 case evidence includes per-item predictions and gained/regressed IDs in paired
tables. A legacy v1 PASS is not a v3 scientific sign-off.

Resource impact: CPU-only replay/statistics; zero new GPU generations or paid API
calls. Comparability is within the fixed v3 protocol. Do not choose between parser
versions based on which model wins, or describe v3 as a pre-test frozen protocol.
