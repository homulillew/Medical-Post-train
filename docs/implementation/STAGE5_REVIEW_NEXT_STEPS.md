# Stage5 remaining review steps

Objective exams are verified under v3. The next step is independent assessment
of the existing open-QA answers. Do not regenerate candidate responses, add training,
or interpret the empty templates as judgments.

1. Assign an independent judge (model ID/version and local endpoint, or an actual
   human evaluator). Candidates cannot judge themselves. The owner has authorized
   the full 1347-entry Atria evaluation after the 20-entry pilot and bounded repair
   diagnostic. See [the full-run decision](../decisions/STAGE5_ATRIA_FULL_EVALUATION.md).
   Preserve the exact judge instructions, model configuration
   and every raw response. Use the existing 1347-entry packet with its frozen order.
2. Fill the five A/B rubric scores and preference, rationale and safety flags.
   In addition to the blank template fields, each record needs
   `judge_is_candidate: false` and `raw_judgment: {path, sha256, bytes}` referencing
   the retained raw judge artifact. These declarations require later provenance
   verification and do not themselves establish independence.
3. A real person reviews the existing 84-item/252-pair queue and all additional
   judge-critical items. Record reviewer identity, qualification, an observation
   and `reviewed_in_full: true` only after reading. `disagreement` is an explicit
   boolean; disputes need both `resolution` and `adjudicator_id`. Non-clinicians
   must include `disclaimer: NON_CLINICIAN_REVIEW_NOT_CLINICAL_VALIDATION`.
4. Validate a new submission (files must be separate from the immutable blanks):

```bash
PYTHONPATH=src:scripts python scripts/validate_stage5_review_submission.py \
  --judgments /absolute/path/judgments.jsonl \
  --human-reviews /absolute/path/human_reviews.jsonl \
  --output /absolute/path/new_validation_receipt.json
```

The command never invokes an API or model. Missing/invalid data returns exit 2 and
an immutable receipt; schema/coverage success returns 0. Success is not Stage5
acceptance: judge provenance, attributed safety review of all 333 responses,
position consistency, clinical case-cluster statistics and final acceptance are
still required. Keep the current Stage5 state pending until these are complete.

The final gate precedes Stage6's 100-prompt offline/serving consistency check and
concurrency 1/4/8/16 benchmarks. No safety or clinical conclusion is inferred from
the four machine-triage flags or from unflagged outputs.
