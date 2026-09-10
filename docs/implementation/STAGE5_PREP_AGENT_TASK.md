# Stage 5 Evaluation Preparation + External SOTA Probe — Agent Task

This task may be executed while Stage 4 is `FULL_RUNNING`. It is **preparation only** and must not alter the frozen Stage 4 scientific run or mark Stage 5 started/completed.

Read first:

1. `AGENTS.md`
2. `docs/stages/05_evaluation.md`
3. `docs/implementation/STAGE5_EVALUATION_DESIGN.md`
4. Stage 1 data/decontamination artifacts
5. Stage 2 CMExam candidate/heldout manifests
6. current `project_state.json`

## Objective

Build and freeze the multi-dimensional Stage 5 evaluation manifests from the actual datasets already available to this repository, then optionally run a low-cost strong third-party API baseline (for example DeepSeek) on the frozen SOTA probe.

Do not assume dataset schemas from documentation alone. Inspect the pinned/local dataset versions and record the actual fields, counts, IDs, and source revisions used.

## Hard boundaries

Do not:

- change Stage 4 code/config/reward/sampling/optimizer/budgets;
- evaluate any SFT/Vanilla/Dynamic checkpoint on CMExam/CMB final test data while Stage 4 is still running;
- select checkpoints from test data;
- redesign subsets after seeing project-model test outputs;
- resample wrong API answers until correct;
- expose or commit API secrets.

External API results may be produced now only as contextual reference/SOTA probes.

## Part 1 — Dataset inventory

Inspect the actual locally pinned sources and produce a machine-readable inventory covering at least:

- CMExam test;
- CMB-Exam test;
- CMB clinical/open-ended component if available;
- unused/held-out Medical-o1 source records;
- unused/held-out Huatuo source records;
- Stage 1 SFT train IDs/hashes;
- Stage 2/4 CMExam RL pool IDs/hashes;
- Stage 4 monitor/selection IDs/hashes.

Record source revision/path/hash, raw count, usable count, and relevant schema fields.

## Part 2 — Reuse decontamination machinery

Reuse the repository's existing normalization and exact/near-duplicate logic/thresholds wherever possible.

For every evaluation candidate, check contamination against the relevant training/validation pools and emit reason-coded exclusions. Preserve excluded IDs and summary counts.

Do not weaken or change near-duplicate thresholds just to hit a desired sample count. If a requested count cannot be reached credibly, report the shortfall and reason.

## Part 3 — Freeze evaluation manifests

Construct deterministic manifests using a stable named seed such as `medical-posttrain-stage5-eval-v1` plus domain-specific derivation.

Required targets:

### CMExam

- `cmexam_test_full`: all valid held-out CMExam test questions, expected target 6,811 if the pinned source matches.
- `cmexam_sota_probe_512`: deterministic 512-item subset stratified across available difficulty/category metadata as evenly as practical.

### CMB-Exam

- `cmb_exam_clean_2000`: deterministic decontaminated 2,000-item external test subset.
- if the pinned schema confirms 28 subcategories, balance approximately equally across categories.
- `cmb_sota_probe_280`: 10 per confirmed category, preferably a subset of the frozen 2,000.

### External clinical/open-ended

- `cmb_clin`: retain all valid CMB clinical cases when available after decontamination.
- if the pinned source lacks the expected clinical component, do not silently substitute; create a decision record proposing the best independent alternative.

### In-domain medical QA/dialogue retention

- `open_qa_retention_200`: 200 clean unused Stage 1 source items, preferably approximately 100 Medical-o1-style and 100 Huatuo-style when feasible.
- `open_qa_sota_probe_100`: deterministic subset of the 200.

This set measures behavior retention, not external generalization.

### Safety slice

- tag a credible high-risk slice from the open-ended sets before seeing candidate responses;
- target at least 50 if naturally supported;
- keep source-derived tags and rationale;
- if insufficient, report limitation rather than fabricate many synthetic cases.

For each manifest record, retain at least stable source ID, source split, source hash/reference, selection stratum, selection seed/hash, prompt/reference fields needed for evaluation, and decontamination disposition.

## Part 4 — Freeze evaluation prompts/parsers

Implement/freeze a provider-neutral exam prompt that asks for one unambiguous final-answer marker while allowing a model to reason internally.

The scoring parser must:

- support the project's `<answer>...</answer>` output;
- support the external final-answer marker;
- canonicalize single/multi-answer formats according to the existing answer contract;
- be tested on adversarial formatting examples;
- be identical across project models and external API models.

Do not require proprietary reasoning APIs to reveal chain-of-thought. Score canonical final answers only.

For open-ended QA, score only the visible user-facing answer. Hidden reasoning is diagnostic-only and must not advantage one provider.

## Part 5 — Freeze open-ended rubric

Create a versioned rubric covering:

1. medical factual correctness;
2. relevance/completeness;
3. safety/escalation appropriateness;
4. uncertainty calibration;
5. clarity/usefulness.

Prepare blinded pairwise evaluation infrastructure for later SFT vs Vanilla vs Dynamic comparison.

If an LLM judge is used later, candidates may not self-judge. Randomize A/B order, retain raw judge outputs, and plan at least 20% manual audit plus all critical safety failures.

## Part 6 — External API baseline adapter

Implement a provider-agnostic external baseline runner if one does not exist. DeepSeek may be the first provider, but do not hard-code project logic to a single vendor.

API credentials must come only from environment/secret configuration and must never be printed or committed.

Each external run must record:

- run ID;
- provider;
- exact model identifier exposed by the API;
- date/time;
- endpoint/mode;
- request parameters actually accepted;
- prompt/protocol version/hash;
- whether tools/web/search/retrieval are disabled;
- raw response;
- parsed answer where relevant;
- input/output token usage if exposed;
- latency;
- retry/error status;
- monetary cost if calculable from provider metadata/pricing pinned for the run.

Retry transport/rate-limit failures only. Do not use repeated stochastic attempts to improve benchmark correctness.

## Part 7 — Cheap SOTA probe

Before making paid calls, produce a dry-run report with exact request counts and an estimated token/cost envelope.

Recommended first external probe bundle:

- 512 CMExam exam questions;
- 280 CMB-Exam questions when category schema permits;
- all valid frozen CMB-Clin cases;
- 100 held-out medical-QA retention prompts.

If API budget is limited, prioritize in this order:

1. CMExam 512;
2. CMB-Exam 280;
3. CMB-Clin;
4. open-QA retention 100.

Do not silently shrink a frozen probe after seeing partial scores. A smaller budget tier must be declared and frozen before calls.

If a DeepSeek API credential is available in the execution environment, the agent may run the frozen probe. If no credential is available, stop after producing the manifests, runner, dry-run command, and cost estimate. Do not ask the project owner to paste secrets into chat or commit them.

## Part 8 — External baseline report

For exam probes report:

- exact/canonical accuracy;
- parse success rate;
- category/difficulty slices where sample counts support them;
- total requests/tokens/cost;
- invalid/failed API requests separately from wrong answers.

For open-ended probes, retain raw responses now. Absolute rubric scoring may be run only with an independent frozen judge/human protocol; do not use the same candidate model to certify itself.

Label all such results clearly as:

- `external SOTA/reference probe`;
- preliminary/contextual;
- not a Stage 5 project-checkpoint result;
- not an online A/B test.

## Required outputs

Adapt paths to the existing repository conventions, but produce at minimum:

- evaluation source inventory;
- deterministic split builder code;
- all frozen manifest files;
- decontamination/exclusion report;
- prompt/parser protocol + tests;
- open-ended rubric protocol;
- external API runner + secret-handling documentation;
- dry-run/cost report;
- external baseline raw artifacts and compact summary if API execution occurred;
- a decision/observation record for any schema or source mismatch.

Large raw API responses may live in bulk artifact storage; Git must retain compact manifests, hashes, configs, summaries, and representative cases according to the existing artifact policy.

## Completion of this preparation task

This task is complete when the evaluation sets and protocols are reproducibly frozen and an external SOTA probe is either:

1. successfully executed with auditable artifacts, or
2. ready to execute with a tested runner and dry-run/cost report, with the only blocker being absent external API credentials/budget.

Do **not** change `project_state.json` Stage 5 from `NOT_STARTED` merely because this preparation task is complete.