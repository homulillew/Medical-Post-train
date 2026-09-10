# Stage 5 Evaluation Design

This document refines the Stage 5 contract into a product-relevant, multi-dimensional evaluation plan for the medical reasoning assistant. It also defines a low-cost third-party API baseline protocol that may be prepared and run while Stage 4 is still `FULL_RUNNING`, provided it does not influence Stage 4 training, checkpoint selection, or scientific configuration.

## Evaluation goals

The project is not intended to be only a medical-exam solver. The final evaluation must separately answer four questions:

1. **Medical knowledge/reasoning correctness:** did post-training improve verifiable medical answers?
2. **Cross-dataset generalization:** did gains transfer beyond the CMExam RL distribution?
3. **Open-ended medical QA/dialogue retention:** did exam-style RL preserve useful medical-assistant behavior rather than collapse into choice-only answering?
4. **Training/system efficiency:** what accuracy was obtained per policy update and per generated rollout token, and what serving performance is achievable?

Do not collapse these dimensions into one synthetic score. Report them separately.

## Primary model states

The final internal comparison remains:

1. verified Medical SFT checkpoint;
2. SFT + Vanilla GSPO checkpoint selected using validation data only;
3. SFT + Dynamic Sampling + GSPO checkpoint selected using validation data only.

Third-party API models are contextual reference baselines, not causal controls for the Vanilla-vs-Dynamic experiment.

## Evaluation matrix

| Track | Dataset / slice | Purpose | Headline metric |
|---|---|---|---|
| A. Verifiable in-domain | CMExam test | Medical knowledge/reasoning after CMExam RL | exact/canonical answer accuracy |
| B. External exam | CMB-Exam clean fixed subset; optional full test | Cross-dataset medical knowledge/reasoning generalization | exact/canonical answer accuracy |
| C. External clinical/open QA | CMB-Clin or locally available equivalent | Open-ended clinical reasoning / response quality | blinded pairwise + fixed rubric |
| D. In-domain dialogue retention | decontaminated held-out Medical-o1/Huatuo QA | Detect loss of medical QA style/behavior after exam RL | blinded pairwise + fixed rubric |
| E. Safety/behavior slice | tagged high-risk subset of C/D | Detect obvious unsafe or overconfident regressions | critical-safety-flag rate + rubric |
| F. Efficiency | retained Stage 4 metrics | Update efficiency vs rollout cost | accuracy vs updates; accuracy vs generated tokens |
| G. Serving | Stage 6 | Product-facing runtime feasibility | P95 latency, TTFT/TPOT, throughput, error rate |

Tracks A and B remain the primary quantitative capability benchmarks. Tracks C-E are formal secondary product-relevance evaluations, not an informal afterthought.

## Deterministic set construction

All evaluation manifests must be built **before looking at outputs from the project checkpoints on those sets**. The construction code, seed, source IDs, hashes, exclusions, and final manifests must be retained.

Reuse the repository's existing normalization, exact-duplicate, and near-duplicate/decontamination implementation and thresholds where possible. Do not invent a new near-duplicate threshold solely to make a set look cleaner. If a new rule is necessary, create a decision record before evaluating model outputs.

Exclusion sources must include, where applicable:

- the 20k Stage 1 SFT training set;
- Stage 2/4 CMExam RL candidate/training pools;
- Stage 4 monitor/selection validation subsets;
- exact/near duplicates across the final evaluation tracks themselves;
- malformed or answer-ambiguous exam items according to the existing parser contract.

Retain exclusion counts and reason codes.

### A. CMExam final test

Use the full held-out CMExam test split (6,811 questions) for final model evaluation.

Create a deterministic `cmexam_sota_probe` subset for cheap external-API reconnaissance before Stage 5. Recommended size: **512 questions**, stratified across available difficulty/category metadata as evenly as practical while preserving deterministic sampling.

The probe is a subset of the frozen final test manifest and is for contextual SOTA/reference measurement only. Seeing third-party model results on it must not change Stage 4 training or the final evaluation protocol.

### B. CMB-Exam final clean subset

Build a fixed, decontaminated **2,000-question** subset from the appropriate CMB-Exam test source. If the locally pinned dataset exposes the expected 28 subcategories, sample approximately equally across them. A deterministic allocation such as 71 per category plus the 12 hash-selected remainder categories receiving one extra item is acceptable.

Also create `cmb_sota_probe_280`: **10 questions per subcategory** when the 28-category schema is confirmed locally. Prefer drawing this probe as a deterministic subset of the frozen 2,000-item final manifest.

If the local CMB schema/version differs, inspect it and record the actual stratification rule rather than forcing assumed field names.

Full CMB-Exam evaluation remains optional if compute/API budget permits, but the fixed 2,000-item clean subset is mandatory for the project checkpoints.

### C. CMB-Clin / external open-ended clinical set

Inspect the locally available CMB clinical component and pin its exact version/schema. If the expected small clinical-case set is available, retain all valid cases after decontamination rather than subsampling purely for convenience.

This track evaluates visible medical responses, not hidden chain-of-thought. Preserve source references/answers/rubrics when available.

If CMB-Clin is unavailable in the pinned source, use an equivalent independent public open-ended Chinese medical QA/clinical set only after documenting the substitution in a decision record.

### D. Held-out medical QA/dialogue retention set

Construct a fixed **200-item** retention set from Stage 1 source material that was not used in the 20k SFT training set. Prefer a balanced source composition (for example, approximately 100 Medical-o1-style reasoning QA and 100 Huatuo-style direct medical QA) when enough clean items remain.

Requirements:

- not present in SFT train;
- not exact/near duplicate of SFT train;
- not exact/near duplicate of CMExam/CMB evaluation or RL pools;
- source proportions and sampling seed frozen before evaluation;
- the set is explicitly labeled **in-domain behavior retention**, not external generalization.

Create a deterministic **100-item** subset for the cheap third-party SOTA probe.

### E. Safety/behavior slice

Do not create a large synthetic safety benchmark merely to inflate scope. Instead, tag a fixed high-risk slice from C and D using source content only, before seeing candidate-model responses.

Useful tags include, when naturally present:

- emergency/red-flag escalation;
- medication/dose/contraindication questions;
- pregnancy/pediatrics/elderly high-risk contexts;
- insufficient-information/uncertainty cases;
- situations where overconfident diagnosis or treatment would be problematic.

Aim for at least 50 well-supported cases if the source pool permits. If the existing data does not support a credible safety slice, report the limitation rather than fabricating prompts. Any later addition of an external safety benchmark requires a decision record.

This is a heuristic product-safety audit, not clinical certification.

## Cheap third-party SOTA/reference probe

A third-party API baseline may be run **before Stage 4 finishes** because the Stage 4 scientific configuration is already frozen. This work is evaluation preparation only and must not change Stage 4 training decisions.

Recommended first probe bundle:

- CMExam: frozen 512-question SOTA probe;
- CMB-Exam: frozen 280-question SOTA probe when 28-category stratification is available;
- CMB-Clin: all valid frozen clinical cases;
- held-out medical QA/dialogue: frozen 100-item probe.

This gives a relatively low-cost multi-dimensional snapshot of a strong external model before paying for full external-baseline evaluation.

Example providers may include DeepSeek API or other strong contemporary models. At execution time, do not write only a marketing family name such as `DeepSeek`. Record:

- provider;
- exact API model identifier;
- model/version/revision information exposed by the provider;
- request date/time;
- endpoint/mode (chat/reasoning, if relevant);
- system/user prompt template hash;
- decoding parameters actually accepted by the API;
- max output tokens;
- tool/web/retrieval settings;
- request IDs where available;
- raw responses;
- usage tokens, latency, retries, status, and monetary cost when exposed.

External APIs must run without web search, browsing, retrieval, or tools when the provider allows these to be disabled. Otherwise document the limitation and do not present the result as a clean closed-book comparison.

Transport/rate-limit failures may be retried. A valid but wrong/unparseable answer must **not** be repeatedly resampled until correct.

## Matched inference protocol

For exam tasks, use one response per question and a common evaluation prompt that asks for an unambiguous final answer marker. Do not require proprietary reasoning models to expose chain-of-thought.

Score the canonical final answer only. The parser may support the project's `<answer>...</answer>` format plus the frozen external-evaluation final-answer marker/fallback, but parser behavior must be identical across candidate models and must be tested before scoring.

For open-ended tasks, score the **user-visible final response**. Hidden/internal reasoning is neither required nor scored. The project's `<think>` content may be retained for internal diagnostics but should not give the local model extra credit unavailable to API models.

Use deterministic/lowest-variance decoding when the provider supports it. If a reasoning endpoint ignores or disallows temperature/top-p controls, record the actual provider behavior rather than emulating unsupported parameters.

## Open-ended evaluation rubric

Do not use BLEU/ROUGE or embedding similarity as the headline open-ended metric.

Use a frozen rubric covering at least:

1. medical factual correctness;
2. relevance and completeness;
3. safety / escalation appropriateness;
4. uncertainty calibration (avoids unwarranted certainty when information is insufficient);
5. clarity and usefulness of the visible answer.

Recommended reporting:

- blinded pairwise win/tie/loss for Dynamic vs SFT and Dynamic vs Vanilla;
- per-dimension rubric score distribution;
- critical-safety-flag rate;
- representative wins, ties, regressions, and safety failures.

If an LLM judge is used, the candidate model must not judge itself. Prefer a judge/provider distinct from the compared candidate when practical. Pin the judge version, rubric, prompt, ordering/randomization, and raw judge outputs. Randomize A/B order and measure position consistency on a subset.

At least 20% of the open-ended items, and all critical safety failures, should receive manual audit. If the reviewer is not a clinician, label the audit as non-expert and do not claim clinical validation.

## Third-party baseline interpretation

Third-party API results answer: **what level does a strong external model achieve under this frozen protocol?** They do not prove equivalence of model scale, training data, compute, latency, or cost.

Do not call an external API comparison an online A/B test. Use terms such as `external reference baseline`, `SOTA probe`, or `blinded pairwise evaluation` as appropriate.

Do not compare vendor API latency directly to single-GPU local vLLM latency as if the serving environments were controlled. Keep capability and serving comparisons separate.

## Leakage and governance rules

While Stage 4 is `FULL_RUNNING`, GPT-6 may:

- inspect local evaluation sources;
- implement deterministic split/decontamination scripts;
- freeze evaluation manifests and hashes;
- implement parsers/rubrics and unit tests;
- run third-party API baselines on the frozen SOTA probe;
- record API raw outputs and costs.

While Stage 4 is `FULL_RUNNING`, GPT-6 must **not**:

- score SFT/Vanilla/Dynamic on CMExam test or CMB final test sets;
- use external-baseline results to change Stage 4 reward, sampling, optimizer, budget, or frozen config;
- select a Stage 4 checkpoint from test results;
- redefine the final test subsets after seeing project-model outputs.

Stage 5 status remains `NOT_STARTED` until its prerequisite contract is satisfied; preparation artifacts do not count as Stage 5 completion.

## Required preparation artifacts

GPT-6 should create machine-readable artifacts sufficient to reproduce every split, for example:

- `experiments/stage5/manifests/cmexam_test_full.json`;
- `experiments/stage5/manifests/cmexam_sota_probe_512.json`;
- `experiments/stage5/manifests/cmb_exam_clean_2000.json`;
- `experiments/stage5/manifests/cmb_sota_probe_280.json`;
- `experiments/stage5/manifests/cmb_clin.json`;
- `experiments/stage5/manifests/open_qa_retention_200.json`;
- `experiments/stage5/manifests/open_qa_sota_probe_100.json`;
- `experiments/stage5/manifests/safety_slice.json`;
- `experiments/stage5/decontamination_report.json`;
- `experiments/stage5/evaluation_protocol.json`;
- `experiments/stage5/external_baselines/<provider_model_run_id>/...`.

The exact path layout may be adapted to the repository architecture, but every item must retain stable source IDs, hashes, selection reason/stratum, and provenance.

## SOTA-probe acceptance gate

Before spending API budget, verify:

- split construction is deterministic;
- no project-model test output has been observed;
- decontamination/exclusion report exists;
- exam parser unit tests pass;
- prompt templates and API protocol are frozen;
- candidate API model identity will be recorded exactly;
- retry policy cannot convert wrong responses into repeated attempts;
- raw outputs and usage/cost metadata will be retained.

A SOTA probe may be reported while Stage 4 is running, but must be labeled as an external preliminary/reference result rather than a Stage 5 project result.

## Final Stage 5 headline structure

The final report should present separate headline blocks rather than one composite score:

- **Verifiable medical reasoning:** CMExam and CMB-Exam accuracy for SFT / Vanilla / Dynamic;
- **Open medical QA/dialogue:** blinded pairwise and rubric results on CMB-Clin + held-out retention set;
- **Safety/behavior:** critical flags and regression cases;
- **Efficiency:** accuracy vs updates and accuracy vs generated tokens;
- **Serving:** Stage 6 latency/throughput reported separately.

The final project claim should be scoped to the evidence: improving medical exam correctness does not by itself prove clinical-dialogue quality, and open-ended dialogue quality does not replace verifiable exam accuracy.