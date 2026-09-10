# Stage 5 — Controlled Evaluation and Ablation

## Why

Turn training runs into defensible evidence. The product target is a medical reasoning/QA assistant, not only an exam solver, so Stage 5 must separate:

1. verifiable medical knowledge/reasoning;
2. cross-dataset generalization;
3. open-ended medical QA/dialogue retention and safety;
4. update efficiency and rollout cost.

Do not collapse these dimensions into one synthetic score or report a single favorable metric as overall medical capability.

Detailed split-construction, third-party API baseline, open-ended rubric, and leakage rules are defined in [`docs/implementation/STAGE5_EVALUATION_DESIGN.md`](../implementation/STAGE5_EVALUATION_DESIGN.md). That document is part of this stage contract.

## Primary model states

Evaluate at minimum:

1. verified Medical SFT checkpoint;
2. primary Vanilla GSPO checkpoint;
3. primary Dynamic GSPO checkpoint.

Checkpoint selection must use validation data only. Test results may not select or tune checkpoints.

Third-party API models are contextual reference baselines, not causal controls for the Vanilla-vs-Dynamic experiment.

## Evaluation preparation while Stage 4 is running

Because the Stage 4 formal scientific configuration is frozen, GPT-6 may prepare Stage 5 evaluation assets before Stage 4 finishes, without changing Stage 5 status from `NOT_STARTED`.

Allowed preparation includes:

- inspect/pin local CMExam/CMB/open-QA source schemas;
- reuse repository decontamination logic to build deterministic evaluation manifests;
- freeze source IDs, seeds, hashes, strata, exclusions, prompt templates, parsers, rubrics, and retry rules;
- create a low-cost third-party SOTA/reference probe;
- run external API baselines such as DeepSeek on that frozen probe and retain raw outputs/cost metadata.

During Stage 4, do **not** evaluate SFT/Vanilla/Dynamic checkpoints on final test sets, change Stage 4 configuration because of API results, select checkpoints from test performance, or redefine evaluation subsets after seeing project-model outputs.

## Track A — CMExam verifiable in-domain evaluation

Use the held-out **CMExam test split (6,811 questions)** for the main in-distribution capability comparison.

All three primary model states must be evaluated under a matched inference and parsing protocol.

Primary metric: exact/canonical medical answer accuracy.

Also freeze a deterministic **512-question `cmexam_sota_probe`** stratified across available difficulty/category metadata as evenly as practical. This probe may be used now for cheap third-party API reconnaissance, but must not drive Stage 4 training decisions.

## Track B — CMB-Exam external evaluation

Use CMB-Exam as a cross-dataset external comparison after checking/removing overlap with SFT/RL data as reasonably possible.

Mandatory first target: a fixed, stratified **2,000-question clean subset** evaluated on all three primary checkpoints.

If the pinned local schema exposes the expected 28 subcategories, construct an approximately equal-per-category subset and freeze a **280-question SOTA probe (10/category)** from it. If the schema differs, record the actual deterministic stratification rather than forcing assumed fields.

If compute permits, extend to the full appropriate CMB test set. Whether full CMB is run or not must be stated explicitly.

Do not present CMB as guaranteed contamination-free pretraining evidence; use it primarily for matched relative comparison among post-training variants and external reference models.

## Track C — External open-ended clinical/medical QA

Open-ended evaluation is now a **formal secondary product-relevance evaluation**, not merely an informal sanity check.

Prefer the pinned CMB clinical component (CMB-Clin or locally available equivalent). If the expected small clinical-case set exists, retain all valid cases after decontamination. If unavailable, substitute an independent public Chinese open-ended medical QA/clinical set only after a decision record.

Score the **user-visible answer**, not hidden chain-of-thought. Proprietary reasoning models are not required to expose internal reasoning.

Primary reporting for this track:

- blinded pairwise win/tie/loss for Dynamic vs SFT and Dynamic vs Vanilla;
- fixed per-dimension rubric distributions;
- representative wins, ties, regressions, and safety failures.

## Track D — In-domain medical dialogue/QA retention

Construct a deterministic **200-item** held-out retention set from Stage 1 source material not used in the 20k SFT train set, preferably balanced between Medical-o1-style reasoning QA and Huatuo-style direct medical QA when enough clean samples remain.

Requirements:

- disjoint from SFT train;
- exact/near-duplicate filtered against SFT train and exam/RL/evaluation pools;
- frozen source mix, seed, IDs, hashes, and exclusions before model evaluation;
- explicitly reported as **in-domain behavior retention**, not external generalization.

Freeze a deterministic **100-item SOTA-probe subset** for low-cost third-party API comparison.

This track answers whether exam-style RL damaged normal medical QA behavior or pushed the model toward rigid choice-answering.

## Track E — Safety and behavior slice

Tag a fixed high-risk slice from Tracks C/D using source content only, before seeing candidate-model outputs. Useful categories include naturally occurring emergency/red-flag, medication/contraindication, pregnancy/pediatrics/elderly, insufficient-information, and overconfidence-sensitive cases.

Aim for at least 50 credible cases if the source pool supports them. Do not fabricate a large synthetic benchmark merely to satisfy a count.

Report a critical-safety-flag rate and rubric breakdown, with all flagged cases retained for review. This is a heuristic safety audit, not clinical certification.

## Open-ended evaluation rubric

Do not use BLEU/ROUGE or embedding similarity as the headline open-ended metric.

Freeze a rubric covering at least:

1. medical factual correctness;
2. relevance/completeness;
3. safety and escalation appropriateness;
4. uncertainty calibration;
5. clarity/usefulness of the visible response.

If an LLM judge is used:

- a candidate model must not judge itself;
- prefer a judge/provider distinct from the compared candidate when practical;
- pin judge model/version, prompt, rubric, A/B randomization, and raw outputs;
- randomize answer order and check position consistency on a subset.

At least 20% of open-ended items, plus all critical safety failures, require manual audit. If the reviewer is not a clinician, label it non-expert review and do not claim clinical validation.

## Third-party SOTA/reference baseline

A low-cost external probe may run before Stage 4 completes using the already frozen evaluation subsets. Recommended first bundle:

- CMExam SOTA probe: 512 questions;
- CMB-Exam SOTA probe: 280 questions when the 28-category schema is confirmed;
- all valid frozen CMB-Clin cases;
- 100-item held-out medical-QA SOTA probe.

For DeepSeek or any other external API, record the exact provider/model identifier exposed at execution time, date, endpoint/mode, accepted decoding parameters, max output tokens, prompt-template hash, tool/web/retrieval settings, raw responses, usage tokens, latency, retries, errors, and monetary cost when available.

Disable web/search/retrieval/tools when possible so the result is a closed-book model baseline. Transport/rate-limit failures may be retried; valid wrong or unparseable answers must not be resampled until correct.

Call these results `external reference baseline` or `SOTA probe`, not an online A/B test.

## Difficulty and category analysis

Use CMExam difficulty metadata where available. Report at least easy / medium / hard buckets (or a justified mapping from the original levels).

Analyze whether Dynamic Sampling gains/losses concentrate near medium/intermediate difficulty as hypothesized.

For CMB, report category/specialty breakdowns when sample sizes are adequate.

## Sampling/training-dynamics analysis

For the primary Stage 4 runs, produce time-series and final summaries for:

- all-correct group ratio;
- mixed/correctness-contrast group ratio;
- all-wrong group ratio;
- generated groups;
- accepted groups;
- sampling amplification;
- cumulative generated tokens;
- policy updates;
- unparseable/mixed-unparseable-only behavior where available.

## Fair efficiency comparisons

At minimum create:

1. **Validation accuracy vs cumulative generated tokens** — addresses rollout/compute efficiency.
2. **Validation accuracy vs policy updates / accepted-group progress** — addresses update efficiency.

Do not claim Dynamic Sampling is more compute-efficient solely because it achieves higher accuracy after the same number of updates if it used substantially more rollout tokens.

## Statistical analysis

For Vanilla GSPO vs Dynamic GSPO on the same held-out exam questions, perform a paired comparison such as:

- paired bootstrap for accuracy delta and 95% confidence interval; and/or
- McNemar test.

Report the method and actual interval/statistic. Do not treat a tiny raw delta as meaningful without context.

For open-ended pairwise evaluation, retain per-item outcomes so confidence intervals and order/position checks can be reconstructed.

## Reward sanity analysis

Using retained rollout/evaluation samples, quantify whether correctness gating prevented wrong answers with high semantic similarity from receiving large shaping reward.

This can be a diagnostic analysis rather than another training ablation. Do not evaluate the project primarily using the same semantic metric that was optimized during training.

## Required case comparisons

Mine paired cases across checkpoints, especially:

- SFT wrong -> Vanilla/Dynamic correct;
- SFT correct -> RL regression;
- Vanilla wrong -> Dynamic correct;
- Vanilla correct -> Dynamic wrong;
- medium/hard questions where methods diverge;
- open-ended QA where RL improves or damages usefulness;
- safety/uncertainty regressions;
- questions illustrating suspected reward/sampling failure modes.

Do not select only favorable cases for the final report; include meaningful regressions.

## Mandatory output artifacts

Produce machine-readable results sufficient to reconstruct the report, including:

- final per-question exam predictions/correctness for each primary checkpoint;
- deterministic evaluation manifests, seeds, source IDs, hashes, and exclusion reasons;
- `summary.csv/json`;
- CMExam difficulty breakdown;
- CMB clean-subset manifest and overlap/dedup report;
- CMB-Clin/open-QA raw visible responses;
- held-out dialogue-retention manifest and outputs;
- safety-slice manifest and flags;
- pairwise/rubric judge outputs and manual-audit records;
- external API baseline raw outputs/config/usage/cost metadata when run;
- sampling metrics table;
- paired statistical analysis;
- plot source data;
- key plots.

Required plots include at least:

- accuracy vs cumulative generated tokens;
- accuracy vs update/accepted-group progress;
- all-correct/mixed/all-wrong group distribution over training.

## Acceptance criteria

Stage 5 reaches `FULL_PASS` only if:

- all three primary checkpoints complete the full CMExam test protocol;
- all three complete the fixed clean external CMB-Exam subset protocol;
- all three complete the frozen open-ended external clinical/QA protocol;
- all three complete the 200-item held-out dialogue/QA retention protocol;
- the safety/behavior slice is evaluated or a documented source-availability limitation explains why it could not be credibly built;
- open-ended pairwise/rubric scoring and required manual audit are complete;
- difficulty/category analysis is complete;
- generated-token vs update efficiency analyses are complete;
- statistical comparison is complete;
- underlying per-question/per-response and plot-source data are retained;
- test data was not used for checkpoint selection/tuning;
- positive and negative cases are both retained;
- every headline number can be traced to a run and file.

Dynamic GSPO does not need to win for the stage to pass.

## Autonomous exploration

The agent may add useful diagnostics such as:

- additional specialty breakdowns;
- calibration/confidence proxies if available without changing generation behavior;
- response-length/correctness relationships;
- error taxonomy;
- subgroup analyses motivated by observed cases;
- one or more additional contemporary external API reference models when budget allows.

Optional analyses must not delay the mandatory evaluation indefinitely.

## Stage report must answer

1. Does SFT -> GSPO improve verifiable medical accuracy?
2. Does Dynamic Sampling outperform Vanilla under the same update budget?
3. Does it outperform under the same generated-token budget?
4. Do CMExam gains transfer to CMB-Exam?
5. Does exam-style RL preserve or improve open-ended medical QA/dialogue behavior?
6. Are there safety, uncertainty, or communication regressions?
7. Where by difficulty/category do gains and regressions occur?
8. What is the measured cost of refill?
9. Are differences statistically credible?
10. How far are the project checkpoints from strong third-party API baselines under the same frozen protocol?
11. What real cases explain the aggregate result?
12. Which project claims are supported, unsupported, or falsified?

## Interview-story deliverable

Produce an evidence-linked project results narrative that clearly separates exam correctness, cross-dataset generalization, open-ended medical QA behavior, safety limitations, update efficiency, rollout cost, and serving performance without overstating causality or clinical validity.
