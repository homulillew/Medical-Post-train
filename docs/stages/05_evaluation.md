# Stage 5 — Controlled Evaluation and Ablation

## Why

Turn training runs into defensible evidence. This stage must separate model capability, update efficiency, and rollout cost rather than reporting a single favorable score.

## Primary model states

Evaluate at minimum:

1. verified Medical SFT checkpoint;
2. primary Vanilla GSPO checkpoint;
3. primary Dynamic GSPO checkpoint.

Checkpoint selection must use validation data only. Test results may not select or tune checkpoints.

## Primary capability evaluation

### CMExam

Use the planned held-out **CMExam test split (6,811 questions)** for the main in-distribution capability comparison.

All three primary model states must be evaluated under a matched inference protocol.

Primary metric: exact/canonical medical answer accuracy.

### CMB external evaluation

Use CMB as a cross-dataset external comparison after checking/removing overlap with SFT/RL data as reasonably possible.

Mandatory first target: a fixed, stratified **2,000-question clean subset** evaluated on all three checkpoints.

If compute permits, extend to the full appropriate CMB test set. Whether full CMB is run or not must be stated explicitly.

Do not present CMB as guaranteed contamination-free pretraining evidence; use it primarily for matched relative comparison among post-training variants.

## Difficulty analysis

Use CMExam difficulty metadata where available. Report at least easy / medium / hard buckets (or a justified mapping from the original levels).

Analyze whether Dynamic Sampling gains/losses concentrate near medium/intermediate difficulty as hypothesized.

## Sampling/training-dynamics analysis

For the primary Stage 4 runs, produce time-series and final summaries for:

- all-correct group ratio;
- mixed/correctness-contrast group ratio;
- all-wrong group ratio;
- generated groups;
- accepted groups;
- sampling amplification;
- cumulative generated tokens;
- policy updates.

## Fair efficiency comparisons

At minimum create:

1. **Validation accuracy vs cumulative generated tokens** — addresses rollout/compute efficiency.
2. **Validation accuracy vs policy updates / accepted-group progress** — addresses update efficiency.

Do not claim Dynamic Sampling is more compute-efficient solely because it achieves higher accuracy after the same number of updates if it used substantially more rollout tokens.

## Statistical analysis

For Vanilla GSPO vs Dynamic GSPO on the same held-out questions, perform a paired comparison such as:

- paired bootstrap for accuracy delta and 95% confidence interval; and/or
- McNemar test.

Report the method and actual interval/statistic. Do not treat a tiny raw delta as meaningful without context.

## Reward sanity analysis

Using retained rollout/evaluation samples, quantify whether correctness gating prevented wrong answers with high semantic similarity from receiving large shaping reward.

This can be a diagnostic analysis rather than another training ablation. Do not evaluate the project primarily using the same semantic metric that was optimized during training.

## Open-ended sanity check

Because the RL source is exam-style choice QA while the project concerns medical QA/reasoning, retain a small independent open-ended medical QA sanity set or equivalent qualitative audit.

Purpose: detect obvious regression such as the model becoming rigidly choice-format-only. This is a guardrail, not the primary headline benchmark.

## Required case comparisons

Mine paired cases across checkpoints, especially:

- SFT wrong -> Vanilla/Dynamic correct;
- SFT correct -> RL regression;
- Vanilla wrong -> Dynamic correct;
- Vanilla correct -> Dynamic wrong;
- medium/hard questions where methods diverge;
- questions illustrating suspected reward/sampling failure modes.

Do not select only favorable cases for the final report; include meaningful regressions.

## Mandatory output artifacts

Produce machine-readable results sufficient to reconstruct the report, for example:

- final per-question predictions/correctness for each primary checkpoint;
- `summary.csv/json`;
- CMExam difficulty breakdown;
- CMB clean-subset manifest and overlap/dedup report;
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
- all three complete the fixed clean external CMB subset protocol;
- difficulty analysis is complete;
- generated-token vs update efficiency analyses are complete;
- statistical comparison is complete;
- underlying per-question and plot-source data are retained;
- test data was not used for checkpoint selection/tuning;
- positive and negative cases are both retained;
- every headline number can be traced to a run and file.

Dynamic GSPO does not need to win for the stage to pass.

## Autonomous exploration

The agent may add useful diagnostics such as:

- category/specialty breakdowns;
- calibration/confidence proxies if available without changing generation behavior;
- response-length/correctness relationships;
- error taxonomy;
- subgroup analyses motivated by observed cases.

Optional analyses must not delay the mandatory evaluation indefinitely.

## Stage report must answer

1. Does SFT -> GSPO improve medical accuracy?
2. Does Dynamic Sampling outperform Vanilla under the same update budget?
3. Does it outperform under the same generated-token budget?
4. Where by difficulty/category do gains and regressions occur?
5. What is the measured cost of refill?
6. Are differences statistically credible?
7. What real cases explain the aggregate result?
8. Which project claims are supported, unsupported, or falsified?

## Interview-story deliverable

Produce an evidence-linked project results narrative that can explain both favorable and unfavorable findings without overstating causality.
