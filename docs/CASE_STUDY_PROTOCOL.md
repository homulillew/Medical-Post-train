# Case Study Protocol

Quantitative metrics are not enough for this project. The repository must preserve representative model behaviors and system failures that explain **why** a metric changed and provide evidence for later technical interviews.

## 1. Case categories

Capture cases using one of these categories where applicable:

- `BAD_CASE`: clearly undesirable model/system behavior;
- `GOOD_CASE`: behavior that demonstrates a meaningful capability improvement;
- `BOUNDARY_CASE`: current policy sometimes succeeds and sometimes fails under group rollout;
- `REGRESSION_CASE`: a later checkpoint becomes worse than an earlier checkpoint;
- `REWARD_CASE`: reward components reveal a useful or pathological discrepancy;
- `SAMPLING_CASE`: group behavior illustrates Dynamic Sampling decisions;
- `SYSTEM_CASE`: OOM, checkpoint, serving, parser, or framework issue with useful engineering lessons.

## 2. Suggested subtypes

Examples include:

- answer-parser failure;
- high semantic similarity but incorrect final answer;
- reward hacking;
- overly long reasoning;
- format collapse;
- all-correct group;
- all-wrong group;
- high-value mixed group;
- SFT wrong -> RL correct;
- SFT correct -> RL wrong;
- Vanilla wrong -> Dynamic correct;
- Vanilla correct -> Dynamic wrong;
- medical reasoning inconsistency;
- checkpoint/resume mismatch;
- training/serving behavior mismatch;
- memory/performance pathology.

The agent may add new subtypes when useful.

## 3. Minimum structured fields

Each retained model case should preserve enough information to understand it later. Recommended fields:

```json
{
  "case_id": "...",
  "run_id": "...",
  "stage": 4,
  "category": "BOUNDARY_CASE",
  "subtype": "mixed_group",
  "prompt_id": "...",
  "prompt": "...",
  "ground_truth": "...",
  "responses": [],
  "parsed_answers": [],
  "reward_components": {},
  "why_interesting": "...",
  "observation": "...",
  "hypothesis": "...",
  "alternative_explanations": [],
  "followup": "..."
}
```

For system cases, use an analogous structure with environment, error, reproduction command, attempted fixes, root cause, and final resolution.

## 4. Automatic case mining

Where practical, automatically retain top/representative examples for:

- highest and lowest total rewards;
- longest responses;
- highest semantic reward among wrong answers;
- lowest semantic reward among correct answers;
- each group-accuracy bucket;
- all-correct/all-wrong/mixed groups;
- evaluation disagreements between SFT, Vanilla GSPO, and Dynamic GSPO;
- serving outputs that differ unexpectedly from offline evaluation.

Automatic mining should complement, not replace, manually identified interesting cases.

## 5. Stage-specific emphasis

### Stage 1

Capture malformed training examples, extreme token lengths, duplicated/near-duplicated examples, and before/after SFT examples showing meaningful medical-domain adaptation.

### Stage 2

Capture parser ambiguity, incorrect answers with high semantic similarity, correct answers with poor explanation similarity, format failures, and representative reward-component disagreements.

### Stage 3

Capture easy all-correct groups, too-hard all-wrong groups, informative mixed groups, and prompts whose group classification changes under repeat/on-policy rollout.

### Stage 4

Capture signs of reward hacking, entropy/length collapse, strong policy improvements, regressions, and representative trajectories around training-dynamics changes.

### Stage 5

Capture paired evaluation disagreements across the three checkpoints, especially medium/hard questions and statistically influential patterns.

### Stage 6

Capture offline-vs-serving mismatches, latency outliers, malformed API responses, and concurrency-related failures.

## 6. Preservation rule

Do not overwrite a case because a later explanation is better. Preserve the original observation, then append or link the revised interpretation.

Large raw responses may live outside Git, but Git should retain compact representative cases and a manifest pointing to the full artifact.

## 7. Interview value

At the end of each stage, select 2-5 cases that best explain:

- the hardest engineering problem;
- the most important model behavior;
- a failed hypothesis or surprising result;
- a concrete decision and trade-off.

These cases should feed the stage report rather than being presented as cherry-picked proof of overall performance.
