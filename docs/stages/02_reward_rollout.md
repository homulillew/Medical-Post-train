# Stage 2 — RL Dataset, Rollout Profiling, and Hybrid Reward

## Why

Build a clean, verifiable medical RL environment before policy optimization. This stage must prove that answer parsing, reward decomposition, group rollout, and profiling are trustworthy enough for Dynamic Sampling and GSPO.

## Scientific question

Can we obtain a reliable sequence-level reward that keeps final-answer correctness dominant while using semantic explanation similarity only as gated reward shaping?

## Required inputs

- Verified Stage 1 SFT adapter.
- CMExam train split as the planned RL source.
- CMExam reference answer and explanation fields.
- A medical embedding model, initially planned as MedEmbed, after the implementation agent verifies a suitable available model/version.

## RL candidate pool

Construct a **15,000-prompt candidate pool** from CMExam train, preferably stratified by available difficulty/category fields.

The pool must not overlap CMExam validation/test used for later selection/evaluation.

Preserve source IDs, selection seed, counts, and split/dedup evidence.

## Output format

Use a stable machine-parsable response format. The planned logical form is:

```text
<think>
...
</think>
<answer>...</answer>
```

The implementation agent may adapt to the current Qwen3 chat template as long as reasoning and final-answer extraction remain unambiguous.

## Reward hypothesis

Initial formal design:

`R_total = 0.8 * R_acc + 0.15 * R_acc * R_sem + 0.05 * R_format`

Where:

- `R_acc`: verifiable exact/canonical answer correctness;
- `R_sem`: normalized semantic similarity between model reasoning and reference explanation;
- `R_format`: valid structured-output indicator.

Key property: a wrong final answer receives **no semantic shaping reward** through the `R_acc * R_sem` gate.

The Dynamic Sampling metric is separate and must be `R_acc`, not `R_total`.

Material changes to the reward require a decision record and evidence.

## Parser requirements

Support CMExam single-/multi-choice answer representations robustly. Canonicalize harmless formatting/order differences where semantically appropriate while keeping format reward separable from correctness.

Required unit cases should cover at least:

- lowercase/uppercase;
- whitespace/punctuation;
- strict tagged answer;
- fallback answer extraction;
- multi-choice ordering/canonicalization;
- malformed/no-answer response;
- multiple ambiguous answer spans.

## Smoke criteria

Run a small real rollout, for example **50 prompts x group size 4**, and verify:

- SFT adapter is actually loaded;
- response generation works;
- parser output is correct on manually inspected samples;
- all reward components are finite/in-range;
- wrong-answer high semantic similarity cannot create a high total reward;
- output records are serializable and traceable to prompt IDs.

This is not the formal profiling run.

## Mandatory full profiling

Execute **1,000 distinct prompts x group size 4 = 4,000 completed responses** using a fixed documented generation configuration.

Do not stop early after seeing a useful group distribution.

Each response record must retain at least:

- prompt/source ID;
- question/options;
- ground truth;
- raw response;
- parsed answer;
- correctness reward;
- semantic reward;
- format reward;
- total reward;
- response token count;
- generation metadata sufficient for analysis.

Each group must retain/derive:

- group accuracy;
- all-correct / mixed / all-wrong label;
- group reward mean/std;
- response-length statistics.

## Required profiling outputs

Report at least:

- all-correct ratio;
- correctness-contrast (mixed) ratio;
- all-wrong ratio;
- answer-parser fallback/error rate;
- reward-component distributions;
- semantic-score distribution split by correct vs wrong answers;
- response-length distribution;
- estimated rollout throughput/cost on the actual GPU.

## Acceptance criteria

Stage 2 reaches `FULL_PASS` only if:

- candidate pool contains 15,000 valid prompts;
- formal profiling contains exactly/plausibly >=1,000 unique prompts and >=4,000 completed trajectories;
- parser tests pass and manual audit finds no unresolved systematic parsing bug;
- reward values are finite and definitions match the recorded formula;
- `R_acc` is retained separately for Stage 3 filtering;
- full profiling data/manifest and summary statistics are retained;
- no validation/test contamination is discovered.

No specific mixed-group percentage is required to pass; an extreme distribution is itself a result to analyze.

## Case-study capture

Especially retain:

- wrong answers with unusually high `R_sem`;
- correct answers with low `R_sem`;
- malformed outputs that are still answer-correct;
- high-total-reward and low-total-reward examples;
- representative all-correct, mixed, and all-wrong groups;
- parser ambiguities and fixes.

## Autonomous exploration

Encouraged low-cost diagnostics include:

- rollout temperature comparison on a small fixed subset;
- response-length cap sanity test;
- alternate normalization for semantic similarity;
- analysis of whether semantic shaping meaningfully ranks correct responses;
- manual medical review of selected high/low semantic cases.

Exploration may improve the formal Stage 4 configuration but must not replace the mandatory Stage 2 profiling dataset.

## Stage report must answer

1. Why is ground-truth correctness superior to an LLM judge for this RL source?
2. Why is semantic reward gated by correctness?
3. Does MedEmbed actually differentiate reasoning quality among correct answers?
4. What failure cases exist in parser/reward logic?
5. What fraction of groups are all-correct, mixed, and all-wrong?
6. Is the RL pool positioned near the SFT policy's capability frontier?
7. What rollout configuration will Stage 4 inherit?

## Interview-story deliverable

Produce a concise story around **verifiable reward + gated shaping**, including at least one real high-semantic/wrong-answer case demonstrating why the gate matters if such cases exist.
