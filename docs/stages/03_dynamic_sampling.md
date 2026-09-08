# Stage 3 — DAPO-style Dynamic Sampling

## Why

Validate the project's central sampling intervention independently before committing expensive full GSPO training.

## Scientific question

Can on-policy group filtering reliably select prompts with **correctness contrast** while rejecting all-correct/all-wrong groups, and what refill/sampling overhead does this introduce?

## Required inputs

- Verified Stage 1 SFT policy.
- Verified Stage 2 RL candidate pool, parser, `R_acc`, and hybrid reward.
- Group rollout implementation using the documented generation policy.

## Sampling rule

For prompt `x`, generate `G` responses and compute binary correctness values.

With planned `G=4`:

- `[1,1,1,1]` -> reject for this update;
- `[0,0,0,0]` -> reject for this update;
- any mixture of 0 and 1 -> accept.

Equivalently, accept when:

`0 < mean(R_acc) < 1`.

**Acceptance must use `R_acc` only.** The continuous hybrid training reward must not decide whether a group is accepted.

## On-policy rule

Filtering applies to the **current rollout group**, not permanently to the prompt.

Do not create a once-and-for-all easy/hard blacklist. As the policy changes, a prompt may move between all-wrong, mixed, and all-correct regions.

## Mandatory unit tests

At minimum verify:

- `[1,1,1,1]` rejected;
- `[0,0,0,0]` rejected;
- `[1,0,1,0]` accepted;
- `[1,1,1,0]` accepted;
- `[0,0,0,1]` accepted;
- total hybrid reward cannot override the decision;
- grouping is by the correct prompt ID and cannot mix trajectories across prompts.

## Smoke criteria

Use a tiny real rollout to exercise generate -> reward -> group -> filter. Confirm counters and records are correct.

## Mandatory real refill integration

Run the real generation/filter/refill path until at least **256 accepted mixed groups** have been accumulated.

The purpose is not model training performance; it is to prove that the same mechanics intended for Stage 4 work under real vLLM/model outputs.

Do not implement the stage as an offline filter over a static pre-generated dataset and claim that on-policy refill is complete.

## Required counters

Retain at least:

- generated prompt groups;
- generated trajectories/responses;
- accepted mixed groups;
- rejected all-correct groups;
- rejected all-wrong groups;
- generated prompt/output tokens;
- elapsed generation time;
- refill iterations/batches.

Compute:

`sampling_amplification = generated_groups / accepted_groups`.

Also report the share of generated groups in each correctness category.

## Important interpretation constraint

Dynamic Sampling does **not** automatically save rollout compute. Rejected groups are identified only after generation/reward evaluation.

Therefore:

- do not claim rollout-token reduction without measurement;
- distinguish update efficiency from generation efficiency;
- report sampling amplification and generated tokens alongside accepted-group count.

## Acceptance criteria

Stage 3 reaches `FULL_PASS` only if:

- all mandatory unit tests pass;
- real vLLM/model refill integration reaches >=256 accepted mixed groups;
- generated/accepted/rejected accounting is internally consistent;
- acceptance uses correctness only;
- prompt filtering is on-policy/nonpermanent;
- full integration metrics and representative groups are retained;
- the resulting sampling path is usable by Stage 4 without manual intervention.

## Case-study capture

Retain representative:

- all-correct easy groups;
- all-wrong too-hard/currently-unsolved groups;
- balanced mixed groups;
- highly imbalanced mixed groups (1/4 or 3/4 correct);
- prompts whose group category changes across repeated/policy-updated rollout if observed;
- any refill starvation/pathology.

## Autonomous exploration

The agent may explore, on small budgets:

- group size 4 vs a larger group size;
- generation temperature effects on correctness contrast;
- batching/refill implementations;
- whether difficulty annotations correlate with group categories;
- alternative scheduling that improves throughput without changing acceptance semantics.

Any change to the primary Stage 4 group size/sampling rule requires a decision record.

## Stage report must answer

1. What problem is Dynamic Sampling actually solving?
2. What percentage of the SFT policy's groups have correctness contrast?
3. What is the measured sampling amplification?
4. Does the intervention save rollout compute, update compute, neither, or both in the measured implementation?
5. Which question types/difficulties dominate all-correct/all-wrong/mixed groups?
6. What failure modes appeared in refill?
7. Why is the sampler kept separate from the GSPO loss?

## Interview-story deliverable

Produce a story that explicitly distinguishes **sampling selection**, **rollout cost**, and **policy optimization**, supported by real group examples and measured amplification rather than a generic “hard-sample mining” claim.
