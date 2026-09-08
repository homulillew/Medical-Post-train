# Autonomy and Exploration Policy

This project is designed for a strong implementation/research agent. The goal is not to prescribe every line of code. The goal is to protect the scientific contract while allowing substantial freedom in how the work is executed.

## 1. Operating principle

**Mandatory outcomes, autonomous means.**

The project documents define what must be demonstrated, compared, measured, and retained. The implementation agent owns the engineering path unless a hard constraint says otherwise.

## 2. Hard constraints

The agent must not silently change or bypass:

- the primary research question;
- the mandatory SFT / Vanilla GSPO / Dynamic GSPO comparison;
- minimum formal-run budgets in the current contract;
- train/validation/test separation;
- the rule that Dynamic Sampling acceptance is based on correctness contrast rather than total hybrid reward;
- evidence-retention requirements;
- preservation of negative results and failed hypotheses;
- the requirement to report generated-token cost separately from gradient-update count;
- the rule that smoke/pilot work is not full-run completion.

A hard constraint may be revised only through an explicit decision record marked as a **project-contract change**, with justification and owner approval when available.

## 3. Autonomous engineering zone

The agent may freely choose or revise, with normal documentation:

- package/repository architecture;
- data-loader implementation;
- dependency pins compatible with current upstream APIs;
- tokenizer/chat-template integration;
- LoRA target-module details;
- micro-batch size and gradient accumulation;
- gradient checkpointing and padding-removal choices;
- vLLM cache/memory settings;
- checkpoint format and resume mechanism;
- logging stack;
- unit/integration test implementation;
- answer-parser implementation;
- exact artifact directory layout;
- monitoring implementation;
- performance optimizations that preserve experiment semantics.

The agent is expected to improve these rather than mechanically follow obsolete examples.

## 4. Autonomous research zone

The agent may run low-cost exploratory or diagnostic experiments when they can clarify a meaningful uncertainty. Examples:

- compare rollout temperature 0.6 vs 0.8 on a small profiling subset;
- test group size 4 vs 6/8 on a small pilot;
- compare plausible LoRA target-module sets;
- evaluate whether MedEmbed semantic reward has pathological high similarity for wrong answers;
- inspect the effect of response-length limits;
- test an OOM mitigation strategy;
- examine whether the current RL pool is too easy/hard for the SFT policy.

Exploration is encouraged when it improves understanding or execution quality.

## 5. Rules for exploratory experiments

Every nontrivial exploration must be labeled `EXPLORATORY` or `DIAGNOSTIC`, have a unique `run_id`, and record:

- question/hypothesis;
- configuration;
- compute/data budget;
- result;
- interpretation;
- whether it changes a later formal configuration.

Exploratory runs **do not replace mandatory baselines** unless the project contract is explicitly revised.

## 6. Material changes require decision records

Create a decision record when changing something that affects interpretation, cost, or reproducibility, for example:

- backbone model or model size;
- full SFT sample count;
- primary RL pool definition;
- reward formula/weights in a material way;
- group size in the primary comparison;
- GSPO clipping/loss mode;
- formal accepted-group target;
- primary evaluation set;
- switching from BF16 LoRA to quantized training for a formal run;
- dropping a mandatory baseline.

A decision record should state alternatives, evidence, trade-offs, and which earlier runs remain comparable.

## 7. Unexpected discoveries

If an unexpected but potentially valuable phenomenon appears:

1. preserve the original evidence immediately;
2. add an observation entry;
3. capture representative cases when relevant;
4. formulate at least one alternative explanation;
5. run a small diagnostic if cost is reasonable;
6. do not rewrite the original hypothesis after the fact;
7. keep the primary project path moving unless the discovery invalidates it.

## 8. When the agent may stop or redirect execution

The agent may pause a formal run for correctness/safety reasons such as:

- persistent NaN/Inf;
- evidence of corrupted data/reward labels;
- implementation bug invalidating collected trajectories;
- nonrecoverable OOM under the documented configuration;
- checkpoint/resume corruption;
- a test-set leakage discovered mid-run.

Do not continue a known-invalid run merely to satisfy a numeric budget. Mark it invalid, preserve evidence, fix the root cause, and execute a valid replacement run.

The agent may not stop merely because the trend looks sufficiently clear.

## 9. Exploration budget discipline

The mandatory pipeline has priority. Avoid allowing optional ideas to consume the compute needed for the primary SFT, Vanilla GSPO, Dynamic GSPO, evaluation, and serving runs.

When proposing a significant optional experiment, estimate:

- GPU hours;
- generated tokens;
- expected information gained;
- whether it threatens the mandatory schedule.

## 10. Desired research behavior

The ideal agent is neither a blind executor nor an unconstrained optimizer. It should:

- challenge weak assumptions;
- notice confounders;
- improve implementation quality;
- preserve fair comparisons;
- separate observations from conclusions;
- quantify costs;
- save surprising cases;
- explain trade-offs;
- leave an audit trail from decision to result.
