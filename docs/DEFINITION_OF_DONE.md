# Definition of Done

This document defines completion semantics for all stages. Its purpose is to prevent partial execution from being mislabeled as a completed project.

## 1. Stage states

Every stage must be described with one of these states:

`NOT_STARTED -> IMPLEMENTED -> SMOKE_PASS -> PILOT_PASS -> FULL_RUNNING -> FULL_PASS -> VERIFIED -> DONE`

Additional terminal/problem states are allowed:

- `BLOCKED`: progress cannot continue because of an external dependency/resource issue.
- `FAILED`: the planned experiment could not be completed after documented recovery attempts.

`BLOCKED` and `FAILED` are not aliases for `DONE`.

## 2. State meanings

### NOT_STARTED

No meaningful implementation or execution exists.

### IMPLEMENTED

Required code/configuration exists and basic static/unit checks are reasonable. No claim about runtime viability is implied.

### SMOKE_PASS

A deliberately tiny execution verifies integration basics such as loading, forward/backward, reward parsing, checkpoint creation, or service startup.

A smoke test may use tiny data and a few steps. **It is never evidence that the formal experiment is complete.**

### PILOT_PASS

A limited but nontrivial run shows that the planned configuration is stable enough to justify full execution. Pilot results may support hyperparameter/engineering decisions but do not replace the formal run.

### FULL_RUNNING

The formal run has started with the planned full-run configuration/budget and is incomplete.

### FULL_PASS

The mandatory formal budget was actually reached and the run ended without invalidating errors. Required raw logs/checkpoints exist.

### VERIFIED

The full run has been checked against the stage acceptance criteria: counts, artifacts, integrity, reproducibility evidence, metrics, and required analysis are present.

### DONE

`VERIFIED` plus the stage's retrospective/report/interview-story deliverables are complete. `DONE` means both execution and understanding were captured.

## 3. Forbidden completion shortcuts

None of the following is sufficient to mark a stage `FULL_PASS`, `VERIFIED`, or `DONE`:

- a process launched successfully;
- loss is finite;
- a checkpoint exists;
- 10, 20, 50, or 100 RL steps completed;
- reward appears to increase;
- early validation looks promising;
- a chart can be drawn;
- a trend appears obvious;
- the agent judges that more training is probably unnecessary;
- compute time feels long;
- a framework example script completed on reduced settings.

A reduced run must be labeled smoke, pilot, diagnostic, or exploratory according to its actual purpose.

## 4. Mandatory formal-budget rule

Each stage specification defines its own formal budget. Formal budgets are contracts, not suggestions.

Examples in the initial plan include:

- Stage 1: 20,000 SFT examples and one complete planned epoch;
- Stage 2: 1,000 profiling prompts with group size 4 (4,000 completed responses);
- Stage 3: at least 256 accepted mixed groups in real refill integration;
- Stage 4: 5,000 accepted groups for each primary formal GSPO run;
- Stage 5: full planned held-out evaluation sets;
- Stage 6: all required serving concurrency conditions and minimum real request counts.

A formal budget may be changed only through an explicit project-contract decision **before** the affected final run is interpreted as the primary experiment. Resource exhaustion during a run does not automatically authorize a smaller definition of completion.

## 5. Resume rule

Long runs should be resumable when practical. If execution is interrupted:

1. preserve logs and state;
2. identify the latest valid checkpoint;
3. verify counters/state before resuming;
4. continue toward the original formal budget;
5. record the interruption and recovery.

Restarting from scratch is acceptable only when resume integrity cannot be trusted; document why.

## 6. Evidence required for verification

A stage cannot be `VERIFIED` if the result exists only in conversational memory. At minimum, verification should be backed by machine-readable or file-based evidence such as:

- run/config manifest;
- command used;
- environment/dependency snapshot;
- code revision;
- raw/structured metrics;
- planned-budget counters;
- artifact/checkpoint manifest;
- evaluation outputs;
- relevant cases/observations;
- stage summary/report.

## 7. Negative results

`FULL_PASS` and `VERIFIED` do not require an improvement over baseline.

Examples of valid completed outcomes:

- Dynamic GSPO scores below Vanilla GSPO after a fair full run;
- Dynamic Sampling improves update efficiency but increases rollout cost enough to erase token-budget gains;
- semantic shaping creates a measurable failure mode and must be reduced;
- no statistically meaningful difference is observed.

These outcomes become valuable when evidence and interpretation are complete.

## 8. Human/agent sign-off checklist

Before calling a stage `DONE`, answer yes to all applicable questions:

- Did the formal run reach the exact/minimum mandatory budget?
- Are the primary baselines preserved?
- Are test sets untouched by tuning/selection?
- Are all important raw metrics and configs retained?
- Are failures/regressions retained?
- Can important checkpoints/artifacts be reloaded?
- Are major deviations captured in decision records?
- Are valuable model/system cases saved?
- Is the stage report based on real evidence?
- Can every headline claim be traced back to a run/artifact?

If any required answer is no, the stage is not `DONE`.
