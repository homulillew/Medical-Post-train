# Execution Workflow for the Implementation Agent

This document describes how a strong coding/research agent should execute the repository after the docs-only bootstrap.

## 1. Do not start with formal training

The first implementation session should **not** immediately download everything and launch a long SFT/RL job.

First produce an implementation plan grounded in:

- all project contracts in `docs/`;
- the actual available GPU/CPU/RAM/storage environment;
- current Qwen3 model interfaces;
- current `Transformers`, `PEFT`, `verl`, and `vLLM` APIs;
- dataset fields/licensing/revisions;
- single-48-GB memory constraints.

The plan should map every stage requirement to concrete code, configs, commands, artifacts, and verification logic.

## 2. Recommended execution loop

For each stage:

1. **Read** the stage contract and unresolved decision/observation records.
2. **Design** the implementation and experiment plan.
3. **Implement** the minimum robust infrastructure needed for the full run, not only a toy demo.
4. **Unit-test** parsers/rewards/samplers/counters as applicable.
5. **Smoke-test** the real stack on a deliberately tiny budget.
6. **Diagnose** bugs/memory/performance problems and create decision records for material changes.
7. **Pilot** when the stage calls for one.
8. **Freeze the formal configuration** and write the formal-run manifest.
9. **Execute/resume the full mandatory run** until its real budget is met or it becomes invalid/blocked.
10. **Preserve raw evidence**, cases, counters, artifacts, and environment metadata.
11. **Verify** every acceptance criterion.
12. **Write the stage report and interview story** from actual evidence.
13. Only then advance to the next stage.

## 3. First implementation deliverable

Before formal Stage 1 training, create an implementation-plan document based on `docs/templates/IMPLEMENTATION_PLAN.md` that includes:

- proposed repository architecture;
- exact upstream dependencies/revisions to pin;
- data acquisition/preprocessing plan;
- SFT implementation choice and memory estimate;
- `verl`/vLLM integration plan for single-GPU GSPO;
- reward/sampler architecture;
- checkpoint/resume design;
- run metadata/artifact layout;
- validation/acceptance tooling design;
- estimated GPU hours/storage;
- top risks and fallback strategies;
- which choices are still exploratory vs already fixed by the contract.

Review the plan against the project charter before expensive execution.

## 4. Stage-gate behavior

Do not chain all six stages into one opaque script that makes it difficult to inspect evidence.

Each stage should expose reproducible commands for smoke/pilot/formal execution and verification. Automation is welcome, but stage boundaries and run identities must remain observable.

A later stage should consume an explicit verified artifact from the earlier stage, not an ambiguous filesystem path such as `latest/` without a manifest.

## 5. Long-running jobs

Long SFT/RL jobs must be designed with interruption in mind.

Before launching:

- verify sufficient disk space;
- establish log/metric paths;
- establish checkpoint cadence;
- define planned budget counters;
- test resume on a short run when required;
- record the exact command/config.

If an interactive agent session ends while an external process continues, the process status/artifact location must be written to durable project state/logs. A future session should inspect and resume/observe the existing run rather than assume it is complete or restart blindly.

## 6. Full-run completion

The implementation agent must distinguish:

- **the code path works**;
- **the formal experiment is currently running**;
- **the formal budget has been reached**;
- **the result has been verified and interpreted**.

For example, 40 RL steps with a valid checkpoint prove integration, not the Stage 4 hypothesis.

## 7. Freedom to improve

The agent is expected to use current best judgment. If upstream APIs differ from the docs' examples, adapt them. If a memory setting is inefficient, improve it. If a parser/reward assumption is wrong, fix it and preserve the invalid-run record.

What must remain stable is the scientific comparison and evidence contract, not obsolete implementation syntax.

## 8. Exploration workflow

For a potentially valuable side observation:

1. log the observation;
2. state a hypothesis and alternative explanation;
3. estimate diagnostic cost;
4. run a small labeled diagnostic if worthwhile;
5. record result/decision;
6. return to the mandatory path.

Do not let optional explorations consume the compute reserved for the primary formal runs without an explicit decision.

## 9. Data/case harvesting during execution

Do not postpone qualitative analysis until the final day. During every stage:

- retain representative bad/good/boundary cases;
- snapshot surprising reward/group examples;
- record major training-dynamics changes;
- save paired checkpoint disagreements during evaluation;
- record system failures and their resolution.

These records feed the final interview story and often reveal bugs before aggregate metrics do.

## 10. End-of-stage output

Every completed stage should leave behind three layers:

1. **Machine evidence:** configs, metrics, manifests, predictions/counters, artifact references.
2. **Research interpretation:** observations, decisions, cases, stage report.
3. **Interview material:** 30-second summary, 2-minute story, deep-dive questions/answers, concrete numbers and cases.

## 11. Final project retrospective

After Stage 6, synthesize all stage reports into a final retrospective that distinguishes:

- what was planned;
- what actually happened;
- which hypotheses were supported/falsified;
- which engineering constraints shaped the design;
- final measured capability/cost/serving results;
- strongest bad/good cases;
- defensible resume claims with exact evidence paths;
- limitations and next steps.
