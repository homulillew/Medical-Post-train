# AGENTS.md

This repository is an experiment-driven medical LLM post-training project. Read this file first, then read the referenced project documents before implementing or running anything.

## Source of truth

Read in this order:

1. `docs/PROJECT_CHARTER.md`
2. `docs/DEFINITION_OF_DONE.md`
3. `docs/AUTONOMY_AND_EXPLORATION.md`
4. `docs/EXPERIMENT_PROTOCOL.md`
5. The current stage specification under `docs/stages/`
6. Relevant templates/protocols under `docs/templates/` and `docs/CASE_STUDY_PROTOCOL.md`

If implementation convenience conflicts with those documents, preserve the scientific contract and acceptance criteria.

## Mandatory behavior

- Treat **smoke tests, pilot runs, launched processes, partial checkpoints, and a few dozen RL steps as incomplete work**.
- Never mark a stage complete because loss looks reasonable or because a trend is visible early.
- Mandatory full-run budgets and baselines may not be silently reduced.
- Never use test-set results to choose a checkpoint, tune hyperparameters, or redefine the experiment.
- Never fabricate, interpolate, or backfill metrics. Missing evidence must remain missing and be reported.
- Negative results, regressions, crashes, reward pathologies, and unexpected behavior are part of the project record and must be retained.
- Do not delete or overwrite a previous experiment to make a new result look cleaner. New runs get new run IDs.
- Preserve enough state to resume interrupted long-running jobs when practical.

## Autonomy

You are encouraged to use strong engineering and research judgment. You may autonomously:

- choose repository/code architecture;
- adapt to current `verl`, `vLLM`, `Transformers`, `PEFT`, CUDA, or dependency APIs;
- tune micro-batching, gradient accumulation, cache use, checkpoint format, and memory settings;
- fix bugs and improve parser/reward/data implementations;
- run low-cost diagnostic or exploratory experiments;
- adjust implementation details that do not invalidate the mandatory comparison;
- propose better alternatives when evidence supports them.

When an exploration changes a material assumption, create a decision record and preserve the mandatory baseline unless the project owner explicitly changes the contract.

## Mandatory baselines

The project must retain these primary checkpoints/conditions unless the project contract is explicitly revised:

1. Medical SFT checkpoint
2. SFT + Vanilla GSPO
3. SFT + DAPO-style Dynamic Sampling + GSPO

The main Dynamic-Sampling comparison must begin from the same SFT initialization and hold the principal GSPO/reward settings constant except for the sampling intervention.

## Experiment evidence

Every meaningful run must have a unique `run_id` and retain configuration, command, environment, code revision, metrics, status, observations, and artifact manifest according to `docs/EXPERIMENT_PROTOCOL.md`.

Capture valuable model cases and system failures according to `docs/CASE_STUDY_PROTOCOL.md`.

After every stage, produce the stage report required by that stage. A stage report must be based on real artifacts and experiments, not a prospective description.

## Completion gate

A stage is not complete until all of the following are true:

1. Mandatory implementation exists.
2. Required smoke/pilot checks have passed.
3. Mandatory full-run budget has actually been reached.
4. Required artifacts and raw evidence exist.
5. Acceptance checks pass.
6. Stage report and interview-story material are complete.

If a future repository validator exists, its successful result is required but does not permit bypassing the documented scientific acceptance criteria.

If a mandatory full run cannot be completed, mark the work `FAILED` or `BLOCKED`, explain why, preserve all evidence, and do not report the stage as done.

## First task for a fresh implementation agent

Do **not** begin formal training immediately. First inspect all project documents, the available hardware/software environment, and current upstream library APIs. Then create an implementation plan and risk assessment that maps the stage contracts to concrete code, commands, artifact paths, and expected resource use. Smoke tests may follow only after that plan is coherent.
