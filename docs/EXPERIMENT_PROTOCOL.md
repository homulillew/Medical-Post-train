# Experiment Protocol

Every meaningful execution must be reproducible enough that a future agent can answer: **what ran, on which code/config/data, for how long, what happened, and where is the evidence?**

## 1. Run identity

Every smoke, pilot, formal, diagnostic, or exploratory experiment gets a unique `run_id`.

Recommended pattern:

`<stage>_<purpose>_<model>_<short-config>_<nnn>`

Examples:

- `s1_sft_qwen3-8b_r32_001`
- `s2_rollout_profile_t07_g4_001`
- `s4_gspo_vanilla_001`
- `s4_gspo_dynamic_001`
- `s4_diag_temp08_001`

Never reuse a run ID for a different execution.

## 2. Run classes

Each run must declare exactly one class:

- `SMOKE`
- `PILOT`
- `FORMAL`
- `DIAGNOSTIC`
- `EXPLORATORY`
- `EVALUATION`
- `SERVING_BENCHMARK`

This label prevents a small run from being misrepresented later.

## 3. Required run record

Use an equivalent of:

```text
runs/<run_id>/
├── config.yaml
├── command.txt
├── environment.json
├── git_commit.txt
├── status.json
├── metrics.jsonl
├── summary.json
├── stdout.log
├── stderr.log
├── observations.md
└── artifacts_manifest.json
```

The implementation may organize storage differently if a reliable manifest maps these concepts.

## 4. Required metadata

At minimum record:

- run ID and class;
- stage;
- start/end timestamps;
- git commit SHA and dirty-state indication;
- hostname/container image if relevant;
- GPU name, count, VRAM, CUDA/driver;
- Python version;
- key package versions (`torch`, `transformers`, `peft`, `verl`, `vllm`, tokenizer/model revision where possible);
- random seed(s);
- model/base revision;
- dataset manifest/version/hash/subset selection;
- exact command/config;
- planned budget and actual completed budget;
- exit status and reason.

## 5. Budget counters

Record counters appropriate to the run, not only optimizer steps. For RL, prefer retaining:

- generated prompt groups;
- generated responses/trajectories;
- accepted groups;
- policy updates;
- prompt tokens;
- output/generated tokens;
- training tokens where available;
- elapsed wall-clock time;
- peak VRAM.

For Dynamic Sampling, generated and accepted counts must remain distinguishable.

## 6. Metric logging

Use append-only structured metrics where practical. For RL formal runs, retain at least:

- total reward;
- accuracy/correctness reward;
- semantic reward;
- format reward;
- policy entropy or closest available entropy diagnostic;
- clip fraction;
- gradient norm;
- response length;
- all-correct group ratio;
- mixed/correctness-contrast group ratio;
- all-wrong group ratio;
- sampling amplification for Dynamic Sampling;
- validation accuracy at planned checkpoints.

If a framework does not expose an exact metric, record the closest meaningful signal and document the difference rather than inventing it.

## 7. Formal-run manifest

Before launching a formal run, create a manifest containing:

- purpose/hypothesis;
- initial checkpoint;
- fixed comparison settings;
- planned full budget;
- stop/invalidity conditions;
- checkpoint interval;
- evaluation schedule;
- expected artifact paths.

A formal run is not made formal merely by running longer than expected; it must be identified as formal before interpretation.

## 8. Invalid runs

A run is `INVALID` when evidence shows that its scientific result cannot be interpreted, for example:

- reward labels/parser were wrong;
- test data leaked into training;
- wrong adapter/checkpoint loaded;
- severe implementation bug;
- metrics/counts were corrupted;
- formal configuration accidentally differed in a confounding way.

Do not delete invalid runs. Mark them invalid, document the root cause, and retain enough evidence to avoid repeating the failure.

## 9. Experiment comparisons

When comparing two runs, explicitly list variables that are intended to be identical and those intended to differ.

For the primary Vanilla vs Dynamic GSPO comparison, create a comparison record verifying at least:

- same SFT initialization;
- same reward implementation/version;
- same model/LoRA architecture;
- same group size;
- same nominal GSPO settings;
- same evaluation protocol;
- sampling intervention is the principal intended difference.

If a difference was unavoidable, document it as a confounder.

## 10. Observation vs conclusion

Run reports must distinguish:

- **Observation:** direct measured fact;
- **Hypothesis:** proposed explanation;
- **Alternative explanation:** plausible confounder;
- **Conclusion:** interpretation supported after analysis.

Example:

> Observation: mixed-group ratio rose from X to Y between checkpoints.
>
> Hypothesis: the policy moved from an all-wrong region toward its capability frontier.
>
> Alternative: a generation-temperature change could produce the same pattern.
>
> Conclusion: only make this claim after checking whether generation settings were constant.

## 11. Full data retention

Headline summaries are insufficient. Preserve full experiment metrics and key per-sample/per-group outputs needed to reproduce analysis, subject to storage constraints described in `ARTIFACT_RETENTION.md`.

For large rollout datasets, keep a manifest and a durable local/external artifact copy even if Git stores only summaries/cases.

## 12. No retroactive metric selection

Do not search many metrics and report only the favorable one without noting the search. Primary metrics should be declared in stage specifications before final evaluation.

## 13. Stage transition

Before advancing to the next stage, produce a stage report that references concrete run IDs and artifact paths. The next stage should consume explicit verified artifacts rather than an ambiguous “latest” directory.
