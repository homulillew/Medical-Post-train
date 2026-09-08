# Stage 6 — LoRA Export and vLLM Serving

## Why

Close the full post-training loop with a reproducible deployable artifact and measure serving behavior using standard latency/throughput metrics without turning the project into a separate inference-optimization study.

## Scope

Mandatory deployment path:

`Qwen3-8B base + selected final LoRA adapter -> vLLM -> OpenAI-compatible API`

Optional merge/export tooling may be provided, but LoRA-native serving is the primary planned path.

Out of mandatory scope:

- quantization research;
- TensorRT;
- speculative decoding;
- RAG;
- agent tools;
- multi-node serving.

## Checkpoint selection

Select the deployed checkpoint using the predefined **validation** protocol only.

Do not use CMExam test or external CMB test performance to choose the checkpoint.

Record:

- selected run ID;
- checkpoint/adapter hash;
- validation criterion;
- base model revision;
- LoRA config.

## Offline-to-serving consistency check

Before benchmarking speed, use a fixed set of at least **100 prompts** to verify:

- intended adapter is loaded;
- chat template/thinking mode is correct;
- answer format/parser remains compatible;
- generation settings are documented;
- serving outputs/accuracy are not unexpectedly inconsistent with offline evaluation.

Any significant mismatch must be investigated before serving benchmarks are treated as valid.

## API requirement

Start a real vLLM OpenAI-compatible service and successfully exercise a chat-completion endpoint end-to-end.

Retain launch command/config, vLLM version, model/adapter identity, and a sample successful request/response.

## Serving benchmark

Benchmark at minimum these concurrency conditions:

- 1
- 4
- 8
- 16

Use a fixed reproducible workload based on real project-style prompts or a documented representative distribution.

Record at least:

- TTFT P50 and P95;
- TPOT P50 and P95;
- end-to-end latency P50 and P95;
- output throughput (tokens/s);
- request throughput (requests/s);
- output-token statistics;
- peak VRAM or best available memory measurement.

The benchmark should complete at least **100 real requests total**, and enough per condition to make reported percentiles meaningful. If the implementation agent chooses a higher minimum, document it.

## Interpretation

Avoid the weak headline “single inference takes 2–5 seconds.” Latency must be interpreted alongside prompt/output length and concurrency.

If RL checkpoints produce longer reasoning, separate:

- model/service token-generation speed;
- increased end-to-end latency caused by longer outputs.

Where useful, compare SFT and final Dynamic GSPO adapters under the same serving setup to identify capability/length/latency trade-offs.

## Artifact/export requirements

Retain:

- final adapter or durable artifact reference/hash;
- base model/revision reference;
- serving command/config;
- environment/package versions;
- consistency-check predictions/results;
- raw serving benchmark output;
- summarized benchmark CSV/JSON;
- deployment runbook.

If a merged model is produced, record its provenance/hash separately from the LoRA-native artifact.

## Case-study capture

Capture:

- offline vs serving mismatches;
- invalid/malformed API outputs;
- latency outliers;
- memory failures at higher concurrency;
- changes needed to chat templates/stopping rules;
- representative SFT vs GSPO response-length differences.

## Autonomous exploration

The agent may tune within the same serving semantics:

- vLLM GPU memory utilization;
- batching/concurrency settings;
- max model length;
- LoRA loading mode;
- benchmark tooling;
- cache settings;
- service process layout.

Do not add quantization or a new serving engine to the mandatory benchmark unless documented as an optional exploration.

## Acceptance criteria

Stage 6 reaches `FULL_PASS` only if:

- final selected LoRA artifact reloads successfully;
- vLLM service starts with the intended model/adapter;
- OpenAI-compatible API requests succeed;
- >=100-prompt offline/serving consistency check is complete;
- concurrency 1/4/8/16 benchmarks are complete;
- required latency/throughput metrics and workload stats are retained;
- serving results are reproducible from the deployment runbook;
- no unexplained major offline-serving accuracy mismatch remains.

## Stage report must answer

1. How is the final LoRA artifact exported/identified?
2. Why use native LoRA serving rather than merge/quantization as the main path?
3. What deployment mismatch bugs were found?
4. What are the measured TTFT/TPOT/throughput trade-offs?
5. Does RL change response length enough to affect latency materially?
6. What is the practical serving footprint on the 48 GB GPU?

## Interview-story deliverable

Produce a concise deployment story focused on turning the post-training artifact into a real API, validating consistency, and measuring latency/throughput correctly rather than presenting deployment as a one-line `vllm serve` task.
