# DeepSeek Flash external reference probe

Run: `s5_external_deepseek_flash_20260910T065617`, class `EVALUATION`.
This is the external reference probe authorized while Stage 4 continues.
Stage 5 remains `NOT_STARTED`; the internal SFT / Vanilla / Dynamic final-test
comparison and open-answer judgment have not been completed.

## Frozen protocol

- Provider: official DeepSeek API, `https://api.deepseek.com/chat/completions`.
- Exact requested and returned identifier: `deepseek-flash`.
- The retained provider documentation identifies it as `DeepSeek-V4.1-Flash`.
  This is a documentation snapshot, not a verifiable model-weight revision.
- Thinking enabled, reasoning effort `high`, `max_tokens=8192`, concurrency 12,
  non-streaming. Temperature omitted because the provider documents that it is
  ignored in thinking mode. The output cap includes reasoning tokens.
- `tool_choice=none`; no tools, search or retrieval enabled. Only frozen
  system/user messages are sent; gold answers and reference responses stay local.
- One returned response per question. Only transport/rate-limit errors can retry
  (maximum three attempts). Valid wrong, truncated or unparseable responses are
  retained and never resampled. The conservative reservation ceiling is USD 15.
- Dataset freeze: `s5_data_20260910T064143_v4`; request-file SHA256
  `a8c0420be739ea38e2eea4bca46073fcbb6ad15fb3130c48c5a81a71682d4325`.
  All 1,100 IDs and prompts were frozen before the first candidate response.
- Exam scoring uses the unchanged frozen parser and canonical option-set exact
  match. Invalid outputs remain in the denominator. Hidden reasoning is not scored.

Official documentation snapshots are retained with hashes in the run manifest:
[chat completions](https://api-docs.deepseek.com/api/create-chat-completion) and
[pricing](https://api-docs.deepseek.com/quick_start/pricing).

## Results

All **1,100 / 1,100** responses completed and passed request/raw-response identity
checks. Execution ran on 2026-09-10 from 14:56:17 to 15:10:14 Asia/Shanghai
(836.77 seconds). There were exactly 1,100 attempts, no retries, no transport
failures, no unknown outcomes and no missing usage records.

| Track | Correct / planned | Accuracy | Parsed | Output-cap / empty visible answer |
|---|---:|---:|---:|---:|
| CMExam frozen probe | 458 / 512 | 89.45% | 505 / 512 | 7 / 7 |
| CMB-Exam frozen probe | 250 / 280 | 89.29% | 277 / 280 | 2 / 2 |

Descriptive Wilson 95% intervals are 86.49–91.83% for CMExam and 85.12–92.39%
for CMB. They describe these fixed stratified probes and are not weighted
population estimates or paired comparisons against another model.

| Open-ended track | Returned / planned | Nonempty visible answers | Output-cap responses | Quality judgment |
|---|---:|---:|---:|---|
| CMB-Clin, 74 cases | 208 / 208 | 207 | 1 | Pending independent judge and human audit |
| Held-out medical QA | 100 / 100 | 99 | 1 | Pending independent judge and human audit |

Across all tracks, 11 responses (1.00%) reached the output cap and returned empty
visible answers. All remain in their respective denominators and review queues.

The [verified summary](../../experiments/stage5/external_baselines/s5_external_deepseek_flash_20260910T065617/summary.json)
contains per-category and difficulty counts. Full per-question predictions and
raw responses are referenced by hashes from the artifact index.

## Tokens and cost

| Returned usage | Tokens |
|---|---:|
| Prompt total | 220,566 |
| Prompt cache hit / miss | 29,312 / 191,254 |
| Completion total, including reasoning | 1,505,137 |
| Reasoning, already included in completion | 1,325,805 |

All requests began during the documented peak-price window. Applying the
retained USD-per-million prices (cache hit 0.006, miss 0.3, output 1.2) yields
**USD 1.863716472** for the 1,100-question probe. A separately retained connectivity
control used 49 prompt and 28 completion tokens (USD 0.0000483 by the same pricing);
the combined calculated amount is USD 1.863764772. These are usage-based
calculations, not account invoices. Reasoning is not counted a second time.
The pre-request conservative reservation total was USD 11.1521202; this was a
spending guard, not actual cost. Mean request latency was 8.666 seconds with 12
concurrent client requests; this is not comparable to local single-GPU serving.

These are separate stratified-probe results, not full CMExam 6,811-question or
CMB 2,000-question scores. The CMExam probe uses clean scorable source records;
the two missing-option anomalies in the original source are outside this probe.
The CMB probe has exactly ten questions per each of 28 categories.
Later project-model comparisons must use the same IDs and parser.

## Observations and cases

Nine exam responses returned `finish_reason=length` with no visible answer.
The raw responses and billed reasoning tokens are retained. This measures the
model under the frozen 8,192-token budget, not its unrestricted reasoning ceiling.
No longer-cap rerun was folded into these scores.

The other unparseable response is request `0756`, source ID
`cmb_test:935fbc09edf1303d89872b21265ff597f426ac0d:10778`.
Its visible prose contains a greater-than comparison symbol before a final-answer
marker. The existing conservative parser rejects `<` or `>` outside recognized
control tags with `unknown_or_malformed_tags`. This is a protocol/parser failure;
the response remains incorrect under the frozen rule. It does not establish a
medical-knowledge error. No parser change was made after seeing the answer.

The 208 clinical and 100 held-out QA responses require independent rubric judging
and human audit before any medical-quality or safety conclusion. DeepSeek must
not judge itself. A correct exam choice does not establish clinical usefulness.

The [16 retained cases](../../experiments/stage5/external_baselines/s5_external_deepseek_flash_20260910T065617/cases.json)
include all 11 capped answers, the one other parser failure, and the first
canonical correct/wrong example from each exam track. They are deterministic
artifact selections, not expert medical reviews.

## Reproducibility and limits

The API model is an external contextual reference: model scale, pretraining data,
reasoning compute and serving infrastructure are not controlled against our 8B
model. No final test result selects a training checkpoint or changes Stage 4.

The run was launched from commit `4b736e5e798ccc259d0aeb9003fed73a7e5134b4`
with new runner source archived and hashed before launch. Six transport/protocol
tests plus ten evaluation-data tests passed; the JUnit receipt is retained.
The client uses Python 3.12.7 and stdlib HTTPS, without local GPU inference.

The [final audit](../../experiments/stage5/external_baselines/s5_external_deepseek_flash_20260910T065617/final_audit.json)
independently summed usage from all raw responses and reproduced the cost. All
121 source files hashed in the Stage 4 formal manifest still match their frozen
hashes. The dedicated credential copy was removed from tmpfs at worker exit;
the artifact credential-pattern scan passed.

Raw requests, attempt reservations, provider responses, visible/hidden response
fields, usage, IDs, timestamps and logs are retained under:

`/data/WSH/medical-post-train-artifacts/runs/s5_external_deepseek_flash_20260910T065617`.

Git keeps the compact evidence index under
[`experiments/stage5/external_baselines/s5_external_deepseek_flash_20260910T065617`](../../experiments/stage5/external_baselines/s5_external_deepseek_flash_20260910T065617).
The API credential is never part of this index, request payloads, logs or source.
The [artifact manifest](../../experiments/stage5/external_baselines/s5_external_deepseek_flash_20260910T065617/artifacts_manifest.json)
also pins the complete local evidence ZIP (8,503,220 bytes) and per-file checksums.
