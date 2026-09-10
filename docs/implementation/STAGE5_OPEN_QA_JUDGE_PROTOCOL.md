# External open-QA rubric judging

Run `s5_judge_v4pro_20260910T073746` evaluates all 308 open-ended responses from
`s5_external_deepseek_flash_20260910T065617`. It is an external reference evaluation,
not the internal Stage 5 comparison. Stage 5 remains `NOT_STARTED`.

## Decision frozen before judgments

Use official `deepseek-v4-pro`, documented `DeepSeek-V4-Pro-0813`, as a different
model from the candidate `deepseek-flash`. The existing authorized DeepSeek
connection permits this separate judge without requiring another provider's
credentials. Same-provider error correlation and preference bias remain limits;
this choice does not establish that the judge is more capable than the candidate.
The retained provider documentation says V4 Pro requests will be redirected to
Flash starting September 14, 2026. This September 10 run predates that transition;
future runs must recheck routing and must never use Flash as its own judge.

The judge receives only source context, the current question, reference answer,
and final visible candidate response. All 306 nonempty requests are anonymous;
none contains a detected candidate model-name self-identification. Separate API
reasoning is never included. The frozen five-dimension 0–4 rubric is retained
verbatim in the judge system prompt. References are fallible aids, not mandatory
wording targets, and unavailable images cannot be assumed visible to the candidate.

Judge settings: thinking enabled, reasoning effort `high`, JSON object output,
16,384-token cap, concurrency 12, 300-second transport timeout. No search,
retrieval, browsing or tools; `tool_choice=none`. Temperature is omitted under
the documented thinking-mode behavior. Every judge response is retained. Only
transport failures may retry up to three times; malformed, truncated or unfavorable
judgments are not silently resampled. Invalid judgments remain missing evidence.

The two empty candidate responses receive factual/relevance/clarity scores of
zero under an explicit missing-answer rule. Safety and uncertainty scores are
null: silence is not automatically a critical harmful instruction. Report these
two missing safety/calibration assessments and all metric denominators explicitly.
No weighted overall score is manufactured.

Only one candidate is being scored, so no pairwise A/B or answer-position
win-rate claim applies. This does not replace the later matched model comparison.

## Cost and execution evidence

The prelaunch conservative first-attempt reservation is USD 23.82321084, based
on UTF-8 input-byte proxies and the maximum output allowance at peak prices.
The runner's reservation ceiling is USD 40 including bounded transport retries.
These are guards, not actual fees. Sum returned token usage with the retained
peak/off-peak price snapshot after execution and report actual calculated cost.

Before launch, 14 runner and judge-validation tests passed. Frozen run config,
source ZIP, input and response hashes, exact prompts and model identifier, raw
requests/responses, usage, latency, status and errors are retained under
`/data/WSH/medical-post-train-artifacts/runs/s5_judge_v4pro_20260910T073746`.
The Git index is
[`experiments/stage5/open_qa_judgments/s5_judge_v4pro_20260910T073746`](../../experiments/stage5/open_qa_judgments/s5_judge_v4pro_20260910T073746).

## Scoring and unfinished human gate

Report each dimension's mean, 0–4 distribution and non-null denominator separately
for 208 clinical questions / 74 cases and 100 retention questions. Use 10,000
bootstrap draws, seed 20260910, sampling clinical cases as clusters and retention
questions as items. Intervals are conditional on this judge pass; they do not
measure judge bias or clinical truth.

Every critical flag requires a quotation found in the actual response and a
specific harm rationale. Flags are potential issues from an LLM, not confirmed
clinical failures. Retain reference disputes and invalid judge outputs.

Before judgments, freeze 62 stratified random human-audit items: 42 clinical,
10 Medical-o1 and 10 Huatuo. Add all critical flags, invalid judgments and empty
answers afterward. The real human reviewer and expertise remain **PENDING**.
Neither this assistant nor the API judge counts as a human reviewer. Scores may
be reported as an LLM preliminary assessment; full clinical/human verification
must not be claimed.
