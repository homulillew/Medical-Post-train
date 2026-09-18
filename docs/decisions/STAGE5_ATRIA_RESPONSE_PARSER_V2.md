# Atria response parser v2 and bounded transport diagnostic

- Date: 2026-09-18
- Stage: 5
- Status: ACCEPTED for client parsing; upstream cause UNRESOLVED
- Contract change: NO
- Diagnostic run: `s5_atria_parser_v2_20260918T093326Z`

The owner requested investigation and repair of apparently garbled API output,
then explicitly authorized at most three additional validation requests.

## Evidence

The original 21 scoring requests had five rejected attempts. Inspection separates
them into one malformed JSON string, one repetitive `/0` string, one nonsensical
text fragment, one response containing only a reasoning item, and one complete
JSON judgment with `safety_safety` in place of `safety_escalation` on both sides.
All outer response envelopes were valid JSON. The client used strict UTF-8
decoding; retained visible strings contain no replacement characters or unpaired
surrogates. There is no evidence of local UTF-8/GBK confusion. The repeated ASCII
characters and missing assistant message are already present in the API response.
Old artifacts retain decoded bodies, not original wire-byte snapshots; that
limitation must not be concealed.

The initial report grouped failures too broadly as non-JSON output. One response
has no visible answer at all; hidden reasoning is never promoted into a score.

Reported input-cache hits occur on four original requests: three invalid and one
valid. Two other invalid attempts report no cache hit. This small, confounded
sample cannot identify caching as a cause, and no undocumented cache-disable
parameter is introduced. Provider generation, serialization, routing and
message conversion remain unverified alternatives.

## Client changes

1. Parse only completed assistant output text. Report separate failure codes for
   missing visible output, repetitive non-JSON output, malformed JSON, schema
   mismatch, UTF-8 errors, refusal/tool output and incomplete responses.
2. Reject duplicate JSON keys and non-finite numbers; require exact rubric keys,
   integer scores, and attributable safety evidence. Validation must also work
   when Python assertions are disabled.
3. Permit one documented key alias, `safety_safety` → `safety_escalation`, only
   when the canonical key is absent and all other dimensions are exact. Preserve
   every original numerical value, preference and rationale; record the mapping
   in each compatible judgment. Reject conflicting or additional keys and never
   guess missing scores. This is a retrospective parser correction, not a claim
   that the model complied with the original schema.
4. Preserve new response bytes as Base64 before strict decoding, along with an
   allowlist of response headers. Credentials are redacted in both byte and text
   representations. No lossy replacement decoding is used.
5. Encode requests explicitly as UTF-8 and use system/user message objects with
   `input_text` content. Prompt text, rubric and anonymous candidate answers stay
   identical; tools remain disabled and temperature/native JSON mode stay omitted.
   Explicit roles are a compatibility hypothesis, not an established root-cause fix.
6. Progress counters count saved judgments rather than the current schedule index.

## Replay and live validation

Replay all original responses without any network request. Previously valid
judgments must compare exactly equal to their saved records. New normalized
results and their validation receipt get a separate immutable artifact path;
original 16/20 results and five failure attempts remain unchanged.

The live diagnostic uses only frozen entries 1, 4 and 5 (zero-based), previously
repetitive, nonsensical and missing visible output respectively. It issues at
most one request per entry and stops on transport/HTTP failure. There are no
automatic retries. New judgments are diagnostic only and must not replace the
original evaluation answers or silently increase the evaluation denominator.
Three successes, if observed, would show compatibility on these probes, not
prove a stable failure rate or isolate the cause of earlier errors.

No training, checkpoint selection or full-set evaluation changes. Real human
audit and Stage5 completion remain outstanding. See the associated run summary
and parser repair report for measured results.
