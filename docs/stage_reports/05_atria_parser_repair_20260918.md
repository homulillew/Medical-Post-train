# Atria response investigation and repair — 2026-09-18

Client repair and the bounded live diagnostic are complete. Offline replay
preserves all 16 originally valid judgments and recovers one schema-compatible
judgment through an explicit field-name alias. Three newly authorized API probes
all returned valid judgments without normalization. Upstream root cause remains
unconfirmed; three successes do not prove the service is permanently reliable.

## What looked like garbled text

The five rejected attempts in the original pilot were different failures:

| Failure | Attempts | Client treatment |
|---|---:|---|
| Malformed JSON in native JSON mode | 1 | Reject; no inferred score |
| Repeated `/0` output | 1 | Reject as degenerate non-JSON output |
| Nonsensical text fragment | 1 | Reject as non-JSON output |
| Reasoning item with no visible message | 1 | Reject as missing visible answer |
| `safety_safety` in an otherwise complete judgment | 1 | Explicit alias normalization with provenance |

The response envelopes decoded successfully as UTF-8 and parsed as JSON. The
visible strings have no replacement characters or unpaired surrogates. The
abnormal ASCII strings and missing assistant message are present in the retained
API output itself; converting UTF-8 to another encoding cannot restore scores.
We cannot distinguish a provider generation problem from a serving/conversion
bug with this client evidence alone.

There was also a local reporting problem: the generic JSON/KeyError messages
obscured the distinction between content failure, absent output and key mismatch.
The old progress counter used the schedule index as the completed count, including
skipped/failed entries. Both are corrected.

## Changes and verification

- Dedicated parser with explicit failure codes; strict completed-message checks,
  duplicate-key rejection, finite JSON and exact integer rubric validation.
- A single compatible alias `safety_safety` → `safety_escalation`, allowed only
  with the other dimensions intact and no conflicting canonical key. Values,
  preference and rationale stay unchanged; the record includes both key names.
- Explicit system/user message objects with `input_text` content and UTF-8
  serialization. The judge prompt, rubric and anonymous answers are unchanged.
  Tools remain disabled. No undocumented cache control or JSON-repair generation.
- Future responses retain Base64 bytes before strict decoding, even when decoding
  fails, plus selected response headers. Secret values are redacted. Old decoded
  artifacts cannot retroactively prove their original wire-byte representation.
- Source hashes include both runner and parser; source snapshots are retained.

Twenty-five targeted tests passed. All 21 original responses were replayed:
16 original judgments compare exactly equal, one alias-normalized judgment is
recorded separately, and four failed attempts remain rejected. Because one failed
attempt concerned an item with a later valid technical retry, unique compatibility
coverage is **17/20**, with **3 entries still missing from the original evidence**.
Schema/coverage validation correctly remains `INCOMPLETE` for the full 1347-entry
schedule. All historical artifact hashes still match.

## Live diagnostic

Run: `s5_atria_parser_v2_20260918T093326Z`.

The owner authorized at most three additional calls. Frozen indices 1, 4 and 5
were chosen for their earlier repetition, nonsense and absent visible-answer
failures. The client sent one request per entry with explicit message roles.
All three returned HTTP 200 and valid rubric JSON; none required the field alias.
The first response reported a full input-cache hit and still succeeded. This
further rules out any claim that cache hits always cause failure.

Usage: **5,488 input + 1,733 output = 7,221 reported tokens**, exactly 3 requests,
zero automatic retries. Currency cost is unknown. All requests have stopped.
These are diagnostic resamples, explicitly excluded from evaluation totals.
They show that the new request format works on the three probes. Inputs were
selected after failures and the service was queried at a later time, so this is
not a randomized estimate of improvement or proof of a specific root cause.

The original 16/20 strict pilot remains frozen; 17/20 is a separately versioned
compatibility replay. Position-preference disagreement from the original pilot
is not a parser issue and remains unresolved. Human reviews remain zero,
Stage5 is incomplete, and no full-set paid run is authorized by these probes.

## Evidence and reproduction

- [Decision](../decisions/STAGE5_ATRIA_RESPONSE_PARSER_V2.md)
- [Verification](../../experiments/stage5/atria_judge/s5_atria_parser_v2_20260918T093326Z/verification.json)
- [Replay and per-attempt failure codes](../../experiments/stage5/atria_judge/s5_atria_parser_v2_20260918T093326Z/replay.json)
- [Live results and usage](../../experiments/stage5/atria_judge/s5_atria_parser_v2_20260918T093326Z/summary.json)
- [Original pilot](05_atria_api_pilot_20260918.md)

`python scripts/diagnose_stage5_atria.py` performs an offline replay with no API
requests. `--live` adds the three diagnostic requests and must only be used under
a new explicit request budget. No retry or background job is scheduled.
