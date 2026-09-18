# Full Atria blind evaluation — 2026-09-18

- Stage: 5
- Status: AUTHORIZED / EXECUTION RUNNING
- Run: `s5_atria_full_20260918T095056Z`
- Contract change: NO

The owner requested the complete evaluation in the background after the
three-request transport diagnostic. This supersedes the earlier pilot-only paid
request limit for this full blind-judging run.

Evaluate all 1347 entries in the frozen public packet (1224 base comparisons and
123 position flips), derived from the existing 1224 candidate answers over 408
questions. Use a new run ID and one uniform judge configuration: Atria-Dawn-Preview,
explicit system/user messages, frozen rubric, 4096 output-token cap, no tools,
temperature omitted and parser `atria-visible-v2`. Keep original pilot and
diagnostic results separate; do not regenerate candidate answers or choose new
checkpoints from test results.

Run serially to limit rate/concurrency pressure on the preview API. Store every
request, raw byte/text response, normalized judgment or explicit invalid-output
record, actual token usage and timestamps. Monetary pricing is unknown. The
initial ceiling is one request per entry (1347 calls); no unparseable output is
resampled. A transport/HTTP error or ten consecutive invalid outputs pauses the
run for investigation. There are no automatic paid retries.

On resume, verify saved artifacts and skip valid/invalid terminal records. If a
raw response was saved before interruption, parse it offline. An in-flight request
without a saved response is uncertain and must not be silently resent.

Launch as a detached process with stdin disconnected and separate stdout/stderr
logs. Retain PID, command, source revision/hashes, configuration and artifact
root. Publish progress after each response, including valid count, invalid count,
attempted count and usage. A launch is not completion.

After all entries, export judgments, missing-coverage verification and the updated
human-audit packet including all new critical flags. Compute the predefined
position consistency and case-cluster preference statistics only when all judge
entries are valid. Missing judgments remain missing. Real human review, safety
adjudication and full Stage5 acceptance remain separate gates; Stage6 must not be
started merely because this API run finishes.
