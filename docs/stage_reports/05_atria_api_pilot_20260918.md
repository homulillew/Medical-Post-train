# Stage5 Atria API integration pilot — 2026-09-18

The owner-authorized 20-entry test has stopped. It produced **16 valid judgments
and 4 entries without a valid score**. This is a transport/schema pilot, not a
completed Stage5 evaluation or clinical validation.

The frozen prefix covers 6 clinical questions from 3 cases, with 17 base
comparisons and 3 position flips. Valid results comprise 14 base judgments and
2 flips. Only one flip has a valid matching base judgment: the original verdict
was `tie`, while the flipped verdict was `B win`. One pair cannot establish a
population consistency rate, but this disagreement must remain visible.

| Request configuration | Requests | Valid judgments | Reported tokens |
|---|---:|---:|---:|
| Native JSON / temperature 0 | 1 | 0 | 2,398 |
| Plain text / temperature 0 | 2 | 1 | 6,750 |
| Plain text / temperature omitted | 18 | 15 | 52,809 |

All 21 scoring requests returned HTTP 200. Five attempts were invalid: malformed
JSON, repetitive `/0` output, two non-JSON outputs, and a JSON object that used
`safety_safety` instead of the required `safety_escalation` field. The first entry
had one explicitly recorded technical retry; no accepted judgment was replaced.
No missing score was inferred from text, repaired, or assigned zero.

The final configuration produced 15/18 valid outputs. This does not establish a
causal explanation for failures: inputs differ, and the preview service may vary.
API success status alone is insufficient to accept a judgment. No automatic
full-set run or additional paid retry is scheduled.

Including the initial `hi` probe, there were **22 API requests**, with **39,755 input + 22,429 output = 62,184 tokens** reported. These counts include invalid attempts. Currency cost is unknown; no pricing was assumed.

Scores below use only the final configuration's valid base comparisons. They
are repeated answer appearances, not independent questions; coverage differs by
model because four entries are missing. Do not infer a ranking or compare these
means to a full-set result. The temperature-zero judgment is retained separately
in the machine-readable analysis.

| Candidate | Appearances | Factual | Relevance | Safety | Uncertainty | Clarity |
|---|---:|---:|---:|---:|---:|---:|
| sft | 8 | 2.250 | 2.500 | 3.250 | 3.000 | 3.000 |
| selected_vanilla | 10 | 2.300 | 2.400 | 3.200 | 3.000 | 3.100 |
| selected_dynamic | 8 | 2.250 | 2.250 | 3.250 | 3.000 | 3.125 |

Each rubric dimension is 0–4. The valid judgments contained no critical safety
flags; this is not a finding that the answers are safe. Real human reviews remain
zero. Submission validation found 16 schema-valid, attributable judgments with
no validation errors and correctly returned `INCOMPLETE` for the 1347-entry gate.
Stage5 stays `FULL_RUNNING`; Stage6 remains `NOT_STARTED`.

Before a larger run, address malformed output reliability and independently
review position consistency and medical judgments. The current owner budget
only covers this pilot; full judging and human audit remain outstanding.

## Evidence

- [Compatibility decision](../decisions/STAGE5_ATRIA_API_PILOT.md)
- [Accounting and verification pointers](../../experiments/stage5/atria_judge/pilot_20260918_summary/accounting.json)
- [Scores and missing entries](../../experiments/stage5/atria_judge/pilot_20260918_summary/pilot_analysis.json)
- [Per-request diagnostics](../../experiments/stage5/atria_judge/pilot_20260918_summary/request_diagnostics.json)
- [Artifact hashes](../../experiments/stage5/atria_judge/pilot_20260918_summary/artifacts_manifest.json)

Raw requests/responses, source snapshots and normalized judgments remain under
`/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1/atria_judge`.
Authentication is never written to those artifacts. Tools were explicitly
disabled in scoring requests; no tool-call output was returned. The client used
the local key; no internet search was performed. Twelve targeted tests passed.
Earlier v3 handoff and empty review templates remain historical frozen snapshots.
