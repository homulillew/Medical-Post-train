# Stage Report — Stage `<N>` `<name>`

> This report is written **after** the stage's mandatory evidence exists. Do not fill it with planned or invented results.

## 1. Stage objective

What problem did this stage need to solve, and why was it necessary for the full project?

## 2. Initial design

- planned method:
- mandatory budget:
- initial assumptions:
- expected risks:

## 3. What was actually implemented

Describe the final implementation, including material differences from the initial design and links to decision records.

## 4. Experiments executed

| Run ID | Class | Purpose | Budget | Status | Key artifact |
|---|---|---|---|---|---|
| | | | | | |

Include failed/invalid runs that materially informed the final implementation.

## 5. Quantitative results

Present the complete stage-relevant metrics with exact run/artifact references. Do not show only the best run.

## 6. Important observations

Separate direct observations from interpretations.

### Observation 1

- Evidence:
- Hypothesis:
- Alternative explanations:
- Conclusion/current confidence:

Repeat as needed.

## 7. Bad / good / boundary cases

Select 2–5 representative case IDs. For each:

- what happened;
- why it matters;
- which aggregate metric/decision it helps explain;
- limitations of the case.

Include at least one meaningful failure/regression when one exists.

## 8. Engineering challenges

Describe the most important real implementation/system issues:

- symptom;
- diagnosis;
- attempted solutions;
- final resolution;
- trade-off/cost.

## 9. Decisions and trade-offs

List material decision records and why the final choices were made.

## 10. Acceptance evidence

Go through every acceptance criterion from the stage document and link evidence. Mark each `PASS`, `FAIL`, or `N/A` with explanation.

A stage should not be called DONE with unresolved mandatory `FAIL` items.

## 11. What this stage established

State defensible conclusions only.

## 12. What this stage did not establish

List limitations/confounders/unanswered questions.

## 13. Handoff to next stage

- exact verified artifact(s) to consume;
- fixed configs/assumptions inherited;
- unresolved risks/observations the next stage should monitor.

---

# Interview Story

## 30-second version

Problem -> action -> measured outcome/lesson.

## 2-minute version

Cover context, technical choice, implementation difficulty, experiment evidence, and result.

## Deep-dive topics

Prepare concise answers for:

1. Why this design instead of the obvious alternative?
2. What exactly did you implement yourself?
3. What failed and how did you debug it?
4. Which metric/case changed your mind?
5. What trade-off did the 48 GB constraint create?
6. How do you know the result is not an evaluation artifact?
7. What would you do next with more compute?

## Resume-ready claims

For each potential resume bullet/number list:

- exact wording candidate;
- supporting run ID(s);
- supporting artifact/file;
- caveat/conditions;
- whether the claim is safe to use: YES / NO / NEEDS MORE EVIDENCE.
