# Stage 5 evaluation datasets — frozen preparation v1

Prepared and verified on2026-09-10. Run: `s5_data_20260910T064143_v4`.
Dataset status: **DATASETS_FROZEN**. Stage5 remains **NOT_STARTED**.
No project checkpoint test generation or third-party API call was performed.

## Frozen collections

| Collection | Full collection | First external API probe |
| --- | ---: | ---: |
| CMExam official source |6811 retained records | — |
| CMExam scorable / clean unique |6809 /6732 |512 clean questions |
| CMB-Exam |2000,28 categories |280,10/category |
| CMB-Clin |74 cases,208 questions |208 questions |
| Medical QA retention |200,100/source |100,50/source |
| Source-risk safety slice |111 questions from clinical/retention | Reuse existing open-QA responses |

The first external bundle contains **1100 unique requests**. The full executable
project bundle contains9217 requests per checkpoint, or27651 for three models.
The two malformed CMExam source records are separately retained; these request
counts must not be described as6811 valid CMExam model responses per checkpoint.

## Inputs for the next API task

The machine-readable entry point is
[`experiments/stage5/dataset_freeze_v1.json`](../../experiments/stage5/dataset_freeze_v1.json).
It pins the source/manifest/protocol hashes and the following1100-request file:

```text
/data/WSH/medical-post-train-artifacts/data/stage5/s5_data_20260910T064143_v4/requests/external_probe_all.jsonl
```

Each line contains `id`, `task`, and `messages`. Send only `messages` plus the
chosen API's explicitly frozen model/decoding parameters. Retain `id` locally to
join outputs to evaluation references. Do not send `sets/*.jsonl` to the candidate
API: those files include standard answers or reference responses for later scoring.
Track-specific input files are in the same `requests/` directory.

The approximately16.8MB compressed data bundle is:

```text
/data/WSH/medical-post-train-artifacts/data/stage5/s5_data_20260910T064143_v4/evaluation_dataset_bundle.tar.gz
```

It contains sets, request files, manifests, protocols, exclusions and verification.
Git retains compact manifests with stable IDs; bulk files retain complete content.

Before paid execution, freeze the exact provider/model identifier, API mode,
accepted decoding parameters, output/reasoning budgets, retry limits, and spend
envelope in a new external-run manifest. The current dataset protocol freezes
the questions and prompts, not an unspecified provider's pricing or tokenizer.
No API credential has been inspected or used during preparation.

## Data governance and source findings

All sources retain revision, file hash and stable row IDs. Existing Stage1 lexical
duplicate clusters were extended with clinical case/context questions using the
same thresholds. Exclusions cover20000 SFT train items,1000 SFT validation items,
the15000 RL pool, monitor512, selection1024, and cross-track duplicate clusters.
The final graph contains241996 text records. This is lexical decontamination and
does not prove absence of semantic paraphrases or pretraining exposure.

CMExam retains the original6811 records. Source rows2866 and5331 have no A option;
their original gold fields are A and BCE respectively. No missing option was
invented. The scorable set has6809 rows; removing77 duplicate-cluster members
gives6732 clean unique questions. The512 probe is drawn from that clean set with
seeded difficulty/clinical-department balancing. Full/scorable/clean denominators
are explicitly distinct. The original full-source Stage5 acceptance requirement
cannot be satisfied by silently renaming the clean subset.

CMB official gold keys were pinned to Git revision
`6c8ece46097dae736c6805dd3b831e1a38c08971`. All11200 IDs match the pinned question
file, and exam type/class/subject metadata align. Eleven original C-type entries
are labeled single-choice by the answer-key metadata; both labels are retained.
Forty-two valid source questions have A-F options. The unchanged canonical parser
receives each question's actual legal alphabet; no F option is discarded.
Before selection,132 CMExam-overlap and153 within-CMB duplicate members were
excluded. The2000-item set allocates71 questions to16 categories and72 to12
seed-selected categories. The280 probe has exactly10 per category.

CMB-Clin retains all74 cases and208 questions. Inputs are the source case
description plus the current question, with no prior reference answers. Repeated
questions within a case remain correlated; use case-level clustered statistics.
This is text-based QA. A CT-interpretation question includes a written CT report;
the evaluation does not test image interpretation or invent unavailable images.

Retention selection excludes both SFT train and SFT validation, all benchmark
clusters, explicit choice-option blocks, and unavailable figure/attachment
references. The source-only filters excluded1392 Medical-o1 choice-format records,
12 Medical-o1 visual-reference records, and1123 Huatuo attachment/visual-reference
records from the unused candidate population. The final200 retain100 records
from each source and measure in-domain open-QA behavior, not external generalization.

Safety tags are source-derived screening metadata, not candidate-model failure
labels or clinical certification. Negated histories, insulin-like biomarker
mentions and pregnancy similes were excluded from positive tags. A weak uncertainty
phrase alone does not qualify a record for the111-question risk slice. The frozen
rubric requires independent judging and real human review after outputs exist;
neither has been represented as completed by this preparation task.

## Verification and retained history

Ten unit tests passed. The final real-data verifier passed source-to-record joins,
gold matching, ID/cluster exclusions, quotas, subset inclusion, reference-free
request construction and reverse-order deterministic selection replay. The main
preparation took68.45 seconds; verification took7.33 seconds. The unchanged exam
and clinical collections were also byte-identical across v3/v4 rebuilds. All49
frozen Stage4 execution files remained unchanged.

The first run's A-E-only CMB schema assumption failed and is retained. v2/v3
preparations remain retained with their verification results and source-inspection
limitations; they were superseded before any API/model outputs were observed.
Only v4 is the frozen evaluation release. See
[`preparation_history.json`](../../experiments/stage5/preparation_history.json).

The original `contracts/stage_budgets.json` remains unchanged for prior-stage hash
continuity. New Stage5 requirements are recorded in
[`stage5_evaluation_supplement_v1.json`](../../contracts/stage5_evaluation_supplement_v1.json).
This preparation does not select checkpoints or mark Stage5 evaluation complete.

## Reproduction

```bash
python scripts/fetch_stage5_sources.py
.venv-analysis/bin/python scripts/prepare_stage5_datasets.py --run-id <new-unique-run-id>
.venv-analysis/bin/python scripts/verify_stage5_datasets.py --run /data/WSH/medical-post-train-artifacts/data/stage5/<new-unique-run-id>
```

Reproduction consumes the pinned Stage1 governance and Stage2 pool artifacts.
Use a new run ID; never overwrite the frozen release. The published manifests
and request hashes remain the authoritative inputs for the upcoming external API
evaluation.
