# Stage 5 evaluation-set preparation v1

2026-09-10. Authorized scope: prepare and freeze complete evaluation sets and
API-ready inputs. Paid API calls and project-checkpoint test generation are not
part of this preparation run. Stage 5 remains NOT_STARTED.

## Inputs and method

Pin the existing Stage 1 source snapshots and sealed lexical duplicate clusters,
the 20,000 training IDs, 1,000 Stage 1 validation IDs, the complete 15,000 RL pool,
and Stage 4 monitor/selection IDs. Acquire official CMB answer keys and CMB-Clin
from Git revision `6c8ece46097dae736c6805dd3b831e1a38c08971`, retaining downloaded
hashes and the clinical ZIP member provenance. Match all 11,200 answer IDs and
category metadata. Never infer gold answers using a model.

Reuse Stage 1 normalization and DuplicateGraph thresholds: candidate char-3
Jaccard .65, confirmation char-5 Jaccard .85 or SequenceMatcher .90. Retain the
sealed historical cluster memberships; extend/check the graph with the clinical
questions. Keep the clinical case description with each question for lexical
matching and inference. Check the clinical case description separately so that
a shared case cannot hide behind different follow-up questions. Clinical questions
are independent case-context questions: no previous reference answers enter the
candidate prompt. Report 74 source cases / 208 source questions separately and
cluster later uncertainty estimates by case.

Preserve the entire official CMExam test source (6,811 records) and additionally
freeze scorable and decontaminated subsets with explicit membership. Priority
for cross-track duplicate exclusion is CMExam, then CMB-Exam, then CMB-Clin, then
retention QA. A duplicate within a clinical case is grouped by case; it is not
treated as an independent case. Preserve every rejection and its matched cluster.

Build CMB 2,000 with equal-category quotas and seeded remainder assignment; probe
280 has 10 per category. Build CMExam probe512 from clean scorable records with
deterministic balanced difficulty/category strata, retaining realized quotas.
Build retention200 (100/source) and probe100 (50/source) from previously governed
clean candidates, excluding both SFT train and SFT validation, all benchmark
clusters and the selected clinical cases. No model output informs selection.

Safety tagging uses source questions/context only, a frozen rule inventory and
matched evidence spans. These are source-derived screening tags, not model safety
judgments. Manually inspect proposed tags before freezing; retain limitations and
never generate synthetic fillers to reach50.

The source-only audit of preparation v2 found false risk tags from negated
histories (for example denied drug allergy/chest pain) and an insulin-like
biomarker mention. Before any model output, v3 excludes clause-local negations,
drug-allergy-history-only and insulin-like matches. A weak uncertainty phrase
alone does not enter the high-risk slice. Both earlier datasets/results remain
retained. Clinical case48 requests CT interpretation but includes a written CT
report in the supplied description; it is evaluated as text-based clinical QA,
not as image-reading capability. No image or reference answer is invented.

The v3 source inspection also found exam-option blocks and unavailable visual
references among unused Medical-o1/Huatuo candidates. Before freezing or any API
output, v4 applies a shared source-only retention eligibility filter: exclude
explicit multiple-option formatting and references to unavailable figures,
chemical structure images, or uploaded attachments. Refill the same100/source
quota deterministically from the remaining governed population. References and
candidate response quality never drive this filter. Pregnancy similes are also
excluded from positive pregnancy-risk tags. v2/v3 preliminary artifacts remain
retained; they are superseded preparations, not final frozen evaluation sets.

## Source discrepancies / decisions

The pinned CMExam source has two missing-A-option records: source row2866 has gold
A with only B-E options; row5331 has gold BCE with only B-E options. Preserve both
verbatim in the official6,811 manifest, mark them unscorable under the existing
strict schema, and exclude them from clean/API probes. Do not invent missing
options. The scorable subset must be explicitly named; it is not a claim that
6,811 clean questions were evaluated. Final Stage5 reporting must disclose the
official/source, scorable and decontaminated denominators and cannot silently
substitute a smaller set for the original full-source protocol.

The official answer key labels11 original C-type questions as single-choice.
Their IDs and exam type/class/subject align; canonical answers are valid. Preserve
both question-type labels and record each mismatch. Keep the source question text
and options unchanged; do not derive question type from gold answer cardinality.

The CMB question source also has42 six-option A-F questions. The first preparation
attempt retained a schema-assertion failure because it assumed the CMExam A-E
limit. Evaluation accepts each source item's actual contiguous option alphabet
and passes it explicitly to the unchanged canonical parser; no options are
discarded and no training parser/config is modified. This is source adaptation,
not a reason to exclude valid six-option questions.

The old contracts/stage_budgets.json is hash-bound to earlier stages. Add an
evaluation-only versioned contract supplement rather than rewriting it. This
preserves Stage4's frozen contract and adds new Stage5 data/secondary-review gates.

## Outputs, cost and checks

Use a unique PREPARATION run under the bulk artifact root, with command, source
archive, source/config/environment hashes, logs, exclusions, manifest JSONL,
prompt-only API requests, summary and verification receipt. Git retains compact
manifests, protocols, provenance, tests and the preparation report. Every split
must be reproducible from its seed and pinned inputs; freeze only after count,
subset, unique-ID/cluster, gold-join and source-to-request isolation checks pass.

This is CPU/disk work. Historical full-source governance took approximately118s;
allow several minutes for graph work and independent verification, with several GB
of RAM. No GPU allocation or Stage4 restart is needed. Model API costs are zero
for preparation. Actual API pricing/model choice remains for the next task.
