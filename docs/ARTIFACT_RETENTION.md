# Artifact Retention Policy

This project must preserve enough evidence to reproduce conclusions without forcing large model/data artifacts into Git.

## 1. Principle

**Git is the experiment index and audit trail. Local/external storage is the bulk artifact store.**

Never depend only on conversational memory or a temporary dashboard for a headline result.

## 2. Commit to Git

Git should normally retain:

- source code and configs;
- Markdown documentation;
- environment/requirements lock files;
- experiment manifests;
- summary JSON/CSV files;
- compact metrics needed for plots;
- generated plots;
- selected case-study JSONL/Markdown;
- decision/observation records;
- stage/final reports;
- small test fixtures and parser/reward unit-test examples.

## 3. Do not commit by default

Large or frequently changing artifacts should normally stay outside Git:

- downloaded dataset caches;
- full model/base weights;
- LoRA checkpoints if repository-size policy makes them impractical;
- optimizer-state checkpoints;
- full rollout corpora;
- huge raw stdout logs;
- temporary vLLM caches;
- framework caches;
- large benchmark request/response dumps.

Use `.gitignore` when implementation begins.

## 4. Required artifact manifest

For every important non-Git artifact, retain a manifest entry such as:

```json
{
  "artifact_id": "s4_dynamic_best_adapter",
  "run_id": "s4_gspo_dynamic_001",
  "kind": "lora_adapter",
  "path": "/durable/path/...",
  "size_bytes": 123,
  "sha256": "...",
  "created_at": "...",
  "git_commit": "...",
  "description": "..."
}
```

Where practical, include hashes for datasets, final adapters, important checkpoints, and analysis inputs.

## 5. Durability tiers

### Tier A: critical

Must be retained until the project is finished and resume/verification no longer requires them:

- final Stage 1 SFT adapter;
- primary Vanilla GSPO checkpoints/final adapter;
- primary Dynamic GSPO checkpoints/final adapter;
- optimizer/trainer state needed for active formal-run resume;
- full primary formal-run structured metrics;
- final evaluation outputs;
- dataset split manifests and dedup records.

### Tier B: valuable

Retain if storage allows:

- pilot checkpoints;
- profiling/full rollout outputs;
- exploratory experiment outputs;
- raw benchmark request/response sets;
- detailed system profiling.

### Tier C: disposable cache

May be regenerated and cleaned after verification:

- package/model download cache when durable source/revision is known;
- temporary preprocessing shards;
- vLLM temporary/cache files;
- intermediate files already summarized and not needed for reanalysis.

## 6. Logs

Prefer structured metrics plus complete enough raw logs to debug failures. Huge text logs may be compressed or stored externally, but do not discard the only evidence of a failure before the root cause is understood.

## 7. Weights and checkpoints

For each final/important LoRA adapter, retain:

- base model/revision reference;
- PEFT configuration;
- adapter files;
- hash/size;
- source run ID;
- validation metric used for selection.

For active resumable formal RL runs, retain sufficient optimizer/scheduler/trainer state to continue counters correctly.

## 8. Dataset provenance

Do not commit large upstream datasets unless their license/size make that appropriate. Instead retain:

- upstream dataset/revision identifier;
- selection/filtering script version;
- random seed;
- selected sample IDs where available;
- counts before/after filters;
- split hashes/manifests;
- dedup report;
- license/source notes.

## 9. Case-study retention

Compact representative cases should be committed to Git when licensing/privacy permits. Full raw response corpora may remain in artifact storage with manifests.

## 10. Cleanup rule

Before deleting a large artifact, verify that:

1. it is not required to resume an active formal run;
2. it is not the only source of a published metric/plot;
3. a manifest and sufficient summary remain;
4. it can be regenerated from retained inputs if needed.

Never clean up simply because an experiment produced an unfavorable result.
