# Stage 4 closure and Stage 5 execution

Owner authorization: experiments/stage4/closure_source_snapshots/owner_instruction.txt. This is an accepted project-contract change limited to the Stage 4 documentation manual-review gate; scientific requirements and Stage 5 human/judge requirements are unchanged. Zero human reviews have been completed.

The start audit verifies remote main, frozen Stage 5 sources/data/candidates, unchanged primary execution and zero selection/final project generations. Stop on scientific hash drift or prior unexpected consumption. Retain historical failure and draft records.

1. Retain the waiver, original verifier and exact diff; prepare final deliverables. Test valid/invalid waivers and preservation of all scientific verifier functions. Run `.venv-train/bin/python scripts/verify_stage.py --stage 4 --output experiments/stage4/verification-final.json` with PYTHONPATH=src:scripts. This is the complete raw verifier; no budget or window sampling.
2. Only after FULL PASS: Stage 4 DONE, publish closure commit, revalidate Stage 5 frozen seal/data/candidate hashes, then Stage 5 SELECTION_RUNNING.
3. Use the frozen native worker for all 20 x 1024 selection outputs. Freeze complete rankings and selected checkpoints before any final test. Commit selection evidence.
4. Run all CMExam6809/CMB2000 unique checkpoint jobs before open QA. Deduplicate identical selected/final5000 adapters. Frozen clean CMExam6732/CMBmedical1929 and safety111 reuse original outputs.
5. Generate 408 open items for each of SFT/selected Vanilla/selected Dynamic. Retain visible answers and hidden think separately; build frozen blind bundle with no external judge calls.
6. Independently replay raw tokens, parser, identifiers, adapter lineage, attempts, rankings and paired statistics; retain regressions, slices, and machine-only safety/cases. Report OBJECTIVE_EVAL_PASS with OPEN_QA_JUDGE_PENDING rather than Stage 5 DONE. Stage 6 remains NOT_STARTED.

Existing Stage 5 sealed files remain unchanged. New orchestration and objective-only verification will be added outside that seal, with tests and a source manifest before generation. New orchestration enforces the owner-requested phase order; its decoding, IDs, scoring and native worker remain frozen.

Resources: the full Stage 4 verifier reads all historical CPU artifacts and may take hours; no GPU generation. Selection has 20,480 generations; exams have 26,427–44,045 depending on distinct selected endpoints; open QA adds 1,224. At prior approximately 512 responses per 11–13 minutes, generation alone is roughly 18–28 GPU hours, with uncertainty from final prompt/response lengths. Do not reduce budgets. GPU ownership must be exclusive. Preserve technical failures and resume existing outputs; never regenerate valid wrong/unparseable responses.
