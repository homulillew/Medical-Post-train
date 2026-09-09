"""Seal all completed Stage 2 run evidence without replacing earlier snapshots."""
from pathlib import Path
from medical_posttrain.evidence import now, sha256, write_json
from medical_posttrain.evidence.stage2 import INDEX, read, record, selected_path

formal = selected_path('formal')
assert read(formal/'summary.json')['completed_responses'] == 4000
assert read(formal/'supervisor/status.json')['status'] == 'RUNTIME_CHECKS_COMPLETE'
assert (INDEX/'manual_review.json').exists()
assert Path('docs/stage_reports/02_reward_rollout.md').exists()
assert not (INDEX/'final_artifacts.json').exists(), 'Preserve previous seals'
inventory = []
files = {}
snapshot_differences = []
for index in sorted(INDEX.glob('s2_*')):
    manifest = read(index/'manifest.json')
    root = Path(manifest['artifact_root'])
    status = read(root/'status.json')
    assert status['status'] in ('PASS', 'FAILED', 'INVALID')
    for path in sorted(root.rglob('*')):
        if path.is_file():
            files[str(path)] = record(path)
    snapshot = index/'artifacts_manifest.json'
    if snapshot.exists():
        for earlier in read(snapshot):
            current = files[earlier['path']]
            if current['sha256'] != earlier['sha256']:
                path = Path(earlier['path'])
                # Worker completion snapshots can precede monitor shutdown and
                # buffered log flushes. Raw/config/model evidence cannot change.
                assert path.name in ('heartbeat.json', 'stdout.log', 'stderr.log',
                                     'supervisor.stdout.log', 'supervisor.stderr.log'), path
                snapshot_differences.append(dict(run_id=root.name, earlier=earlier,
                    sealed=current, reason='Completion snapshot preceded monitor shutdown or log flush; original snapshot retained'))
    inventory.append(dict(run_id=root.name, purpose=manifest['purpose'],
        run_class=manifest['run_class'], status=status['status'], artifact_root=str(root),
        config=record(root/'config.json'), manifest=record(root/'manifest.json'),
        source_archive=record(root/'source.zip'), status_receipt=record(root/'status.json')))
write_json(INDEX/'run_inventory.json', inventory)
write_json(INDEX/'artifact_snapshot_audit.json', dict(timestamp=now(),
    final_authority='final_artifacts.json', differences=snapshot_differences,
    note='Original worker-completion manifests remain historical snapshots. Final seal also covers later manual packets and supervisor completion.'))
failed = [r for r in inventory if r['status']=='FAILED']
write_json(INDEX/'system_cases.json', [dict(case_id='S2-SYSTEM-TOKENIZER-API',stage=2,
    category='SYSTEM_CASE', run_id=r['run_id'], subtype='semantic_tokenizer_api',
    observation='Pinned Transformers BertTokenizer lacks build_inputs_with_special_tokens; diagnostic failed at warmup before model comparison scores.',
    root_cause='Legacy tokenizer helper was not supported by the installed tokenizer backend.',
    resolution='New diagnostic run uses native backend post_process; normalized short-text vectors checked against SentenceTransformer.encode.',
    reproduction='Run the archived source.zip with its retained config and attempt command in the recorded environment.',
    evidence=r['status_receipt'], original_source=r['source_archive'],
    successful_followup=selected_path('semantic').name) for r in failed])
write_json(INDEX/'final_artifacts.json', list(files.values()))
print('Sealed', len(files), 'files across', len(inventory), 'runs; retained failures', len(failed))
