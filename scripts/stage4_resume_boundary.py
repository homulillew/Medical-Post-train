"""Operational restore: hash the commit chain and load boundary, audit history later.

This is explicitly installed by stage4_resume_runtime.py or the incident
supervisor. The frozen transaction, actor, and final raw verifier stay intact.
Historical checkpoint payloads are stat-checked, NOT claimed as hash-verified.
"""
from pathlib import Path
import time

from medical_posttrain.rl.common import read, record, immutable, now
from medical_posttrain.rl.online import initial_state, verify_checkpoint


def restore_boundary(out, initial):
    out = Path(out)
    started = time.monotonic()
    state = initial_state(initial)
    checkpoint = None
    commits = []
    metadata_bytes = 0
    history_payload_bytes = 0
    gap = False
    for window in sorted((out / 'windows').glob('*')):
        if not (window / 'commit.json').exists():
            gap = True
            continue
        assert not gap, 'Committed window after an uncommitted gap'
        assert window.name == f'{len(commits):04d}', 'Non-contiguous window indices'
        commit = read(window / 'commit.json')
        assert commit['state_before'] == state, 'Broken state lineage'
        for ref in commit['artifacts']:
            assert Path(ref['path']).resolve().is_relative_to(window.resolve())
            assert record(ref['path']) == ref, ref['path']
            metadata_bytes += ref['bytes']
        checkpoint = window / 'checkpoint'
        marker_path = checkpoint / 'COMMITTED.json'
        assert record(marker_path) in commit['artifacts'], 'Unbound checkpoint marker'
        marker = read(marker_path)
        files = {ref['path']: ref for ref in marker['files']}
        assert len(files) == len(marker['files']), 'Duplicate checkpoint file'
        for ref in files.values():
            path = checkpoint / ref['path']
            assert path.resolve().is_relative_to(checkpoint.resolve())
            assert path.is_file() and path.stat().st_size == ref['bytes'], str(path)
            history_payload_bytes += ref['bytes']
        after = commit['state_after']
        assert after['policy_windows'] == state['policy_windows'] + 1
        assert after['optimizer_steps'] == state['optimizer_steps'] + 2
        assert after['training_groups'] == state['training_groups'] + 8
        assert marker['optimizer_step'] == after['optimizer_steps']
        assert files['adapter/adapter_model.safetensors']['sha256'] == after['policy_version']
        state = after
        commits.append(record(window / 'commit.json'))
    boundary_bytes = 0
    if checkpoint is not None:
        marker = verify_checkpoint(checkpoint)  # Always full bytes; no stat-only cache here.
        boundary_bytes = sum(ref['bytes'] for ref in marker['files'])
        controller = read(checkpoint / 'controller.json')
        assert controller['run_id'] == out.name
        assert controller['state_before'] == commit['state_before']
        for key, path in (('config', out / 'config.json'),
                          ('selection', checkpoint.parent / 'selection.json'),
                          ('update', checkpoint.parent / 'update/update.json')):
            assert controller[key] == record(path), key
        # A convenience pointer may be stale; never use it as the restore authority.
    immutable(out / 'boundary_restore_audits' / f'{time.time_ns()}.json', dict(
        result='PASS', timestamp=now(), audit_scope='COMMIT_CHAIN_AND_FULL_LOAD_BOUNDARY',
        state=state, checkpoint=str(checkpoint) if checkpoint else None,
        committed_windows=len(commits), commits=commits,
        boundary_marker=record(checkpoint / 'COMMITTED.json') if checkpoint else None,
        commit_artifact_bytes_hashed=metadata_bytes,
        boundary_payload_bytes_hashed=boundary_bytes,
        historical_payload_bytes_stat_checked=history_payload_bytes - boundary_bytes,
        historical_payload_hashes_verified=False,
        full_history_audit_required_at_acceptance=True,
        seconds=time.monotonic() - started, implementation=record(__file__)))
    return state, checkpoint
