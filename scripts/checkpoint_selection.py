#!/usr/bin/env python3
"""Offline selection rules and fail-closed prerequisite checks; no inference code."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GROUPS = [512, 1024, 1536, 2048, 2560, 3072, 3584, 4096, 4608, 5000]


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path = Path(path).resolve()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=path.stat().st_size)


def freeze_json(path, value):
    """Creation only: an existing protocol can never be silently replaced."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def require_stage4_gate(root=ROOT):
    root = Path(root)
    stage = read(root / 'project_state.json')['stages']['4']
    if stage['status'] != 'DONE':
        raise PermissionError('Stage4 must be DONE before selection1024 inference')
    ref = stage.get('verification_receipt')
    if not ref:
        raise PermissionError('Missing Stage4 full verification receipt')
    path = root / ref['path']
    if reference(path)['sha256'] != ref['sha256']:
        raise PermissionError('Stage4 receipt hash mismatch')
    receipt = read(path)
    required = dict(stage=4, scope='FULL', result='PASS', READY_FOR_STAGE5='YES')
    if any(receipt.get(k) != v for k, v in required.items()):
        raise PermissionError('A full Stage4 PASS with READY_FOR_STAGE5=YES is required')
    if receipt.get('verifier_sha256') != reference(root / 'scripts/verify_stage4.py')['sha256']:
        raise PermissionError('Stage4 verifier revision changed')
    if receipt.get('contract_sha256') != reference(root / 'contracts/stage_budgets.json')['sha256']:
        raise PermissionError('Stage4 contract revision changed')
    manifest = read(root / 'experiments/stage4/deliverables.json')
    if manifest.get('READY_FOR_STAGE5') != 'YES':
        raise PermissionError('Deliverables are not ready')
    for key in ['stage_report', 'interview_story', 'cases', 'resume_evidence', 'plots', 'checkpoint_index', 'pilot_review']:
        refs = manifest[key] if isinstance(manifest[key], list) else [manifest[key]]
        if not refs or any(reference(root / r['path']) != r for r in refs):
            raise PermissionError('Missing or changed Stage4 deliverable: ' + key)
    return receipt


def validate_selection_input(protocol, split, prompt_ids):
    if split != 'cmexam_val_selection1024':
        raise ValueError('Only frozen selection1024 validation inputs are allowed')
    if prompt_ids != protocol['dataset']['prompt_ids'] or len(set(prompt_ids)) != 1024:
        raise ValueError('Selection IDs/order must exactly match the frozen protocol')


def ranking_key(row):
    required = {'checkpoint_id', 'training_groups', 'n', 'correct', 'unparseable', 'truncated'}
    if set(row) != required or row['n'] != 1024:
        raise ValueError('Only complete selection1024 counts and checkpoint identity are accepted')
    for key in ['n', 'correct', 'unparseable', 'truncated', 'training_groups']:
        if type(row[key]) is not int or row[key] < 0:
            raise ValueError('Counts must be nonnegative integers')
    if any(row[k] > row['n'] for k in ['correct', 'unparseable', 'truncated']):
        raise ValueError('Invalid counts')
    return (-row['correct'], row['unparseable'], row['truncated'], row['training_groups'], row['checkpoint_id'])


def choose(protocol, variant, summaries, split, prompt_ids, root=ROOT):
    require_stage4_gate(root)
    validate_selection_input(protocol, split, prompt_ids)
    candidates = {c['checkpoint_id']: c for c in protocol['candidates'][variant]}
    if len(summaries) != len(candidates) or {s['checkpoint_id'] for s in summaries} != set(candidates):
        raise ValueError('Every frozen candidate must complete selection')
    for row in summaries:
        if row['training_groups'] != candidates[row['checkpoint_id']]['training_groups']:
            raise ValueError('Checkpoint identity/count mismatch')
    return min(summaries, key=ranking_key)['checkpoint_id']


if __name__ == '__main__':
    require_stage4_gate()
    print('Stage4 prerequisite gate PASS; this module does not execute inference.')
