"""Recovery scope/equivalence and corruption checks without GPU training."""
import copy
import importlib.util
from pathlib import Path

import pytest
from medical_posttrain.rl.common import durable, read, record, sha256
from medical_posttrain.rl import online

spec = importlib.util.spec_from_file_location('resume_boundary',
    Path(__file__).parents[1] / 'scripts/stage4_resume_boundary.py')
boundary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boundary)


def fixture_run(out):
    initial = dict(adapter_sha256='sft')
    state = online.initial_state(initial)
    durable(out / 'config.json', dict(frozen=True))
    for i in range(2):
        w = out / 'windows' / f'{i:04d}'
        cp = w / 'checkpoint'
        durable(w / 'selection.json', dict(groups=list(range(8))))
        durable(w / 'update/update.json', dict(optimizer_steps_after=(i + 1) * 2))
        durable(cp / 'adapter/adapter_model.safetensors', dict(weights=i))
        durable(cp / 'native/optim.pt', dict(step=(i + 1) * 2))
        durable(cp / 'controller.json', dict(run_id=out.name, state_before=state,
            config=record(out / 'config.json'), selection=record(w / 'selection.json'),
            update=record(w / 'update/update.json')))
        refs = [dict(path=str(p.relative_to(cp)), bytes=p.stat().st_size, sha256=sha256(p))
                for p in sorted(cp.rglob('*')) if p.is_file()]
        durable(cp / 'COMMITTED.json', dict(files=refs, optimizer_step=(i + 1) * 2))
        after = copy.deepcopy(state)
        after.update(policy_windows=i + 1, optimizer_steps=(i + 1) * 2,
                     training_groups=(i + 1) * 8,
                     policy_version=sha256(cp / 'adapter/adapter_model.safetensors'))
        durable(w / 'commit.json', dict(state_before=state, state_after=after,
            checkpoint=str(cp), artifacts=[record(w / p) for p in
                ('selection.json', 'update/update.json', 'checkpoint/COMMITTED.json')]))
        state = after
    return initial


def test_exact_equivalence_to_full_restore_and_stale_pointer_ignored(tmp_path):
    initial = fixture_run(tmp_path)
    durable(tmp_path / 'checkpoint.json', dict(state='stale'))
    assert boundary.restore_boundary(tmp_path, initial) == online.restore(tmp_path, initial)
    audit = read(next((tmp_path / 'boundary_restore_audits').glob('*')))
    assert audit['committed_windows'] == 2
    assert audit['boundary_payload_bytes_hashed'] > 0
    assert audit['historical_payload_hashes_verified'] is False
    assert audit['full_history_audit_required_at_acceptance'] is True


def test_full_hash_only_latest_checkpoint(tmp_path, monkeypatch):
    initial = fixture_run(tmp_path)
    visited = []
    verify = boundary.verify_checkpoint
    def track(path):
        visited.append(path.parent.name)
        return verify(path)
    monkeypatch.setattr(boundary, 'verify_checkpoint', track)
    boundary.restore_boundary(tmp_path, initial)
    assert visited == ['0001']


def test_same_size_boundary_corruption_fails(tmp_path):
    initial = fixture_run(tmp_path)
    p = tmp_path / 'windows/0001/checkpoint/native/optim.pt'
    p.write_bytes(p.read_bytes().replace(b'4', b'9'))
    with pytest.raises(AssertionError):
        boundary.restore_boundary(tmp_path, initial)
    assert not (tmp_path / 'boundary_restore_audits').exists()


def test_historical_same_size_corruption_is_explicitly_deferred_to_full_audit(tmp_path):
    initial = fixture_run(tmp_path)
    p = tmp_path / 'windows/0000/checkpoint/native/optim.pt'
    p.write_bytes(p.read_bytes().replace(b'2', b'9'))
    boundary.restore_boundary(tmp_path, initial)
    with pytest.raises(AssertionError):
        online.restore(tmp_path, initial)


@pytest.mark.parametrize('relative', ['windows/0000/checkpoint/native/optim.pt',
                                     'windows/0001/checkpoint/native/optim.pt'])
def test_missing_payload_fails(tmp_path, relative):
    initial = fixture_run(tmp_path)
    (tmp_path / relative).unlink()
    with pytest.raises(AssertionError):
        boundary.restore_boundary(tmp_path, initial)


@pytest.mark.parametrize('relative', ['windows/0000/selection.json',
                                     'windows/0000/checkpoint/COMMITTED.json'])
def test_historical_metadata_corruption_fails(tmp_path, relative):
    initial = fixture_run(tmp_path)
    p = tmp_path / relative
    p.write_text(p.read_text() + ' ')
    with pytest.raises(AssertionError):
        boundary.restore_boundary(tmp_path, initial)


def test_broken_state_lineage_fails(tmp_path):
    initial = fixture_run(tmp_path)
    p = tmp_path / 'windows/0001/commit.json'
    value = read(p)
    value['state_before']['optimizer_steps'] = 99
    durable(p, value)
    with pytest.raises(AssertionError, match='lineage'):
        boundary.restore_boundary(tmp_path, initial)


def test_uncommitted_gap_fails(tmp_path):
    initial = fixture_run(tmp_path)
    (tmp_path / 'windows/0000/commit.json').unlink()
    with pytest.raises(AssertionError, match='gap'):
        boundary.restore_boundary(tmp_path, initial)


def test_inflight_window_remains_uncommitted_and_unchanged(tmp_path):
    initial = fixture_run(tmp_path)
    p = tmp_path / 'windows/0002/batches/000/raw.json'
    durable(p, dict(paid=True))
    before = record(p)
    state, _ = boundary.restore_boundary(tmp_path, initial)
    assert state['training_groups'] == 16
    assert record(p) == before
    assert not (p.parents[2] / 'commit.json').exists()


def test_fresh_initialization_has_no_checkpoint(tmp_path):
    assert boundary.restore_boundary(tmp_path, dict(adapter_sha256='sft')) == (
        online.initial_state(dict(adapter_sha256='sft')), None)
