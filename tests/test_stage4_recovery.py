"""Recovery preserves orphan evidence and refuses to invent missing rollout cost."""
import pytest
from medical_posttrain.rl.common import durable,read
from medical_posttrain.rl.recovery import recover_actor_transaction


def setup_run(tmp_path,monkeypatch):
    from medical_posttrain.rl import common,online,recovery
    state=dict(policy_windows=1,optimizer_steps=2,training_groups=8)
    durable(tmp_path/'config.json',{})
    durable(tmp_path/'status.json',dict(status='FAILED'))
    monkeypatch.setattr(common,'validate',lambda cfg:{})
    monkeypatch.setattr(online,'restore',lambda out,init:(state,tmp_path/'trusted'))
    monkeypatch.setattr(recovery.subprocess,'check_output',lambda *a,**k:'0')
    return state


def test_partial_actor_archived_while_rollout_and_budget_preserved(tmp_path,monkeypatch):
    state=setup_run(tmp_path,monkeypatch)
    w=tmp_path/'windows/0001'
    durable(w/'state_before.json',state)
    durable(w/'batches/000/raw.json',dict(groups=['retained raw']))
    durable(w/'selection.json',dict(groups=['same selected group']))
    durable(w/'actor/partial.json',dict(orphan=True))
    result=recover_actor_transaction(tmp_path)
    assert result['state']==state
    assert read(w/'batches/000/raw.json')==dict(groups=['retained raw'])
    assert (w/'selection.json').exists() and not (w/'actor').exists()
    assert read(w/'recovery/001/actor/partial.json')==dict(orphan=True)
    assert read(tmp_path/'status.json')['status']=='INTERRUPTED'
    assert read(w/'recovery/001/recovery.json')['committed_budget_added']==0


def test_unreturned_generation_refuses_actor_recovery(tmp_path,monkeypatch):
    state=setup_run(tmp_path,monkeypatch)
    w=tmp_path/'windows/0001'
    durable(w/'state_before.json',state)
    durable(w/'batches/000/reservation.json',dict(request='unknown tail'))
    with pytest.raises(AssertionError,match='unknown'):
        recover_actor_transaction(tmp_path)
    assert read(tmp_path/'status.json')['status']=='FAILED'


def test_committed_boundary_recovery_preserves_counters(tmp_path,monkeypatch):
    state=setup_run(tmp_path,monkeypatch)
    result=recover_actor_transaction(tmp_path)
    assert result['state']==state
    assert read(tmp_path/'status.json')['status']=='INTERRUPTED'
