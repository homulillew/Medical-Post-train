from medical_posttrain.rl.common import durable,read,record
import importlib.util
from pathlib import Path


def test_missing_pointer_reconstructed_only_from_verified_synced_commit(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('audit_boundary',Path(__file__).parents[1]/'scripts/audit_stage4_boundary.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from medical_posttrain.rl import online
    checkpoint=tmp_path/'windows/0001/checkpoint'
    checkpoint.mkdir(parents=True)
    state=dict(policy_windows=2,training_groups=16,optimizer_steps=4)
    durable(tmp_path/'config.json',{})
    durable(checkpoint.parent/'commit.json',dict(state_after=state))
    monkeypatch.setattr(module,'validate',lambda cfg:{})
    monkeypatch.setattr(online,'restore',lambda *args:(state,checkpoint))
    assert module.reconcile(tmp_path)==state
    assert read(tmp_path/'checkpoint.json')['commit']==record(checkpoint.parent/'commit.json')
    assert len(list((tmp_path/'pointer_reconciliations').glob('*')))==1
    module.reconcile(tmp_path)
    assert len(list((tmp_path/'pointer_reconciliations').glob('*')))==1
