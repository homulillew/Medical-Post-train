import json
import pytest
from medical_posttrain.rl.common import durable,read,record
from medical_posttrain.rl.schedule import validation_trigger
from medical_posttrain.rl.transactions import cost_ledger,rng_digest
from medical_posttrain.rl.health import assess,DiagnosticPause


def state(tokens,n=1):
    return dict(prompt_tokens=tokens//10,output_tokens=tokens-tokens//10,policy_windows=n,training_groups=n*8)


def test_first_crossing_exact_boundary_and_no_later_substitution():
    p=dict(checkpoint_windows=[0,64],generated_total_token_stride=100)
    assert validation_trigger(p,state(99),state(100,2),5000)['generated_total_token_milestones']==[100]
    assert not validation_trigger(p,state(100),state(101,2),5000)['evaluate']
    assert validation_trigger(p,state(90),state(305,2),5000)['generated_total_token_milestones']==[100,200,300]
    assert validation_trigger(p,state(90,63),state(305,64),5000)['group_schedule']
    assert validation_trigger(p,state(0,0),state(0,0),5000)['evaluate']
    assert validation_trigger(p,state(310,624),state(320,625),5000)['evaluate']


def test_physical_cost_retains_archived_optimizer_work_and_sync(tmp_path):
    durable(tmp_path/'windows/0000/batches/000/raw.json',dict(generation_seconds=3,groups=[dict(prompt_tokens=10,responses=[dict(output_tokens=4)]*4)]))
    for name in ('windows/0000/update','windows/0000/recovery/001/update'):
        p=tmp_path/name;p.mkdir(parents=True)
        (p/'events.jsonl').write_text(json.dumps(dict(event='optimizer_step',seconds=2))+'\n')
    for name in ('attempt_001/initial_sync/first.json','windows/0000/recovery/001/sync/repeat.json'):
        durable(tmp_path/name,dict(token_ids=[1,2],prompt_logprobs=[None,{}]))
    c=cost_ledger(tmp_path)
    assert c['physical_optimizer_steps']==2 and c['known_optimizer_seconds']==4
    assert c['known_prompt_tokens']==10 and c['known_output_tokens']==16
    assert c['known_control_output_tokens']==4 and c['known_control_prompt_tokens']==4


def test_rng_fingerprint_changes_with_state():
    import torch,numpy as np
    a=dict(torch=torch.tensor([1,2],dtype=torch.uint8),numpy=np.array([3,4],dtype=np.uint32))
    assert rng_digest(a)==rng_digest(a)
    b=dict(a,torch=torch.tensor([1,3],dtype=torch.uint8))
    assert rng_digest(a)!=rng_digest(b)


def test_health_single_high_clip_does_not_pause_but_persistent_does(tmp_path):
    cfg=dict(health_protocol=dict(clip_consecutive_windows=3,clip_threshold=.9,length_block_windows=16,length_consecutive_blocks=3,length_cap_fraction=.9))
    for i,c in enumerate([.95,.75,.95,.95,.95]):
        durable(tmp_path/'windows'/f'{i:04d}'/'update/update.json',dict(minibatches=[{},dict(clip_fraction=c)]))
    assess(tmp_path,state(1,3),cfg)
    with pytest.raises(DiagnosticPause):assess(tmp_path,state(1,5),cfg)
    with pytest.raises(DiagnosticPause):assess(tmp_path,state(1,5),cfg)


def test_length_pause_requires_multiple_complete_blocks(tmp_path):
    cfg=dict(sampling=dict(max_tokens=1024),health_protocol=dict(clip_consecutive_windows=100,clip_threshold=.9,
        length_block_windows=2,length_consecutive_blocks=3,length_cap_fraction=.9))
    for i in range(6):
        durable(tmp_path/'windows'/f'{i:04d}'/'batches/000/raw.json',dict(groups=[dict(responses=[dict(output_tokens=950)]*4)]))
    assess(tmp_path,state(1,4),cfg)
    with pytest.raises(DiagnosticPause):assess(tmp_path,state(1,6),cfg)
    assert read(tmp_path/'health/0006.json')['failures'][0]['block_p95']==[950,950,950]
