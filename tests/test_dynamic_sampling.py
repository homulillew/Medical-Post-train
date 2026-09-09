import copy
import itertools
import random
import pytest
from medical_posttrain.sampling.dynamic import Stream, classify, eligible, initial_state, apply_batch, stop_reason


def rows(pattern):
    return [dict(prompt_id='p', group_id='g', trajectory_id=f'g:{i}', policy_version='v', config_sha256='c', reward_version='r',
                 acc=a, finish_reason='stop', parsed_answer=['A'], output_tokens=10) for i,a in enumerate(pattern)]


@pytest.mark.parametrize('pattern', list(itertools.product((0,1),repeat=4)))
def test_all_patterns_and_adversarial_independence(pattern):
    expected=0<sum(pattern)<4
    assert eligible(pattern)==expected
    rr=rows(pattern)
    rng=random.Random(42)
    for _ in range(20):
        for row in rr:
            row.update(semantic=rng.random(), format=rng.randrange(2), total_reward=rng.uniform(-100,100),
                       response_length=rng.randrange(100000), output_tokens=rng.randrange(100000))
        assert classify(rr,'v')['eligible']==expected


@pytest.mark.parametrize('field', ['prompt_id','group_id','policy_version','config_sha256','reward_version'])
def test_lineage_mismatch(field):
    rr=rows([1,0,1,0]);rr[1][field]='different'
    assert not classify(rr,'v')['valid']


@pytest.mark.parametrize('problem', ['missing','duplicate','transport','unfinished','nonbinary','boolean','barrier','absent'])
def test_invalid_distinct_from_all_wrong(problem):
    rr=rows([0]*4)
    if problem=='missing':rr.pop()
    if problem=='duplicate':rr[1]['trajectory_id']=rr[0]['trajectory_id']
    if problem=='transport':rr[1]['transport_error']='timeout'
    if problem=='unfinished':rr[1]['finish_reason']=None
    if problem=='nonbinary':rr[1]['acc']=.5
    if problem=='boolean':rr[1]['acc']=True
    if problem=='absent':del rr[1]['acc']
    result=classify(rr,'different' if problem=='barrier' else 'v')
    assert not result['valid'] and result['classification']=='invalid' and not result['eligible']


def test_subtypes_do_not_change_eligibility():
    rr=rows([1,1,0,0])
    assert classify(rr,'v')['mixed_subtype']=='mixed_parsed_wrong'
    rr[2]['parsed_answer']=None
    assert classify(rr,'v')['mixed_subtype']=='mixed_both'
    rr[3]['parsed_answer']=None
    assert classify(rr,'v')['mixed_subtype']=='mixed_unparseable_only'
    assert classify(rr,'v')['eligible']


def config():
    return dict(seed=42, stream_domain='stage3:test',policy_version='v',target_accepted=2,mode='formal',max_num_generation_batches=3)


def group(stream,index,pattern):
    encounter=stream.encounter(index,'run','v'); rr=rows(pattern)
    for i,row in enumerate(rr):row.update(prompt_id=encounter['prompt_id'],group_id=encounter['group_id'],trajectory_id=encounter['group_id']+f':{i}')
    return dict(**encounter,responses=rr,prompt_tokens=7)


def test_cyclic_no_blacklist_resume_overflow_and_costs():
    cfg=config();stream=Stream(['a','b'],42,cfg['stream_domain']);state=initial_state(cfg)
    first=[group(stream,0,[0]*4),group(stream,1,[1]*4)]
    state,decisions=apply_batch(state,first,cfg,2)
    assert state['accepted_mixed_groups']==0 and state['epoch']==1 and state['cursor']==0
    # Round-trip exactly like checkpoint serialization, then same formerly rejected prompts can qualify.
    import json
    restored=json.loads(json.dumps(state))
    next_groups=[group(stream,i,[1,0,0,0]) for i in range(2,5)]
    actual,decisions=apply_batch(restored,next_groups,cfg,2)
    assert actual==apply_batch(state,next_groups,cfg,2)[0]
    assert [d['disposition'] for d in decisions]==['accepted','accepted','overflow_eligible']
    assert actual['accepted_mixed_groups']==2 and actual['generated_groups']==5
    assert sum(actual['output_tokens_by_disposition'].values())==200
    assert actual['completed_valid_trajectories']==20
    assert stop_reason(actual,cfg)=='FULL_PASS'
    assert len(actual['dispositions'])==5 and max(actual['prompt_exposures'].values())==3
    with pytest.raises(ValueError):apply_batch(actual,next_groups,cfg,2)
    other=dict(cfg,policy_version='v2')
    with pytest.raises(ValueError):apply_batch(actual,[],other,2)
    # New policy gets a fresh window and no stale prompt exclusion.
    assert Stream(['a','b'],42,cfg['stream_domain']).order(1)==stream.order(1)


def test_starvation_and_invalid_accounting():
    cfg=config();state=initial_state(cfg);stream=Stream(['p'],42,'test')
    for i in range(3):state,_=apply_batch(state,[group(stream,i,[0]*4)],cfg,1)
    assert stop_reason(state,cfg)=='BLOCKED_STARVATION'
    state=initial_state(cfg);g=group(stream,0,[0]*4);g['responses'].pop()
    state,decisions=apply_batch(state,[g],cfg,1)
    assert state['invalid_groups']==1 and state['all_wrong_groups']==0
    assert state['output_tokens_by_disposition']['invalid']==30
    assert stop_reason(state,cfg)=='FAILED_INVALID_GROUP'


def test_stream_is_full_deterministic_and_domain_independent():
    ids=[str(i) for i in range(100)]
    a=Stream(ids,42,'stage3:formal');b=Stream(ids[::-1],42,'stage3:formal')
    assert a.order(0)==b.order(0) and set(a.order(0))==set(ids)
    assert a.order(0)!=a.order(1) and a.order(0)!=Stream(ids,42,'stage3:smoke').order(0)
    assert a.encounter(0,'run','v')['group_id']!=a.encounter(100,'run','v')['group_id']
    assert a.encounter(0,'run','v')['request_seed']!=a.encounter(100,'run','v')['request_seed']


def test_finish_accepts_summary_with_existing_run_id(tmp_path):
    from medical_posttrain.evidence.stage3 import Run
    from medical_posttrain.evidence.stage2 import read
    import time
    run=object.__new__(Run);run.run_id='test';run.out=tmp_path/'bulk';run.git=tmp_path/'git';run.attempt=run.out/'attempt_001';run.attempt.mkdir(parents=True);run.start=time.monotonic()
    run.finish(dict(run_id='test',status='SMOKE_PASS',optimizer_updates=0))
    assert read(run.out/'summary.json')['run_id']=='test'
    assert read(run.out/'status.json')['status']=='SMOKE_PASS'
    assert read(run.git/'artifacts_manifest.json')
