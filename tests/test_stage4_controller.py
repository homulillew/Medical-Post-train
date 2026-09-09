import itertools
import pytest

from medical_posttrain.rl.controller import select_groups,advance
from medical_posttrain.rl.common import encounter
from medical_posttrain.sampling.dynamic import Stream


def group(acc,index=0,policy='P0'):
    return dict(group_id=f'g{index}',prompt_id=f'p{index}',encounter_index=index,prompt_tokens=3,
        responses=[dict(group_id=f'g{index}',prompt_id=f'p{index}',trajectory_id=f'g{index}:{i}',
            policy_version=policy,config_sha256='cfg',reward_version='reward',acc=a,
            finish_reason='stop',output_tokens=5,total_reward=a*.8,parsed_answer='C') for i,a in enumerate(acc)])


@pytest.mark.parametrize('acc',list(itertools.product((0,1),repeat=4)))
def test_vanilla_accepts_every_valid_group_dynamic_only_contrast(acc):
    g = group(acc)
    assert len(select_groups([g],'P0','vanilla')[0]) == 1
    assert len(select_groups([g],'P0','dynamic')[0]) == int(0<sum(acc)<4)
    for r in g['responses']:
        r['total_reward'] = 10000
        r['output_tokens'] = 1
    assert len(select_groups([g],'P0','dynamic')[0]) == int(0<sum(acc)<4)


def test_overflow_never_becomes_next_policy_data():
    groups = [group([0,1,1,1],i) for i in range(4)]
    selected,decisions = select_groups(groups,'P0','dynamic',already=6)
    assert len(selected)==2
    assert [d['disposition'] for d in decisions] == ['selected','selected','overflow_eligible','overflow_eligible']
    assert not select_groups(groups[2:],'P1','dynamic')[0]


def test_invalid_not_mislabeled_allwrong():
    g = group([0,0,0,0])
    g['responses'].pop()
    selected,decisions = select_groups([g],'P0','vanilla')
    assert not selected and decisions[0]['classification']=='invalid'


def test_pair_randomness_independent_of_policy_and_run():
    s = Stream(['a','b','c'],42,'stage4:pair:test')
    for i in range(12):
        a = encounter(s,i,'vanilla','P0')
        b = encounter(s,i,'dynamic','P9')
        assert a['prompt_id']==b['prompt_id'] and a['request_seed']==b['request_seed']
        assert a['group_id']!=b['group_id']
    assert len({encounter(s,i,'x','P0')['prompt_id'] for i in range(3)})==3


def test_commit_only_complete_window_and_changed_policy():
    state = dict(policy_version='P0',cursor=0,policy_windows=0,optimizer_steps=0,
        training_groups=0,generated_groups=0,output_tokens=0,prompt_tokens=0,
        generated_exposure={},training_exposure={})
    groups = [group([0,1,0,1],i) for i in range(10)]
    _,decisions = select_groups(groups,'P0','dynamic')
    after = advance(state,groups,decisions,'P1',2)
    assert (after['training_groups'],after['generated_groups'],after['output_tokens'],after['cursor'])==(8,10,200,10)
    assert (after['policy_windows'],after['optimizer_steps'])==(1,2)
    assert state['cursor']==0
    with pytest.raises(ValueError):
        advance(state,groups,decisions,'P0',2)
    with pytest.raises(ValueError):
        advance(state,groups,decisions[:7],'P1',2)
