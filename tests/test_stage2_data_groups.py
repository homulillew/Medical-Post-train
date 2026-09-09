from copy import deepcopy
import pytest
from medical_posttrain.data.exam import messages, options_schema
from medical_posttrain.rollout.statistics import group_summary


def test_actor_side_channel_separation():
    row=dict(question='题干',options={'A':'选项一','B':'选项二'},answer_set='B',reference_explanation='SECRET_REFERENCE',cluster_id='SECRET_CLUSTER')
    first=messages(row)
    row.update(answer_set='A',reference_explanation='OTHER_SECRET',cluster_id='OTHER_CLUSTER')
    assert messages(row)==first
    assert 'SECRET' not in str(first)


def test_options_schema_multiline():
    assert options_schema('A 首行\n续行\nB 其他')=={'A':'首行\n续行','B':'其他'}
    for text in ('A one\nA two','A one','A \nB two','A one\nC two'):
        with pytest.raises(ValueError):options_schema(text)


def rows(acc):
    return [dict(trajectory_id=str(i),prompt_id='p',group_id='g',policy_version='v',config_sha256='c',reward_version='r',
                 acc=a,total_reward=.8*a+.05,finish_reason='stop',output_tokens=10,format=1,raw_output=str(i),ground_truth='A',prompt_tokens=15)
            for i,a in enumerate(acc)]


def test_all_16_group_configurations():
    import itertools
    for acc in itertools.product((0,1),repeat=4):
        result=group_summary(rows(acc));assert result['acc_vector']==list(acc)
        assert result['classification']==('all_wrong' if sum(acc)==0 else 'all_correct' if sum(acc)==4 else 'mixed')
        assert result['sampling_metric']=='acc'
        assert 'advantage_std' not in result


def test_group_lineage_failures_and_reward_independence():
    good=rows([0,0,0,0]);good[0]['total_reward']=0
    result=group_summary(good)
    assert result['hybrid_reward_std']>0 and result['classification']=='all_wrong'
    for field,value in [('trajectory_id','1'),('policy_version','other'),('prompt_id','other'),('reward_version','other'),('config_sha256','other'),('finish_reason','engine_error')]:
        bad=deepcopy(good);bad[0][field]=value
        with pytest.raises(AssertionError):group_summary(bad)
    with pytest.raises(AssertionError):group_summary(good[:3])
