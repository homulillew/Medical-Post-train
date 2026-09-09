"""Validation cardinality/isolation and durable incomplete-request protection."""
import csv
from types import SimpleNamespace
import pytest
from medical_posttrain.rl.common import durable,record
from medical_posttrain.rl.validation import monitor_rows,evaluate


def protocol(tmp_path):
    source=tmp_path/'val.csv'
    with source.open('w') as f:
        w=csv.DictWriter(f,fieldnames=['Question','Options','Answer'])
        w.writeheader()
        w.writerows(dict(Question=str(i),Options='A. one\nB. two',Answer='A') for i in range(513))
    ids=[f'cmexam_val:revision:{i}' for i in range(512)]
    partition=tmp_path/'partition.json'
    durable(partition,dict(monitor=ids,selection=['cmexam_val:revision:512']))
    p=dict(source=record(source),partition=record(partition),monitor_ids=ids,
        decoding=dict(n=1,temperature=0,max_tokens=1024),seed=42)
    path=tmp_path/'protocol.json'
    durable(path,p)
    return p,record(path)


class Rollout:
    policy='fake-policy-for-unit-test'
    request=None
    def __init__(self,incomplete=False):
        self.llm=self
        self.incomplete=incomplete
        self.calls=0
    def prompt(self,r):
        return r['question'],[int(r['question'])]
    def generate(self,prompts,params,**kwargs):
        self.calls+=1
        result=[SimpleNamespace(prompt_token_ids=[int(p)],outputs=[SimpleNamespace(
            text='<think>x</think><answer>A</answer>',token_ids=[1,2,3],finish_reason='stop')]) for p in prompts]
        return result[:-1] if self.incomplete else result


def state():
    return dict(policy_windows=0,optimizer_steps=0,training_groups=0,
        prompt_tokens=0,output_tokens=0,policy_version=Rollout.policy)


def test_monitor_excludes_selection_and_hash_mismatch(tmp_path):
    p,_=protocol(tmp_path)
    rows=monitor_rows(p)
    assert len(rows)==512 and rows[-1]['prompt_id'].endswith(':511')
    p['monitor_ids'][-1]='cmexam_val:revision:512'
    with pytest.raises(AssertionError):
        monitor_rows(p)


def test_validation_full_count_cost_and_idempotent_read(tmp_path):
    _,ref=protocol(tmp_path)
    rollout=Rollout()
    summary=evaluate(rollout,tmp_path/'run',state(),ref)
    assert summary['accuracy']==1 and summary['n']==512
    assert summary['validation_output_tokens']==1536 and summary['validation_prompt_tokens']==512
    assert summary['cumulative_generated_total_tokens']==0
    assert rollout.calls==32
    assert evaluate(rollout,tmp_path/'run',state(),ref)==summary and rollout.calls==32


def test_incomplete_validation_is_retained_and_cannot_silently_retry(tmp_path):
    _,ref=protocol(tmp_path)
    with pytest.raises(AssertionError,match='Incomplete validation batch'):
        evaluate(Rollout(incomplete=True),tmp_path/'run',state(),ref)
    assert (tmp_path/'run/validation/0000/reservation_000.json').exists()
    with pytest.raises(AssertionError,match='unknown tail cost'):
        evaluate(Rollout(),tmp_path/'run',state(),ref)


def test_formal_monitor_first_token_crossing_reuses_same_measured_checkpoint(tmp_path,monkeypatch):
    from medical_posttrain.rl.online import monitor
    from medical_posttrain.rl import validation
    from medical_posttrain.rl.common import read
    p,ref=protocol(tmp_path)
    p.update(checkpoint_windows=[0,64],generated_total_token_stride=100)
    durable(ref['path'],p);ref=record(ref['path'])
    cfg=dict(mode='formal',target_training_groups=5000,validation_protocol=ref)
    out=tmp_path/'run'
    before=dict(state(),policy_windows=1,training_groups=8,prompt_tokens=10,output_tokens=89)
    after=dict(before,policy_windows=2,training_groups=16,optimizer_steps=4,output_tokens=91)
    durable(out/'windows/0001/state_before.json',before)
    rollout=Rollout()
    monitor(rollout,out,after,cfg)
    assert rollout.calls==32
    trigger=read(out/'validation/0002/trigger.json')
    assert trigger['generated_total_token_milestones']==[100] and trigger['actual_generated_total_tokens']==101
    monitor(rollout,out,after,cfg)
    assert rollout.calls==32
    later=dict(after,policy_windows=3,training_groups=24,output_tokens=92)
    durable(out/'windows/0002/state_before.json',after)
    monitor(rollout,out,later,cfg)
    assert rollout.calls==32 and not (out/'validation/0003').exists()
