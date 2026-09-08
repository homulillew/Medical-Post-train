import importlib.util
import json
from pathlib import Path
import random
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from verify_stage import verify_training_records

@pytest.fixture
def evidence():
    rows=[dict(sample_id=str(i),source='medical_o1' if i<10000 else 'huatuo',total_tokens=3,labels=[-100,3,4]) for i in range(20000)]
    order=list(rows);random.Random(42).shuffle(order);ids=[r['sample_id'] for r in order]
    metrics=[dict(event='update',global_step=i//16+1,examples=16,sample_ids=ids[i:i+16],sample_cursor=i+16,loss=1.,grad_norm=.5,learning_rate=.0001,next_learning_rate=.0001,update_seconds=1.,tokens_per_second=48.,processed_tokens=48,supervised_tokens=32) for i in range(0,20000,16)]
    coverage=dict(planned_ids=ids,covered_ids=ids,missing_ids=[],fraction=1.,processed_tokens=60000,supervised_tokens=40000)
    config=dict(run_class='FORMAL',planned_examples=20000,planned_epochs=1,seed=42)
    manifest=dict(run_class='FORMAL',dirty_state='')
    summary=dict(coverage_fraction=1.,unique_examples=20000,global_step=1250,processed_tokens=60000,supervised_tokens=40000,initial_trainable_digest='initial',final_trainable_digest='final')
    return config,manifest,rows,metrics,coverage,summary

def test_complete_lineage_is_consistent(evidence):
    assert verify_training_records(*evidence)['unique_examples']==20000

@pytest.mark.parametrize('mutation',['pilot','short','duplicate','nan','wrong_tokens','wrong_source','unchanged_adapter'])
def test_reject_completion_shortcuts(evidence,mutation):
    config,manifest,rows,metrics,coverage,summary=evidence
    if mutation=='pilot':config['run_class']='PILOT'
    elif mutation=='short':metrics[:]=metrics[:100]
    elif mutation=='duplicate':metrics[-1]['sample_ids'][0]=metrics[0]['sample_ids'][0]
    elif mutation=='nan':metrics[50]['loss']=float('nan')
    elif mutation=='wrong_tokens':metrics[40]['supervised_tokens']+=1
    elif mutation=='wrong_source':rows[0]['source']='huatuo'
    else:summary['final_trainable_digest']=summary['initial_trainable_digest']
    with pytest.raises(AssertionError):verify_training_records(*evidence)
