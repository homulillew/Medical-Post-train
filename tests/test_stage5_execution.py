"""Meaningful CPU checks for phase order, identity/retry replay and private views."""
import copy
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.decoders import DecodeStream

from execute_stage5_closure import final_jobs
from stage5_objective_evidence import verify_item, open_view, machine_triage, put
from medical_posttrain.evaluation.core import digest, score


@pytest.mark.parametrize('selected_groups',[('5000','5000'),('512','5000'),('512','1024')])
def test_all_exam_jobs_precede_open_and_identical_endpoints_reused(selected_groups):
    def c(arm,n):return dict(checkpoint_id=arm+':'+n,adapter={'sha256':arm+n})
    protocol=dict(fixed_sft=dict(adapter={'sha256':'sft'}),scientific_endpoints={a:c(a,'5000') for a in ['vanilla','dynamic']})
    selected=dict(selected_checkpoints={a:c(a,n) for a,n in zip(['vanilla','dynamic'],selected_groups)})
    exams,opened=final_jobs(protocol,selected)
    assert len(exams)==2*len({'sft','vanilla:5000','dynamic:5000'}|{a+':'+n for a,n in zip(['vanilla','dynamic'],selected_groups)})
    assert all(j[2] in ['cmexam_test_scorable','cmb_exam_clean_2000'] for j in exams)
    assert len(opened)==6 and all(j[2] in ['cmb_clin','open_qa_retention_200'] for j in opened)
    assert len({(cid,ds) for cid,_,ds in exams+opened})==len(exams+opened)
    assert not any('grpo' in cid or 'random' in cid for cid,_,_ in exams+opened)


def replay_fixture():
    backend=Tokenizer(WordLevel({'[UNK]':0,'Q':1,'最终答案：A':2},unk_token='[UNK]'))
    tok=SimpleNamespace(_tokenizer=backend,apply_chat_template=lambda *a,**k:'Q',encode=lambda *a,**k:[1])
    decoder=DecodeStream(ids=[1],skip_special_tokens=True)
    text=decoder.step(backend,2) or ''
    raw=dict(raw_output=text,finish_reason='stop',prompt_tokens=1,output_tokens=1,prompt_token_ids=[1],output_token_ids=[2],adapter_sha256='adapter')
    item=dict(id='synthetic:1',options={'A':'a','B':'b'},answer='A');request=dict(id=item['id'],messages=[])
    reservation=dict(prompt_id=item['id'],checkpoint_id='vanilla:512',run_id='synthetic',request_sha256=digest(request),attempt=1,reason='first request')
    response=score(raw,item,'vanilla:512','synthetic')
    return dict(raw=raw,response=response,item=item,request=request,reservation=reservation,attempts=[copy.deepcopy(reservation)],
                adapter_sha='adapter',checkpoint_id='vanilla:512',run_id='synthetic',exam=True,decoding={'max_tokens':1024},tokenizer=tok)


def test_real_token_decode_and_parser_replay():
    args=replay_fixture();verify_item(**args)
    assert args['response']['correct']


@pytest.mark.parametrize('mutation',['adapter','raw_text','prompt_tokens','output_count','result','request','duplicate_attempt','wrong_retry'])
def test_replay_rejects_tampering(mutation):
    a=replay_fixture()
    if mutation=='adapter':a['raw']['adapter_sha256']='other'
    elif mutation=='raw_text':a['raw']['raw_output']='最终答案：B'
    elif mutation=='prompt_tokens':a['raw']['prompt_token_ids']=[0]
    elif mutation=='output_count':a['raw']['output_tokens']=0
    elif mutation=='result':a['response']['correct']=False
    elif mutation=='request':a['request']['messages']=[{'role':'user','content':'changed'}]
    elif mutation=='duplicate_attempt':a['attempts'].append(copy.deepcopy(a['attempts'][0]))
    elif mutation=='wrong_retry':a['attempts'].append(dict(a['attempts'][0],attempt=2,reason='answer was wrong'))
    with pytest.raises(AssertionError):verify_item(**a)


def test_private_think_never_in_visible_view_and_triage_is_not_safety_verdict():
    r=dict(prompt_id='x',checkpoint_id='sft',adapter_sha256='a',raw_output='<think>PRIVATE</think><answer>Visible</answer>',finish_reason='stop',prompt_tokens=5,output_tokens=10)
    v=open_view(r)
    assert v['hidden_think']=='PRIVATE' and v['visible_answer']=='Visible'
    assert machine_triage(v)['status']=='MACHINE_TRIAGE_ONLY' and not machine_triage(v)['clinical_validation']
    malformed=open_view(dict(r,raw_output='<think>PRIVATE',finish_reason='length'))
    assert malformed['visible_answer']=='' and malformed['hidden_or_malformed_raw']=='<think>PRIVATE'
    assert set(machine_triage(malformed)['flags'])=={'EMPTY_VISIBLE_ANSWER','TRUNCATED_OUTPUT'}


def test_immutable_analysis_can_resume_but_not_replace(tmp_path):
    p=tmp_path/'evidence.json'
    assert put(p,{'result':'original'})==put(p,{'result':'original'})
    with pytest.raises(AssertionError):put(p,{'result':'better'})


def test_known_missing_option_is_preserved_without_changing_parser():
    from stage5_selection_source import GAP_ID,GAP_OPTIONS,source_options
    from medical_posttrain.data.exam import options_schema,canonical_answer
    row={'Options':GAP_OPTIONS,'Answer':'A'}
    with pytest.raises(ValueError):options_schema(GAP_OPTIONS)
    options=source_options(GAP_ID,row)
    assert list(options)==['A','B','D','E'] and '\n'.join(k+' '+v for k,v in options.items())==GAP_OPTIONS
    assert canonical_answer('A',''.join(options))=='A'
    with pytest.raises(ValueError):canonical_answer('C',''.join(options))
    with pytest.raises(ValueError):source_options('another-id',row)
    with pytest.raises(ValueError):source_options(GAP_ID,dict(row,Answer='C'))


def test_gpu_wait_preserves_other_worker_and_waits_for_release(monkeypatch):
    import execute_stage5_closure as runner
    attempts=[];status=[]
    def guard():
        attempts.append(1)
        if len(attempts)==1:raise PermissionError('GPU is owned by an active process; Stage5 inference refused')
    monkeypatch.setattr(runner,'gpu_guard',guard)
    monkeypatch.setattr(runner,'durable',lambda p,v:status.append(v['status']))
    monkeypatch.setattr(runner.time,'sleep',lambda t:None)
    runner.wait_for_gpu()
    assert len(attempts)==2 and status==['WAITING_FOR_EXCLUSIVE_GPU']


def test_unknown_gpu_error_fails_closed(monkeypatch):
    import execute_stage5_closure as runner
    def guard():raise PermissionError('GPU ownership cannot be established')
    monkeypatch.setattr(runner,'gpu_guard',guard)
    with pytest.raises(PermissionError):runner.wait_for_gpu()
