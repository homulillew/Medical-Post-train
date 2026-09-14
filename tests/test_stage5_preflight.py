"""CPU-only synthetic integration. These tests never import a model runtime."""
import sys,json,copy,subprocess,importlib.abc
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import freeze,ref,read,digest,score,summarize,paired,sliced,wilson
from medical_posttrain.evaluation.gates import stage4_gate,gpu_guard,final_gate,reservation_guard
from medical_posttrain.evaluation.executor import execute_items
from medical_posttrain.evaluation.blind import visible_answer,make_schedule,build_bundle,human_audit,position_consistency,safety_aggregate,safety_divergences,aggregate_blind,DIMENSIONS
from select_stage5_checkpoint import rank_candidates

def put(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(obj));return ref(path)

def root_fixture(path):
    (path/'scripts').mkdir(parents=True);(path/'scripts/checkpoint_selection.py').write_text((ROOT/'scripts/checkpoint_selection.py').read_text());(path/'scripts/verify_stage4.py').write_text('# synthetic verifier fixture\n')
    put(path/'contracts/stage_budgets.json',{'synthetic':True})
    artifact=put(path/'fake_deliverable.json',{'synthetic':True})
    receipt=dict(stage=4,scope='FULL',result='PASS',READY_FOR_STAGE5='YES',verifier_sha256=ref(path/'scripts/verify_stage4.py')['sha256'],contract_sha256=ref(path/'contracts/stage_budgets.json')['sha256'])
    rr=put(path/'full_receipt.json',receipt)
    state=dict(stages={'4':dict(status='DONE',verification_receipt=rr),'5':dict(status='NOT_STARTED'),'6':dict(status='NOT_STARTED')})
    put(path/'project_state.json',state)
    put(path/'experiments/stage4/deliverables.json',dict(READY_FOR_STAGE5='YES',**{k:artifact for k in ['stage_report','interview_story','cases','resume_evidence','plots','checkpoint_index','pilot_review']}))
    return path

@pytest.mark.parametrize('condition',['FULL_PASS','no_receipt','wrong_hash','not_ready','valid'])
def test_stage4_gate(tmp_path,condition):
    root=root_fixture(tmp_path);state=read(root/'project_state.json')
    if condition=='FULL_PASS':state['stages']['4']['status']='FULL_PASS'
    elif condition=='no_receipt':state['stages']['4']['verification_receipt']=None
    elif condition=='wrong_hash':state['stages']['4']['verification_receipt']['sha256']='0'*64
    elif condition=='not_ready':
        p=root/'experiments/stage4/deliverables.json';d=read(p);d['READY_FOR_STAGE5']='NO';put(p,d)
    put(root/'project_state.json',state)
    if condition=='valid':assert stage4_gate(root)['result']=='PASS'
    else:
        with pytest.raises(PermissionError):stage4_gate(root)

@pytest.mark.parametrize('snapshot',[dict(cuda_processes=['123, vllm,100'],training_workers=[]),dict(cuda_processes=[],training_workers=[dict(pid=123)]),dict(cuda_processes=[],training_workers=[])])
def test_gpu_guard(snapshot):
    if snapshot['cuda_processes'] or snapshot['training_workers']:
        with pytest.raises(PermissionError):gpu_guard(snapshot)
    else:assert gpu_guard(snapshot)==snapshot

def test_ranking_all_tiebreaks():
    base=dict(n=1024,correct=700,unparseable=10,truncated=10,training_groups=512)
    specs=[('best',{}),('accuracy',dict(correct=699)),('unparseable',dict(unparseable=11)),('truncation',dict(truncated=11)),('later',dict(training_groups=1024)),('zz_lexical',{})]
    data=[dict(base,checkpoint_id=n,**changes) for n,changes in specs]
    candidates=[{k:r[k] for k in ['checkpoint_id','training_groups']} for r in data]
    result=rank_candidates(data,candidates);assert result==rank_candidates(data[::-1],candidates)
    assert result['selected_checkpoint']=='best'
    assert {t['deciding_key'] for t in result['tie_break_trace']}=={'accuracy','unparseable','truncation','earlier_checkpoint','checkpoint_id'}
    for forbidden in ['response_length','semantic','test_score','cmb_score','grpo_auxiliary']:
        corrupt=copy.deepcopy(data);corrupt[0][forbidden]=1
        with pytest.raises(ValueError):rank_candidates(corrupt,candidates)
    with pytest.raises(ValueError):rank_candidates(data[:-1],candidates)

def fixture100():
    items=[dict(id=f'synthetic:{i}',question='Synthetic exam only',answer='A',options={'A':'one','B':'two'},source='synthetic') for i in range(100)]
    requests=[dict(id=r['id'],messages=[dict(role='user',content=r['question'])]) for r in items]
    def backend(reqs):
        values=[]
        for r in reqs:
            i=int(r['id'].split(':')[1]);text='<answer>A</answer>' if i<60 else '最终答案：B' if i<90 else '<think>unfinished' if i<95 else 'ambiguous answer'
            values.append(dict(raw_output=text,finish_reason='length' if 90<=i<95 else 'stop',prompt_tokens=10,output_tokens=20+i))
        return values
    return items,requests,backend

def test_synthetic100_integration(tmp_path):
    items,requests,backend=fixture100();a=execute_items(items,requests,backend,tmp_path/'A','synthetic_A','SYNTHETIC_CPU_ONLY')
    s=summarize(a);assert (s['n'],s['correct_count'],s['unparseable_count'],s['truncation_count'])==(100,60,10,5)
    b=copy.deepcopy(a)
    for i in range(10):b[i]['correct']=False
    for i in range(60,80):b[i]['correct']=True
    q=paired(a,b);assert q==paired(a,b) and q['delta_pp']==10 and q['wrong_to_correct']==20 and q['correct_to_wrong']==10
    assert 0<q['exact_mcnemar']<1
    slice_result=sliced(a,dict(first=[r['id'] for r in items[:50]],last=[r['id'] for r in items[50:]]));assert slice_result['first']['correct_count']==50 and slice_result['last']['n']==50
    assert wilson(60,100)[0]<.6<wilson(60,100)[1]
    with pytest.raises(ValueError):paired(a,b[:-1])
    calls=[]
    def forbidden_backend(req):calls.append(req);raise AssertionError('Completed wrong responses must not rerun')
    replay=execute_items(items,requests,forbidden_backend,tmp_path/'A','synthetic_A','SYNTHETIC_CPU_ONLY',resume=True)
    assert replay==a and not calls

def test_technical_retry_requires_explicit_resume(tmp_path):
    items,requests,backend=fixture100();items=items[:1];requests=requests[:1]
    def fail(req):raise ConnectionError('synthetic transport failure')
    with pytest.raises(ConnectionError):execute_items(items,requests,fail,tmp_path,'A','SYNTHETIC')
    with pytest.raises(PermissionError):execute_items(items,requests,backend,tmp_path,'A','SYNTHETIC',resume=True)
    result=execute_items(items,requests,backend,tmp_path,'A','SYNTHETIC',resume=True,technical_retry=True)
    assert len(result)==1 and len(list(tmp_path.glob('*/attempt_*.json')))==2
    with pytest.raises(PermissionError):execute_items(items,requests,backend,tmp_path,'A','SYNTHETIC')

def test_visible_blind_position_human_safety(tmp_path):
    assert visible_answer('<think>SECRET</think><answer>Visible</answer>')=='Visible'
    assert visible_answer('<think>SECRET')=='' and visible_answer('<think>a</think><think>b</think>c')==''
    items=[dict(id=str(i),source='clinical' if i<10 else 'retention',question='Q',**({'case_id':str(i//2)} if i<10 else {})) for i in range(20)]
    schedule=make_schedule(items,[['dynamic','vanilla']]);assert schedule==make_schedule(items,[['dynamic','vanilla']]) and len(schedule['human_audit_ids'])>=4
    outputs={m:{r['id']:dict(raw_output='<think>SECRET</think>'+m+' final') for r in items} for m in ['dynamic','vanilla']}
    public,private=build_bundle(items,outputs,schedule,tmp_path)
    assert all('SECRET' not in r['A']+r['B'] for r in public) and all('models' not in r and 'prompt_id' not in r for r in public)
    judgments=[dict(pair_id=r['pair_id'],preference='tie',judge_id='synthetic',judge_version='1',scores={s:{d:2 for d in DIMENSIONS} for s in ['A','B']}) for r in public]
    assert position_consistency(judgments,schedule)['consistency']==1
    assert human_audit([],schedule['human_audit_ids'],[])['result']=='INCOMPLETE'
    reviews=[dict(prompt_id=i,reviewer_id='synthetic',reviewed_in_full=True,observation='Synthetic fixture',is_clinician=False,disclaimer='NON_CLINICIAN_REVIEW_NOT_CLINICAL_VALIDATION') for i in schedule['human_audit_ids']]
    assert human_audit(reviews,schedule['human_audit_ids'],[])['result']=='PASS'
    assert human_audit(reviews,schedule['human_audit_ids'],['unreviewed-critical'])['result']=='INCOMPLETE'
    stats=aggregate_blind(judgments,schedule,items,resamples=100);assert stats['dynamic_vs_vanilla:clinical']['clusters']==5
    a=[dict(prompt_id='1',status='safe_escalated',evidence='synthetic',reviewer_id='test')];b=[dict(a[0],status='unsafe',critical=True)]
    assert safety_aggregate(b,['1'])['unsafe_count']==1 and len(safety_divergences(a,b)['regressions'])==1

def final_fixture(tmp_path):
    root=root_fixture(tmp_path);idx=root/'experiments/stage5';dummy=put(root/'dummy.json',{})
    put(idx/'preflight_seal_v1.json',dict(artifacts=[],sources=[]))
    candidates={a:[dict(checkpoint_id=a+':512',training_groups=512,adapter=dummy)] for a in ['vanilla','dynamic']}
    p=dict(dataset=dict(partition=dummy,source_metadata=dummy),parser=dummy,prompt=dict(messages_source=dummy,chat_template=dummy),candidates=candidates,scientific_endpoints={a:dict(checkpoint_id=a+':5000',adapter=dummy) for a in candidates})
    pr=put(idx/'checkpoint_selection_protocol_v1.json',p)
    protocols=[put(idx/n,dict(synthetic=True)) for n in ['cmexam_final_eval_protocol_v1.json','cmb_primary_audit_v1.json','open_qa_eval_protocol_v1.json']]
    for n in ['cmexam_test_scorable','cmb_exam_clean_2000','cmb_clin','open_qa_retention_200']:put(idx/'manifests'/f'{n}.json',dict(data=dummy,requests=dummy))
    selected=dict(status='FROZEN_SELECTED',protocol=pr,summary_sources=[dummy],selected_checkpoints={a:c[0] for a,c in candidates.items()})
    sr=put(idx/'selection_result_manifest_v1.json',selected);put(idx/'selection_result_lock_v1.json',dict(selection_result=sr,final_protocols=protocols))
    return root

@pytest.mark.parametrize('mutation',['selection_missing','selection_not_frozen','selection_hash','protocol_hash','source_hash','checkpoint_id','valid'])
def test_final_gate(tmp_path,mutation):
    root=final_fixture(tmp_path);idx=root/'experiments/stage5'
    if mutation=='selection_missing':(idx/'selection_result_lock_v1.json').unlink()
    elif mutation=='selection_not_frozen':
        file=idx/'selection_result_manifest_v1.json';d=read(file);d['status']='RUNNING';r=put(file,d);lock=read(idx/'selection_result_lock_v1.json');lock['selection_result']=r;put(idx/'selection_result_lock_v1.json',lock)
    elif mutation=='selection_hash':(idx/'selection_result_manifest_v1.json').write_text('{}')
    elif mutation=='protocol_hash':(idx/'cmexam_final_eval_protocol_v1.json').write_text('{}')
    elif mutation=='source_hash':(root/'dummy.json').write_text('changed')
    if mutation=='valid':assert final_gate('sft',root)['status']=='FROZEN_SELECTED'
    else:
        with pytest.raises(PermissionError):final_gate('grpo:512' if mutation=='checkpoint_id' else 'sft',root)

def test_cpu_dry_run_import_guard():
    script='''import sys,importlib.abc,runpy
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','vllm','transformers','peft','requests','httpx'}:raise RuntimeError('FORBIDDEN_CPU_PREFLIGHT_IMPORT:'+fullname)
sys.meta_path.insert(0,Block())
sys.argv=['scripts/run_stage5_eval.py','--dry-run']
runpy.run_path(sys.argv[0],run_name='__main__')
'''
    subprocess.run([sys.executable,'-c',script],cwd=ROOT,check=True,capture_output=True,text=True)

@pytest.mark.parametrize('mutation',['valid','adapter','decoding','requests','engine','grpo','prior_attempt'])
def test_worker_spec_binding_before_model_load(tmp_path,monkeypatch,mutation):
    import medical_posttrain.evaluation.gates as g
    root=tmp_path;idx=root/'experiments/stage5';out=root/'artifacts';dummy=put(root/'adapter_fixture.json',{'synthetic':True})
    items=out/'selection_inputs/items.jsonl';items.parent.mkdir(parents=True);items.write_text('{"id":"synthetic:1"}\n')
    requests=out/'selection_inputs/requests.jsonl';requests.write_text('{"id":"synthetic:1","messages":[]}\n')
    engine={'max_model_len':4096};p=dict(candidates={'vanilla':[dict(checkpoint_id='vanilla:512',adapter=dummy)]},dataset=dict(prompt_ids=['synthetic:1']),decoding=dict(n=1),runtime=dict(engine=engine,environment={}))
    put(idx/'selection_execution_plan_v1.json',dict(artifact_root=str(out)));put(root/'configs/stages/s4_formal_shared.json',dict(model='SYNTHETIC_NOT_A_MODEL',engine=engine))
    monkeypatch.setattr(g,'selection_gate',lambda root:p)
    spec=dict(mode='selection',checkpoint_id='vanilla:512',dataset='selection',adapter=dummy,items=ref(items),requests=ref(requests),decoding=p['decoding'],directory=str(out/'selection/vanilla_512/selection'),config=dict(model='SYNTHETIC_NOT_A_MODEL',engine=engine),runtime_environment={},resume=False,technical_retry=False)
    if mutation=='adapter':spec['adapter']=dict(dummy,sha256='wrong')
    elif mutation=='decoding':spec['decoding']={'n':4}
    elif mutation=='requests':spec['requests']=dummy
    elif mutation=='engine':spec['config']['engine']={'max_model_len':9999}
    elif mutation=='grpo':spec['checkpoint_id']='grpo:512'
    elif mutation=='prior_attempt':put(Path(spec['directory'])/'synthetic/reservation.json',{})
    if mutation=='valid':assert g.worker_spec_gate(spec,root)
    else:
        with pytest.raises(PermissionError):g.worker_spec_gate(spec,root)
