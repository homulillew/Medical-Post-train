#!/usr/bin/env python3
"""Sealed Stage5 planner and future guarded execution; dry-run stays CPU-only."""
import argparse,sys,subprocess,os,time,csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze,digest
from medical_posttrain.evaluation.gates import selection_gate,final_gate,gpu_guard,verify_seal
IDX=ROOT/'experiments/stage5'

def validate_manifests():
    index=read(IDX/'checkpoint_candidate_index_v1.json');assert index['candidate_count']==20
    for c in index['candidates']:check_ref(c['adapter']);check_ref(c['source_commit'])
    for r in read(IDX/'preflight_input_audit_v1.json')['protected_refs']:
        if Path(r['path'])!=ROOT/'project_state.json':check_ref(r)
    med=read(IDX/'cmb_medical_only_manifest_v1.json');primary=read(med['primary']['path'])
    assert len(med['ids'])==med['count'] and set(med['ids'])<set(primary['ids']) and digest(med['ids'])==med['ids_sha256']
    assert set(med['ids']).isdisjoint(med['excluded_ids']) and set(med['ids'])|set(med['excluded_ids'])==set(primary['ids'])
    return dict(result='PREFLIGHT_PASS',candidate_count=20,selection_generations=0,final_generations=0,gpu_model_loads=0)

def write_jsonl(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if rows(path)!=data:raise ValueError('Frozen prepared dataset differs')
        return
    with path.open('x') as f:
        for r in data:f.write(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n')

def execute(mode,resume=False,technical_retry=False):
    import fcntl
    p=selection_gate();verify_seal();root=Path(read(IDX/'selection_execution_plan_v1.json')['artifact_root'])
    if mode=='final':final_gate('sft')
    gpu_guard()
    with (ROOT.parent/'stage4-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);gpu_guard()
        cfg=read(ROOT/'configs/stages/s4_formal_shared.json');jobs=[]
        if mode=='selection':
            if (IDX/'selection_result_manifest_v1.json').exists():raise PermissionError('Selection already frozen; no rerun')
            from medical_posttrain.data.exam import messages,options_schema,canonical_answer
            raw=list(csv.DictReader(open(p['dataset']['source_metadata']['path'])))
            items=[];requests=[]
            for pid in p['dataset']['prompt_ids']:
                row=raw[int(pid.rsplit(':',1)[1])];options=options_schema(row['Options']);answer=canonical_answer(row['Answer'],''.join(options))
                item=dict(id=pid,question=row['Question'],options=options,answer=answer)
                items.append(item);requests.append(dict(id=pid,messages=messages(dict(item,answer_set=answer))))
            ip=root/'selection_inputs/items.jsonl';rp=root/'selection_inputs/requests.jsonl';write_jsonl(ip,items);write_jsonl(rp,requests)
            for arm in ['vanilla','dynamic']:
                for c in p['candidates'][arm]:jobs.append(dict(checkpoint_id=c['checkpoint_id'],adapter=c['adapter'],dataset='selection',items=ref(ip),requests=ref(rp),exam=True,decoding=p['decoding']))
        else:
            selected=final_gate('sft');init=read(cfg['initialization']['path']);sft=ref(Path(init['adapter_path'])/'adapter_model.safetensors')
            policies={'sft':sft}
            for c in list(selected['selected_checkpoints'].values())+list(p['scientific_endpoints'].values()):policies[c['checkpoint_id']]=c['adapter']
            primary={'sft'}|{c['checkpoint_id'] for c in selected['selected_checkpoints'].values()}
            for name,adapter in policies.items():
                for dataset in ['cmexam_test_scorable','cmb_exam_clean_2000','cmb_clin','open_qa_retention_200']:
                    exam=dataset in ['cmexam_test_scorable','cmb_exam_clean_2000']
                    if not exam and name not in primary:continue
                    m=read(IDX/'manifests'/f'{dataset}.json');decoding=p['decoding'] if exam else read(IDX/'open_qa_eval_protocol_v1.json')['decoding']
                    jobs.append(dict(checkpoint_id=name,adapter=adapter,dataset=dataset,items=m['data'],requests=m['requests'],exam=exam,decoding=decoding))
        for j in jobs:
            if mode=='final':final_gate(j['checkpoint_id'])
            gpu_guard();out=root/mode/j['checkpoint_id'].replace(':','_')/j['dataset']
            if (out/'complete.json').exists() and not resume:raise PermissionError('Existing final/selection generation requires explicit resume')
            config=dict(cfg,engine=dict(cfg['engine']))
            if not j['exam']:config['engine']['max_model_len']=read(IDX/'open_qa_eval_protocol_v1.json')['max_model_len']
            spec=dict(j,mode=mode,config=config,runtime_environment=p['runtime']['environment'],directory=str(out),run_id='stage5_project_v1',resume=resume,technical_retry=technical_retry)
            sp=out/f'execution_spec_{time.time_ns()}.json';freeze(sp,spec)
            cmd=[sys.executable,str(ROOT/'scripts/stage5_model_worker.py'),'--spec',str(sp)]
            with sp.with_suffix('.stdout.log').open('x') as so,sp.with_suffix('.stderr.log').open('x') as se:
                subprocess.run(cmd,cwd=ROOT,stdout=so,stderr=se,check=True)
        if mode=='selection':freeze_selection(p,root)
        else:
            from analyze_stage5_eval import analyze
            analyze(root)

def freeze_selection(p,root):
    from select_stage5_checkpoint import rank_candidates
    selected={};rankings={};sources=[]
    for arm in ['vanilla','dynamic']:
        summaries=[]
        for c in p['candidates'][arm]:
            path=root/'selection'/c['checkpoint_id'].replace(':','_')/'selection/complete.json';d=read(path);s=d['summary']
            if d['prompt_ids']!=p['dataset']['prompt_ids'] or d['n']!=1024:raise ValueError('Selection IDs/count mismatch')
            for r in d['response_refs']:check_ref(r)
            summaries.append(dict(checkpoint_id=c['checkpoint_id'],training_groups=c['training_groups'],n=s['n'],correct=s['correct_count'],unparseable=s['unparseable_count'],truncated=s['truncation_count']));sources.append(ref(path))
        ranking=rank_candidates(summaries,p['candidates'][arm]);rankings[arm]=ranking
        selected[arm]=next(c for c in p['candidates'][arm] if c['checkpoint_id']==ranking['selected_checkpoint'])
    path=IDX/'selection_result_manifest_v1.json';freeze(path,dict(status='FROZEN_SELECTED',selected_checkpoints=selected,rankings=rankings,summary_sources=sources,protocol=ref(IDX/'checkpoint_selection_protocol_v1.json'),selection_only=True,test_used=False))
    freeze(IDX/'selection_result_lock_v1.json',dict(selection_result=ref(path),final_protocols=[ref(IDX/n) for n in ['cmexam_final_eval_protocol_v1.json','cmb_primary_audit_v1.json','open_qa_eval_protocol_v1.json']]))

def main():
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True)
    for x in ['dry-run','validate-manifests','selection-only','final-only']:g.add_argument('--'+x,action='store_true')
    a.add_argument('--resume',action='store_true');a.add_argument('--technical-retry',action='store_true');args=a.parse_args()
    if args.dry_run or args.validate_manifests:
        result=validate_manifests();result['plan']=read(IDX/'selection_execution_plan_v1.json');result['inference_executed']=False
        print(json.dumps({k:v for k,v in result.items() if k!='plan'},ensure_ascii=False));return
    execute('selection' if args.selection_only else 'final',args.resume,args.technical_retry)
if __name__=='__main__':main()
