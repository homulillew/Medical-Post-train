#!/usr/bin/env python3
"""Owner-authorized closure, frozen selection, once-only final exams and local QA."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read,ref,check_ref,freeze
from medical_posttrain.evaluation.gates import stage4_gate,selection_gate,final_gate,gpu_guard,verify_seal
from medical_posttrain.rl.common import durable
from stage5_objective_evidence import put,roles,verify_selection,objective_results,open_evidence,retain_cases,EXAMS,OPEN

IDX=ROOT/'experiments/stage5'
RUN=IDX/'execution_20260915_v1'
OUT=Path('/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1')


def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()


def event(name,**data):
    row=dict(event=name,timestamp=now(),**data)
    with (RUN/'events.jsonl').open('a') as f:
        f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
    print(json.dumps(row,ensure_ascii=False),flush=True)


def state(status,**data):
    p=ROOT/'project_state.json';s=read(p)
    assert s['stages']['4']['status']=='DONE' and s['stages']['6']['status']=='NOT_STARTED'
    s['stages']['5'].update(status=status,**data);durable(p,s)


def wait_for_gpu():
    while True:
        try:
            gpu_guard();return
        except PermissionError as exc:
            if 'owned by an active process' not in str(exc):raise
            durable(RUN/'status.json',dict(status='WAITING_FOR_EXCLUSIVE_GPU',timestamp=now()))
            time.sleep(15)


def commit(message,paths):
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode==0
    existing=[str(p) for p in paths if Path(p).exists()]
    subprocess.run(['git','add','--',*existing],cwd=ROOT,check=True)
    if subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        subprocess.run(['git','commit','-m',message],cwd=ROOT,check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    # Publication is separate from scientific acceptance; retain auth failures without discarding results.
    result=subprocess.run(['git','push','origin','main'],cwd=ROOT,text=True,capture_output=True)
    event('publication',head=head,result='PASS' if result.returncode==0 else 'FAILED',detail=result.stderr[-1800:])
    return head


def close_stage4():
    receipt=ROOT/'experiments/stage4/verification-final.json'
    if read(ROOT/'project_state.json')['stages']['4']['status']=='DONE':
        stage4_gate();return
    launch=read(ROOT/'experiments/stage4/owner_closure_full_v1/launch.json')
    while not receipt.exists():
        proc=Path('/proc')/str(launch['pid'])
        assert proc.exists() and b'verify_stage.py' in (proc/'cmdline').read_bytes(), 'Full verifier exited without a receipt'
        durable(RUN/'status.json',dict(status='STAGE4_FULL_VERIFYING',timestamp=now(),pid=launch['pid']))
        time.sleep(15)
    verified=read(receipt)
    assert verified['result']=='PASS' and verified['scope']=='FULL' and verified['READY_FOR_STAGE5']=='YES', verified.get('errors')
    proc=Path('/proc')/str(launch['pid'])
    while proc.exists():
        try:running=b'verify_stage.py' in (proc/'cmdline').read_bytes()
        except FileNotFoundError:break
        if not running:break
        time.sleep(1)
    check_ref(read(ROOT/'experiments/stage4/manual_review_waiver_verifier_change_v1.json')['new_verifier'])
    p=ROOT/'project_state.json';s=read(p)
    assert s['stages']['5']['status']==s['stages']['6']['status']=='NOT_STARTED'
    s['stages']['4'].update(status='DONE',verification_receipt=ref(receipt),stage_report='docs/stage_reports/04_gspo.md',
        interview_story='docs/stage_reports/04_gspo_interview_story.md',human_review_status='OWNER_WAIVED',clinical_validation=False,READY_FOR_STAGE5='YES')
    durable(p,s);stage4_gate()
    closure=ROOT/'experiments/stage4'
    files=[p,ROOT/'scripts/verify_stage4.py',ROOT/'tests/test_stage4_owner_waiver.py',ROOT/'docs/implementation/STAGE4_CLOSURE_STAGE5_EXECUTION.md',
        ROOT/'docs/stage_reports/04_gspo.md',ROOT/'docs/stage_reports/04_gspo_interview_story.md',closure/'closure_source_snapshots',closure/'deliverables.json',
        closure/'manual_review_waiver_v1.json',closure/'manual_review_waiver_verifier_change_v1.json',closure/'owner_closure_start_audit_v1.json',
        closure/'owner_waiver_tests_v1.xml',closure/'owner_closure_full_v1',receipt]
    head=commit('chore: close stage4 with owner-waived manual review',files)
    put(RUN/'stage4_closure_commit.json',dict(commit=head,receipt=ref(receipt)))
    event('stage4_done',commit=head,full_verifier='PASS',human_reviews_completed=0)


def child(cmd,label,directory=None):
    out=RUN/'processes'/f'{label}_{time.time_ns()}';out.mkdir(parents=True)
    env=dict(os.environ,PYTHONPATH='src:scripts',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
    env.pop('CUDA_VISIBLE_DEVICES',None)
    with (out/'stdout.log').open('x') as so,(out/'stderr.log').open('x') as se:
        p=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se,env=env,start_new_session=True)
    freeze(out/'launch.json',dict(command=cmd,pid=p.pid,timestamp=now(),head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    while p.poll() is None:
        if directory:
            completed=len(list(Path(directory).glob('*/response.json')))
        else:
            completed=sum(1 for _ in (OUT/'selection').glob('*/selection/*/response.json'))
        durable(RUN/'status.json',dict(status='RUNNING',phase=label,pid=p.pid,completed_responses=completed,timestamp=now()))
        time.sleep(15)
    freeze(out/'exit.json',dict(exit_code=p.returncode,timestamp=now()))
    assert p.returncode==0,f'{label} failed; inspect {out}/stderr.log; no silent retry'
    event('process_completed',phase=label,process_record=str(out))


def final_jobs(protocol,selected):
    adapters={'sft':protocol['fixed_sft']['adapter']}
    for c in list(selected['selected_checkpoints'].values())+list(protocol['scientific_endpoints'].values()):
        if c['checkpoint_id'] in adapters:assert adapters[c['checkpoint_id']]==c['adapter']
        adapters[c['checkpoint_id']]=c['adapter']
    primary=['sft']+[c['checkpoint_id'] for c in selected['selected_checkpoints'].values()]
    exams=[(cid,adapters[cid],dataset) for dataset in EXAMS for cid in adapters]
    opened=[(cid,adapters[cid],dataset) for dataset in OPEN for cid in primary]
    return exams,opened


def generate_jobs(jobs,resume,technical_retry):
    p=selection_gate();cfg=read(ROOT/'configs/stages/s4_formal_shared.json');op=read(IDX/'open_qa_eval_protocol_v1.json')
    with (ROOT.parent/'stage4-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for cid,adapter,dataset in jobs:
            final_gate(cid);wait_for_gpu();out=OUT/'final'/cid.replace(':','_')/dataset
            if (out/'complete.json').exists():
                assert resume, 'Existing completed output requires explicit resume'
                event('reused_complete',checkpoint_id=cid,dataset=dataset);continue
            m=read(IDX/'manifests'/f'{dataset}.json');exam=dataset in EXAMS;config=dict(cfg,engine=dict(cfg['engine']))
            if not exam:config['engine']['max_model_len']=op['max_model_len']
            spec=dict(checkpoint_id=cid,adapter=adapter,dataset=dataset,items=m['data'],requests=m['requests'],exam=exam,
                decoding=p['decoding'] if exam else op['decoding'],mode='final',config=config,runtime_environment=p['runtime']['environment'],
                directory=str(out),run_id='stage5_project_v1',resume=resume,technical_retry=technical_retry)
            sp=out/f'execution_spec_{time.time_ns()}.json';freeze(sp,spec)
            child([sys.executable,str(ROOT/'scripts/stage5_model_worker.py'),'--spec',str(sp)],dataset+'_'+cid.replace(':','_'),out)


def generate_selection(protocol,resume,technical_retry):
    from run_stage5_eval import write_jsonl,freeze_selection
    from stage5_selection_source import selection_inputs
    items,requests=selection_inputs(protocol)
    ip=OUT/'selection_inputs/items.jsonl';rp=OUT/'selection_inputs/requests.jsonl'
    write_jsonl(ip,items);write_jsonl(rp,requests)
    cfg=read(ROOT/'configs/stages/s4_formal_shared.json')
    with (ROOT.parent/'stage4-gpu.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for arm in ['vanilla','dynamic']:
            for c in protocol['candidates'][arm]:
                selection_gate();wait_for_gpu();cid=c['checkpoint_id'];out=OUT/'selection'/cid.replace(':','_')/'selection'
                if (out/'complete.json').exists():
                    assert resume,'Existing selection requires explicit resume'
                    event('reused_complete',checkpoint_id=cid,dataset='selection');continue
                spec=dict(checkpoint_id=cid,adapter=c['adapter'],dataset='selection',items=ref(ip),requests=ref(rp),exam=True,
                    decoding=protocol['decoding'],mode='selection',config=cfg,runtime_environment=protocol['runtime']['environment'],
                    directory=str(out),run_id='stage5_project_v1',resume=resume,technical_retry=technical_retry)
                sp=out/f'execution_spec_{time.time_ns()}.json';freeze(sp,spec)
                child([sys.executable,str(ROOT/'scripts/stage5_model_worker.py'),'--spec',str(sp)],'selection_'+cid.replace(':','_'),out)
    freeze_selection(protocol,OUT)


def final_handoff():
    primary,endpoints=roles();cm=read(IDX/'cmexam_final_results_v1.json');cmb=read(IDX/'cmb_final_results_v1.json')
    report=ROOT/'docs/stage_reports/05_objective_evaluation.md'
    lines=['# Stage 5 objective evaluation and local open QA','',
        'Status: OBJECTIVE_EVAL_PASS; OPEN_QA_JUDGE_PENDING. Stage 5 is not DONE. No paid judge was called; no human or clinical review was fabricated.','',
        '| Model | CMExam 6809 | CMB 2000 | CMB medical-only 1929 |','|---|---:|---:|---:|']
    for role in primary:
        values=[cm['primary']['summary'][role],cmb['primary']['summary'][role],cmb['secondary']['primary']['summary'][role]]
        lines.append('| '+role+' | '+' | '.join(f"{s['correct_count']}/{s['n']} ({s['accuracy']*100:.4f}%)" for s in values)+' |')
    lines+=['','Scientific equal-update endpoints are reported separately in the JSON results; selected checkpoints were frozen using validation only.','']
    lines+=['| Scientific endpoint | CMExam 6809 | CMB 2000 |','|---|---:|---:|']
    for role in endpoints:
        values=[cm['scientific_endpoints']['summary'][role],cmb['scientific_endpoints']['summary'][role]]
        lines.append('| '+role+'5000 | '+' | '.join(f"{s['correct_count']}/{s['n']} ({s['accuracy']*100:.4f}%)" for s in values)+' |')
    lines.append('')
    for label,result in [('CMExam',cm),('CMB',cmb),('CMB medical-only',cmb['secondary'])]:
        for name,q in result['primary']['paired'].items():
            lines.append(f"- {label}, {name}: {q['delta_pp']:+.4f} pp; 95% CI {q['ci95_pp']}; McNemar p={q['exact_mcnemar']:.6g}; W→C={q['wrong_to_correct']}, C→W={q['correct_to_wrong']}.")
    lines+=['','All slices are exploratory, with fixed source-defined membership and unadjusted intervals. Single training seed limits generalization. Safety files are automated review candidates, not clinical validation. All candidate regressions are retained. Stage 6 remains NOT_STARTED.','']
    text='\n'.join(lines)
    if report.exists():assert report.read_text()==text
    else:report.write_text(text)
    names=['selection_results_v1.json','selected_checkpoints_v1.json','cmexam_final_results_v1.json','cmb_final_results_v1.json','open_qa_local_generations_v1.json',
        'open_qa_blind_bundle_v1.json','safety_local_results_v1.json','stage5_objective_verification_v1.json','stage5_case_candidates_v1.json']
    put(ROOT/'experiments/handoffs/stage5_results_to_chatgpt_v1.json',dict(status='OBJECTIVE_EVAL_PASS',open_qa_judge_status='OPEN_QA_JUDGE_PENDING',
        stage4_full_verifier=ref(ROOT/'experiments/stage4/verification-final.json'),selected=read(IDX/'selected_checkpoints_v1.json'),
        cmexam=cm,cmb=cmb,evidence={n:ref(IDX/n) for n in names},report=ref(report),human_reviews_completed=0,clinical_validation=False,paid_api_calls=0,
        training_efficiency_evidence=[ref(ROOT/'experiments/stage4'/n) for n in ['formal_plots/manifest.json','postformal_analysis_v1.json','signal_density_analysis_v1.json','loss_objective_ablation_analysis_v1.json']],
        unsupported_claims=['No clinically validated safety','No open QA preference/rubric scores without judge','No multi-seed robustness','No test-based checkpoint selection'],stage6='NOT_STARTED'))
    state('FULL_RUNNING',objective_evaluation_status='OBJECTIVE_EVAL_PASS',open_qa_status='OPEN_QA_JUDGE_PENDING',local_generation_status='COMPLETE',
        objective_verification_receipt=ref(IDX/'stage5_objective_verification_v1.json'),stage_report=str(report.relative_to(ROOT)),execution_status='LOCAL_TASK_COMPLETE')
    files=[IDX/n for n in names]+[IDX/'selection_result_manifest_v1.json',IDX/'selection_result_lock_v1.json',IDX/'selection_freeze_event_v1.json',
        ROOT/'experiments/handoffs/stage5_results_to_chatgpt_v1.json',report,ROOT/'project_state.json',RUN]
    head=commit('handoff: retain stage5 evaluation evidence',files)
    durable(RUN/'status.json',dict(status='LOCAL_TASK_COMPLETE',objective='OBJECTIVE_EVAL_PASS',open_qa='OPEN_QA_JUDGE_PENDING',commit=head,timestamp=now()))
    event('local_task_complete',commit=head)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--resume',action='store_true');ap.add_argument('--technical-retry',action='store_true');args=ap.parse_args()
    RUN.mkdir(parents=True,exist_ok=True)
    with (RUN/'supervisor.lock').open('a') as execution_lock:
        fcntl.flock(execution_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            close_stage4()
            from run_stage5_eval import validate_manifests
            validate_manifests();verify_seal();p=selection_gate()
            source=read(RUN/'source_manifest.json')
            for r in source['sources']:check_ref(r)
            check_ref(source['context_precheck'])
            assert read(source['context_precheck']['path'])['result']=='PASS'
            check_ref(source['tests'])
            if not (IDX/'selection_result_lock_v1.json').exists():
                if not args.resume:
                    from prepare_stage5_preflight import usage_inventory
                    usage=usage_inventory();assert usage['selection1024_generations']==usage['cmexam_cmb_project_final_generations']==0
                commit('eval: prepare guarded objective evaluation execution',[ROOT/'scripts/execute_stage5_closure.py',ROOT/'scripts/stage5_objective_evidence.py',ROOT/'scripts/stage5_selection_source.py',ROOT/'tests/test_stage5_execution.py',RUN])
                wait_for_gpu();state('SELECTION_RUNNING',run_ids=['stage5_project_v1'],execution_status='RUNNING')
                generate_selection(p,args.resume,args.technical_retry)
            put(IDX/'selection_freeze_event_v1.json',read(IDX/'selection_freeze_event_v1.json') if (IDX/'selection_freeze_event_v1.json').exists() else dict(timestamp=now(),unix_time=time.time(),selection_lock=ref(IDX/'selection_result_lock_v1.json')))
            verify_selection(OUT)
            state('FULL_RUNNING',checkpoint_selection_status='FROZEN_SELECTED',execution_status='RUNNING',progress=dict(checkpoints_evaluated=20,selection_generations=20480))
            commit('eval: complete validation checkpoint selection',[ROOT/'project_state.json',IDX/'selection_results_v1.json',IDX/'selected_checkpoints_v1.json',IDX/'selection_result_manifest_v1.json',IDX/'selection_result_lock_v1.json',IDX/'selection_freeze_event_v1.json',RUN])
            selected=final_gate('sft');exams,opened=final_jobs(p,selected)
            generate_jobs(exams,args.resume,args.technical_retry)
            objective_results(OUT)
            state('FULL_RUNNING',objective_evaluation_status='OBJECTIVE_EVAL_PASS',open_qa_status='LOCAL_GENERATION_RUNNING')
            commit('eval: complete held-out medical benchmarks',[ROOT/'project_state.json',IDX/'cmexam_final_results_v1.json',IDX/'cmb_final_results_v1.json',IDX/'stage5_objective_verification_v1.json',RUN])
            generate_jobs(opened,args.resume,args.technical_retry)
            open_evidence(OUT)
            commit('eval: retain open medical QA generations',[IDX/'open_qa_local_generations_v1.json',IDX/'open_qa_blind_bundle_v1.json',IDX/'safety_local_results_v1.json',RUN])
            retain_cases(OUT);final_handoff()
        except BaseException as exc:
            failure=dict(timestamp=now(),error=repr(exc),traceback=traceback.format_exc(),silent_retry=False)
            freeze(RUN/f'failure_{time.time_ns()}.json',failure);durable(RUN/'status.json',dict(status='FAILED',**failure))
            current=read(ROOT/'project_state.json')
            if current['stages']['5']['status']!='NOT_STARTED':
                current['stages']['5'].update(execution_status='FAILED',execution_failure=str(RUN))
                durable(ROOT/'project_state.json',current)
            event('failed',error=repr(exc));raise


if __name__=='__main__':main()
