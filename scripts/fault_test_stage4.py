#!/usr/bin/env python3
"""Three real GPU transaction crashes. A detached supervisor owns only its runs."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,immutable,durable,inherited,prepare,now
from medical_posttrain.rl.transactions import cost_ledger

SCENARIOS=('before_checkpoint_rename','after_rename_before_pointer','after_optimizer_before_checkpoint')


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def wait_for(predicate,out,phase,seconds=5400):
    start=time.monotonic()
    while not predicate():
        status=read(out/'status.json')['status']
        assert status not in ('FAILED','BLOCKED','PAUSED_DIAGNOSTIC'), f'{out}: {status}'
        assert time.monotonic()-start<seconds, f'Timeout in diagnostic {phase}; run retained'
        durable(INDEX/'fault_queue_status.json',dict(status='RUNNING',phase=phase,run_id=out.name,pid=os.getpid(),timestamp=now()))
        time.sleep(5)


def launch(out):
    subprocess.run([sys.executable,str(ROOT/'scripts/run_stage4.py'),'launch','--run',str(out)],check=True)
    return read(sorted(out.glob('attempt_*'))[-1]/'launch.json')['pid']


def snapshot(out,name):
    w=out/'windows/0001'
    # Hash immutable checkpoints, raw batches and optimizer evidence before kill.
    tree=[]
    for base in (w/'checkpoint',w/'.tmp-checkpoint',w/'update',w/'actor'):
        if base.exists():tree.extend(record(p) for p in sorted(base.rglob('*')) if p.is_file())
    value=dict(timestamp=now(),state=read(out/'checkpoint.json')['state'],cost=cost_ledger(out),transaction_tree=tree)
    immutable(out/'fault'/f'{name}.json',value)
    return value


def one(out,scenario):
    old=launch(out)
    wait_for(lambda:(out/'fault/ready.json').exists(),out,'WAIT_REAL_FAULT_BARRIER')
    ready=read(out/'fault/ready.json');w=out/'windows/0001'
    assert ready['scenario']==scenario and ready['process_group']==old
    assert alive(old) and alive(ready['pid']) and os.getpgid(old)==old
    cmd=Path(f'/proc/{old}/cmdline').read_bytes().decode().replace('\0',' ')
    assert 'run_stage4.py worker' in cmd and str(out) in cmd
    before=snapshot(out,'before_kill')
    assert before['state']['training_groups']==8 and before['state']['optimizer_steps']==2
    expected_steps=3 if scenario==SCENARIOS[2] else 4
    assert before['cost']['physical_optimizer_steps']==expected_steps
    assert not (w/'commit.json').exists()
    if scenario==SCENARIOS[0]:assert (w/'.tmp-checkpoint/COMMITTED.json').exists() and not (w/'checkpoint').exists()
    if scenario==SCENARIOS[1]:assert (w/'checkpoint/COMMITTED.json').exists()
    if scenario==SCENARIOS[2]:
        assert not (w/'checkpoint').exists()
        assert ready['trainable_digest']!=read(out/'windows/0000/checkpoint/COMMITTED.json')['trainable_digest']
        assert ready['optimizer_digest']!=read(out/'windows/0000/checkpoint/COMMITTED.json')['optimizer_digest']
    owned=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            pid=int(p.name)
            if os.getpgid(pid)==old:owned.append(dict(pid=pid,cmdline=(p/'cmdline').read_bytes().decode().replace('\0',' ')))
        except (ProcessLookupError,FileNotFoundError):pass
    immutable(out/'fault/termination.json',dict(timestamp=now(),old_pid=old,actor_pid=ready['pid'],signal='SIGKILL',processes=owned,cmdline=cmd))
    os.killpg(old,signal.SIGKILL)
    deadline=time.monotonic()+120
    while any(alive(x['pid']) for x in owned) or int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=100:
        assert time.monotonic()<deadline,'Old process/GPU failed to exit'
        time.sleep(2)
    immutable(out/'fault/termination_observed.json',dict(timestamp=now(),dead=True,pids=[p['pid'] for p in owned]))
    durable(out/'status.json',dict(status='INTERRUPTED',timestamp=now()))
    from medical_posttrain.rl.recovery import recover_actor_transaction
    recovered=recover_actor_transaction(out)
    assert recovered['state']==before['state']
    after=snapshot(out,'after_recovery')
    for k,v in before['cost'].items():
        if isinstance(v,(int,float)):assert after['cost'][k]==v,(k,v,after['cost'][k])
    new=launch(out);assert new!=old
    wait_for(lambda:read(out/'status.json')['status']=='RECOVERY_TEST_PASS',out,'WAIT_RESUMED_COMPLETION')
    wait_for(lambda:not alive(new),out,'WAIT_OWNER_EXIT')
    final=read(out/'summary.json')['state'];cost=cost_ledger(out)
    assert (final['training_groups'],final['policy_windows'],final['optimizer_steps'])==(24,3,6)
    assert read(out/'attempt_002/resume_start.json')['state']==before['state']
    assert read(out/'attempt_002/resume_start.json')['next_encounters']==[
        # The paused window's retained stream encounters are the actual next work.
        {k:v for k,v in e.items()} for e in read(w/'batches/000/reservation.json')['encounters']]
    assert cost['physical_optimizer_steps']==(6 if scenario==SCENARIOS[1] else 6+expected_steps-2)
    assert cost['known_generated_groups']==final['generated_groups']
    assert cost['known_prompt_tokens']==final['prompt_tokens'] and cost['known_output_tokens']==final['output_tokens']
    for k,v in before['cost'].items():
        if isinstance(v,(int,float)):assert cost[k]>=v
    if scenario==SCENARIOS[1]:
        marker=read(w/'checkpoint/COMMITTED.json');adopt=read(w/'recovered_actor/result.json')
        assert record(w/'checkpoint/COMMITTED.json')==ready['marker']
        assert adopt['optimizer_steps_added']==0 and adopt['identity']['optimizer_digest']==marker['optimizer_digest']
    else:
        receipt=read(w/'actor/resume_receipt.json')
        assert receipt['restored_state']==before['state'] and receipt['restored_rng_digest']
    from verify_stage4 import online_run
    raw=online_run(out,'recovery_test')
    immutable(out/'fault/raw_verification.json',raw)
    immutable(out/'fault/final_cost.json',cost)
    sources=[record(p) for p in sorted((out/'fault').glob('*.json'))]
    sources += [record(out/'attempt_002/resume_start.json'),record(w/'checkpoint/COMMITTED.json'),
                record(w/'recovered_actor/result.json' if scenario==SCENARIOS[1] else w/'actor/resume_receipt.json')]
    result=dict(result='PASS',scenario=scenario,run_id=out.name,run_class='DIAGNOSTIC',formal_training_groups=0,
        real_process_termination=True,old_pid=old,new_pid=new,committed_budget_continuous=True,cost_not_rolled_back=True,
        effective_before=before['state'],effective_final=final,physical_optimizer_steps=cost['physical_optimizer_steps'],sources=sources,timestamp=now())
    immutable(out/'fault/result.json',result)
    immutable(INDEX/out.name/'fault_result.json',result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','run']);a=parser.parse_args()
    if a.action=='prepare':
        assert not (INDEX/'fault_test_plan.json').exists()
        entries={}
        for i,s in enumerate(SCENARIOS):
            cfg=inherited();cfg.update(mode='recovery_test',sampling_mode='dynamic' if i==1 else 'vanilla',
                stream_domain='stage4:transaction:'+s,target_training_groups=24,fault_injection=dict(scenario=s,window=1))
            out=prepare('recovery_'+chr(97+i),'DIAGNOSTIC',cfg)
            entries[s]=dict(run_id=out.name,path=str(out),manifest=record(out/'manifest.json'))
        immutable(INDEX/'fault_test_plan.json',dict(scenarios=entries,created_at=now(),formal_training_groups=0))
    else:
        lock=(INDEX/'fault_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=read(INDEX/'fault_test_plan.json');results={}
        try:
            for scenario,entry in plan['scenarios'].items():
                out=Path(entry['path'])
                results[scenario]=one(out,scenario)
            immutable(INDEX/'recovery_fault_injections.json',dict(result='PASS',scenarios=results,timestamp=now()))
            durable(INDEX/'fault_queue_status.json',dict(status='ALL_THREE_REAL_FAULTS_PASS',timestamp=now()))
        except BaseException as exc:
            immutable(INDEX/f'fault_queue_failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
            durable(INDEX/'fault_queue_status.json',dict(status='FAILED',error=repr(exc),timestamp=now()))
            raise


if __name__=='__main__':main()
