#!/usr/bin/env python3
"""Scoped matched-schedule/random-selection intervention on the frozen online engine."""
import hashlib
import os
from pathlib import Path
import runpy
import signal
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now

PROTOCOL=ROOT/'experiments/stage5/aux_random3x_protocol_v2.json'


def selected_encounters(encounters,window_index,seed=20260914):
    """Selection accepts only pre-generation identities; outcomes cannot enter its API."""
    assert len(encounters)>=8
    identities=[(e['encounter_index'],e['prompt_id']) for e in encounters]
    assert len(set(identities))==len(identities)
    return sorted(identities,key=lambda e:hashlib.sha256(f'{seed}:{window_index}:{e[0]}:{e[1]}'.encode()).hexdigest())[:8]


def technical_valid(g,policy):
    rows=g['responses']
    return (len(rows)==4 and len({r['trajectory_id'] for r in rows})==4 and
        sorted(r['member_index'] for r in rows)==[0,1,2,3] and
        all(r['group_id']==g['group_id'] and r['prompt_id']==g['prompt_id'] and r['policy_version']==policy and
            r['finish_reason'] in ('stop','length') and not r.get('transport_error') for r in rows))


class ScheduleSelector:
    def __init__(self,schedule,seed=20260914):
        self.schedule=schedule;self.seed=seed;self.buffer={}
        self.lookup={e['encounter_index']:(i,e) for i,w in enumerate(schedule) for e in w['encounters']}

    def __call__(self,groups,policy,mode,already=0,target=8):
        from medical_posttrain.sampling.dynamic import classify
        assert mode=='random3x' and target==8 and already==0
        ix=self.lookup[groups[0]['encounter_index']][0]
        w=self.schedule[ix];rank=selected_encounters(w['encounters'],ix,self.seed);chosen=set(rank)
        decisions=[]
        for g in groups:
            gi=g['encounter_index'];wi,expected=self.lookup[gi]
            assert wi==ix and all(g[k]==expected[k] for k in ['prompt_id','encounter_index','candidate_epoch','candidate_cursor','request_seed'])
            assert technical_valid(g,policy),'Invalid/incomplete transport: fail closed, no automatic retry or refill'
            assert gi not in self.buffer,'Duplicate generation encounter'
            self.buffer[gi]=g
            # Logging classification is downstream of immutable identity-only ranking.
            cls=classify(g['responses'],policy);assert cls['valid']
            selected=(gi,g['prompt_id']) in chosen
            decisions.append(dict(group_id=g['group_id'],prompt_id=g['prompt_id'],encounter_index=gi,
                disposition='selected' if selected else 'random_not_selected',**cls))
        if groups[-1]['encounter_index']!=w['encounters'][-1]['encounter_index']:
            return [],decisions
        assert set(self.buffer)=={e['encounter_index'] for e in w['encounters']}
        selected=[self.buffer[gi] for gi,pid in rank]
        self.buffer={}
        return selected,decisions


def check():
    from frontier_diagnostic import gates
    gates();p=read(PROTOCOL)
    assert p['status']=='FROZEN_BEFORE_GENERATION'
    assert p==read(Path(p['artifact_path'])/'auxiliary_protocol.json')
    assert record(p['run_config']['path'])==p['run_config']
    assert record(p['selected_identities']['path'])==p['selected_identities']
    for ref in p['runtime_sources'].values():assert record(ref['path'])==ref
    assert record(p['schedule']['path'])==p['schedule']
    return p


def resume_continuity(pause,restarted,committed):
    assert pause['state']==restarted['state']
    assert pause['next_encounters']==restarted['next_encounters']
    assert committed['state_before']==pause['state']
    assert committed['state_after']['optimizer_steps']==pause['state']['optimizer_steps']+2
    assert committed['state_after']['training_groups']==pause['state']['training_groups']+8


def install():
    p=check()
    from medical_posttrain.rl import online
    from stage4_resume_boundary import restore_boundary
    from stage4_gpu_release_guard import install as guard
    online.select_groups=ScheduleSelector(read(p['schedule']['path'])['windows'],p['selection_seed'])
    online.restore=restore_boundary
    online.monitor=lambda *args,**kwargs:None
    guard()


def launch(out):
    check();out=Path(out)
    used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
    assert used<100,f'GPU occupied ({used}MiB); do not kill unrelated processes'
    assert read(out/'status.json')['status'] in ('PREPARED','INTERRUPTED')
    attempt=out/f'attempt_{len(list(out.glob("attempt_*")))+1:03d}';attempt.mkdir()
    cmd=[sys.executable,str(Path(__file__)),'worker','--run',str(out),'--attempt',str(attempt)]
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    with (attempt/'stdout.log').open('x') as so,(attempt/'stderr.log').open('x') as se:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,env=env)
    immutable(attempt/'launch.json',dict(pid=proc.pid,command=cmd,timestamp=now(),protocol=record(PROTOCOL)))
    return proc


def supervise(out):
    p=check();out=Path(out)
    idx=ROOT/'experiments/stage5'
    try:
        old=launch(out)
        while not (out/'pause_ready.json').exists():
            assert old.poll() is None,'Worker failed before real resume boundary'
            durable(idx/'aux_random3x_status.json',dict(status='RUNNING_INTEGRATION',pid=old.pid,run_id=out.name,timestamp=now()))
            time.sleep(15)
        pause=read(out/'pause_ready.json');assert pause['state']['training_groups']==32 and pause['state']['optimizer_steps']==8
        assert pause['pid']==old.pid and os.getpgid(old.pid)==old.pid
        immutable(out/'termination.json',dict(timestamp=now(),pid=old.pid,signal='SIGTERM',reason='Preregistered fresh-process resume after32groups',state=pause['state']))
        os.killpg(old.pid,signal.SIGTERM);old.wait(timeout=60)
        assert not Path(f'/proc/{old.pid}').exists()
        immutable(out/'termination_observed.json',dict(dead=True,pid=old.pid,timestamp=now(),exit_code=old.returncode))
        durable(out/'status.json',dict(status='INTERRUPTED',timestamp=now()))
        # A real 32-group integration replay uses the same strict final verifier primitives.
        from verify_random3x import verify_run
        integration=verify_run(out,4,require_final=False)
        immutable(idx/'aux_random3x_integration_v1.json',integration)
        deadline=time.monotonic()+60
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=100:
            assert time.monotonic()<deadline;time.sleep(.2)
        new=launch(out)
        while not (out/'windows/0004/commit.json').exists():
            assert new.poll() is None,'Fresh resumed worker failed'
            time.sleep(15)
        restarted=read(out/'attempt_002/resume_start.json')
        assert restarted['state']==pause['state'] and restarted['next_encounters']==pause['next_encounters']
        native=read(out/'windows/0004/actor/resume_receipt.json');assert native['result']=='PASS'
        committed=read(out/'windows/0004/commit.json')
        resume_continuity(pause,restarted,committed)
        assert committed['state_before']==pause['state'] and committed['state_after']['optimizer_steps']==10
        receipt=dict(result='PASS',timestamp=now(),old_pid=old.pid,new_pid=new.pid,
            restored_groups=32,restored_optimizer_steps=8,next_committed_optimizer_steps=10,optimizer_replayed=False,
            controller_rng_scheduler_optimizer_continuous=True,pause=record(out/'pause_ready.json'),
            restart=record(out/'attempt_002/resume_start.json'),native_resume=record(out/'windows/0004/actor/resume_receipt.json'),
            frozen_selection_schedule=record(p['schedule']['path']))
        immutable(out/'physical_resume_receipt.json',receipt)
        immutable(idx/'aux_random3x_resume_v1.json',receipt)
        while new.poll() is None:
            state=read(out/'checkpoint.json')['state']
            durable(idx/'aux_random3x_status.json',dict(status='RUNNING',pid=new.pid,run_id=out.name,
                training_groups=state['training_groups'],generated_groups=state['generated_groups'],timestamp=now()))
            time.sleep(15)
        assert new.returncode==0 and read(out/'status.json')['status']=='FULL_PASS'
        durable(idx/'aux_random3x_status.json',dict(status='RAW_VERIFYING',run_id=out.name,timestamp=now()))
        verified=verify_run(out,64,require_final=True)
        immutable(idx/'aux_random3x_verification_v1.json',verified)
        subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_random3x.py'),'--run',str(out)],cwd=ROOT,check=True)
        subprocess.run([sys.executable,str(ROOT/'scripts/finish_random3x.py'),'--run',str(out)],cwd=ROOT,check=True)
        durable(idx/'aux_random3x_status.json',dict(status='AUXILIARY_VERIFIED_AND_ANALYZED',run_id=out.name,timestamp=now(),READY_FOR_STAGE5='NO'))
    except BaseException as exc:
        import traceback
        immutable(out/f'supervisor_failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
        durable(idx/'aux_random3x_status.json',dict(status='FAILED',error=repr(exc),run_id=out.name,timestamp=now(),READY_FOR_STAGE5='NO'))
        raise


if __name__=='__main__':
    if sys.argv[1]=='worker':
        install()
        sys.argv[0]=str(ROOT/'scripts/run_stage4.py')
        runpy.run_path(sys.argv[0],run_name='__main__')
    else:supervise(Path(sys.argv[sys.argv.index('--run')+1]))
