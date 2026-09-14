#!/usr/bin/env python3
"""Scoped token-objective actor routing; frozen controller/sampling remain unchanged."""
import os,sys,subprocess,time,signal,runpy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now
PROTOCOL=ROOT/'experiments/stage4/grpo_objective_protocol_v1.json'

def check():
    p=read(PROTOCOL)
    assert p['status']=='FROZEN_BEFORE_AUXILIARY_TRAINING'
    for ref in p['runtime_sources'].values():assert record(ref['path'])==ref
    for ref in [p['evaluation_manifest'],p['schedule'],p['selection_protocol'],p['project_state']]:assert record(ref['path'])==ref
    for arm in p['runs'].values():assert record(arm['config']['path'])==arm['config']
    rel=str(PROTOCOL.relative_to(ROOT))
    commit=subprocess.check_output(['git','log','-1','--format=%H','--',rel],cwd=ROOT,text=True).strip();assert commit
    blob=subprocess.check_output(['git','show',commit+':'+rel],cwd=ROOT)
    assert __import__('hashlib').sha256(blob).hexdigest()==record(PROTOCOL)['sha256']
    return p

class RoutedPopen(subprocess.Popen):
    def __init__(self,args,*a,**kw):
        self.actor_directory=None;actual=list(args)
        if len(actual)>2 and Path(str(actual[1])).name=='run_stage4.py':
            assert actual[2] in ('actor-window','adopt-actor')
            self.actor_directory=Path(actual[actual.index('--directory')+1])
            actual[1]=str(Path(__file__));
        elif len(actual)>1 and Path(str(actual[1])).name=='reload_stage4.py':
            actual=[actual[0],str(Path(__file__)),'reload',*actual[2:]]
        super().__init__(actual,*a,**kw)
    def wait(self,timeout=None):
        code=super().wait(timeout=timeout)
        if code==0 and self.actor_directory and not getattr(self,'released',False):
            import pynvml
            from stage4_gpu_release_guard import wait_for_release
            self.released=True;pynvml.nvmlInit()
            try:
                h=pynvml.nvmlDeviceGetHandleByIndex(0)
                result=wait_for_release(lambda:pynvml.nvmlDeviceGetMemoryInfo(h).used)
                immutable(self.actor_directory/'gpu_release_observations'/f'{time.time_ns()}.json',dict(**result,timestamp=now(),actor_pid=self.pid,actual_command=self.args))
            finally:pynvml.nvmlShutdown()
        return code
class Proxy:
    Popen=RoutedPopen
    @staticmethod
    def call(args,**kwargs):
        with RoutedPopen(args,**kwargs) as p:return p.wait()
    def __getattr__(self,name):return getattr(subprocess,name)

def install():
    check()
    from medical_posttrain.rl import actor,online
    from loss_objective_actor import Actor
    from stage4_resume_boundary import restore_boundary
    actor.Actor=Actor;online.restore=restore_boundary;online.monitor=lambda *a,**kw:None;online.subprocess=Proxy()

def launch(out):
    check();out=Path(out)
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<100
    assert read(out/'status.json')['status'] in ('PREPARED','INTERRUPTED')
    attempt=out/f'attempt_{len(list(out.glob("attempt_*")))+1:03d}';attempt.mkdir()
    cmd=[sys.executable,str(Path(__file__)),'worker','--run',str(out),'--attempt',str(attempt)]
    env=dict(os.environ,TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
    with (attempt/'stdout.log').open('x') as so,(attempt/'stderr.log').open('x') as se:
        proc=subprocess.Popen(cmd,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,env=env)
    immutable(attempt/'launch.json',dict(pid=proc.pid,command=cmd,timestamp=now(),protocol=record(PROTOCOL),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    return proc

def supervise_run(name,out):
    from verify_loss_objective import verify_run
    idx=ROOT/'experiments/stage4';out=Path(out);old=launch(out)
    while not (out/'pause_ready.json').exists():
        assert old.poll() is None,f'{name} worker failed before resume'
        durable(idx/'loss_objective_status_v1.json',dict(status='TRAINING_INTEGRATION',arm=name,pid=old.pid,timestamp=now()))
        time.sleep(15)
    pause=read(out/'pause_ready.json');assert pause['state']['training_groups']==32 and pause['state']['optimizer_steps']==8
    assert pause['pid']==old.pid and os.getpgid(old.pid)==old.pid
    immutable(out/'termination.json',dict(timestamp=now(),pid=old.pid,signal='SIGTERM',reason='Preregistered physical resume at32groups',state=pause['state']))
    os.killpg(old.pid,signal.SIGTERM);old.wait(timeout=60);assert not Path(f'/proc/{old.pid}').exists()
    immutable(out/'termination_observed.json',dict(dead=True,pid=old.pid,timestamp=now(),exit_code=old.returncode))
    durable(out/'status.json',dict(status='INTERRUPTED',timestamp=now()))
    immutable(idx/f'grpo_{name}_integration_v1.json',verify_run(out,4,False))
    deadline=time.monotonic()+60
    while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=100:
        assert time.monotonic()<deadline;time.sleep(.5)
    new=launch(out)
    while not (out/'windows/0004/commit.json').exists():
        assert new.poll() is None,'Fresh resumed worker failed';time.sleep(15)
    restarted=read(out/'attempt_002/resume_start.json');committed=read(out/'windows/0004/commit.json')
    from random3x_runtime import resume_continuity
    resume_continuity(pause,restarted,committed)
    native=read(out/'windows/0004/actor/resume_receipt.json');assert native['result']=='PASS'
    receipt=dict(result='PASS',timestamp=now(),old_pid=old.pid,new_pid=new.pid,restored_groups=32,restored_optimizer_steps=8,next_committed_optimizer_steps=10,
        optimizer_replayed=False,controller_rng_scheduler_optimizer_continuous=True,pause=record(out/'pause_ready.json'),restart=record(out/'attempt_002/resume_start.json'),native_resume=record(out/'windows/0004/actor/resume_receipt.json'))
    immutable(out/'physical_resume_receipt.json',receipt);immutable(idx/f'grpo_{name}_resume_v1.json',receipt)
    while new.poll() is None:
        state=read(out/'checkpoint.json')['state']
        durable(idx/'loss_objective_status_v1.json',dict(status='TRAINING',arm=name,pid=new.pid,training_groups=state['training_groups'],generated_groups=state['generated_groups'],timestamp=now()))
        time.sleep(15)
    assert new.returncode==0 and read(out/'status.json')['status']=='FULL_PASS'
    durable(idx/'loss_objective_status_v1.json',dict(status='RAW_VERIFYING',arm=name,timestamp=now()))
    immutable(idx/f'grpo_{name}512_verification_v1.json',verify_run(out,64,True))

def main():
    if sys.argv[1]=='supervise':
        p=check()
        try:
            for name in ['vanilla','dynamic']:supervise_run(name,p['runs'][name]['path'])
            subprocess.run([sys.executable,str(ROOT/'scripts/evaluate_loss_objective.py')],cwd=ROOT,check=True)
            subprocess.run([sys.executable,str(ROOT/'scripts/analyze_loss_objective.py')],cwd=ROOT,check=True)
        except BaseException as exc:
            import traceback
            durable(ROOT/'experiments/stage4/loss_objective_failure_v1.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
            durable(ROOT/'experiments/stage4/loss_objective_status_v1.json',dict(status='FAILED',error=repr(exc),timestamp=now()));raise
    else:
        install()
        if sys.argv[1]=='reload':
            sys.argv=[str(ROOT/'scripts/reload_stage4.py'),*sys.argv[2:]]
            runpy.run_path(sys.argv[0],run_name='__main__')
        else:
            sys.argv[0]=str(ROOT/'scripts/run_stage4.py');runpy.run_path(sys.argv[0],run_name='__main__')
if __name__=='__main__':main()
