#!/usr/bin/env python3
"""Operational recovery of incident001 using the unchanged frozen training code.

No scientific runtime is patched. Read-only history validation can take a long
 time on cold storage; this supervisor survives the interactive session.
"""
from pathlib import Path
import os
import subprocess
import sys
import threading
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,immutable,durable,check_sources,now


def alive(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def main():
    import fcntl
    lock=(INDEX/'formal_incident_resume.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pair=read(INDEX/'formal_pair.json');out=Path(pair['runs']['vanilla']['path']);window=out/'windows/0303'
    incident=read(INDEX/'vanilla_formal_incident_001.json')
    phase={'name':'VERIFYING_CHECKPOINT_HISTORY'};stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):
            durable(INDEX/'formal_recovery_status.json',dict(status=phase['name'],timestamp=now(),pid=os.getpid(),
                run_id=out.name,committed_training_groups=read(out/'checkpoint.json')['state']['training_groups']))
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        check_sources(out)
        for ref in incident['inflight_sources']:assert record(ref['path'])==ref
        assert read(out/'checkpoint.json')['state']==incident['previous_state']
        assert not (window/'actor_launch.json').exists() and not (window/'update').exists()
        from medical_posttrain.rl.recovery import recover_actor_transaction
        recovery=recover_actor_transaction(out)
        assert recovery['state']==incident['previous_state']
        immutable(INDEX/'vanilla_formal_recovery_001.json',dict(timestamp=now(),incident=record(INDEX/'vanilla_formal_incident_001.json'),
            recovery=recovery,frozen_code_unchanged=True,frozen_config_unchanged=True,formal_run_replaced=False))
        phase['name']='STARTING_ORIGINAL_RUN'
        # The trusted convenience pointer already matched the full verified
        # commit chain. Launch directly, then attach the unchanged queue to the
        # RUNNING owner; this avoids an unnecessary third historical scan.
        subprocess.run([sys.executable,str(ROOT/'scripts/run_stage4.py'),'launch','--run',str(out)],check=True)
        attempt=sorted(out.glob('attempt_*'))[-1];owner=read(attempt/'launch.json')['pid']
        while read(out/'status.json')['status']!='RUNNING':
            assert alive(owner) and read(out/'status.json')['status']!='FAILED'
            time.sleep(2)
        directory=Path('/data/WSH/medical-post-train-artifacts/stage4-formal-queue')/pair['pair_id']/'attempt_002'
        directory.mkdir(parents=True,exist_ok=False)
        argv=[sys.executable,str(ROOT/'scripts/continue_stage4_formal.py')]
        immutable(directory/'command.json',argv)
        with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
            queue=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
        launch=dict(timestamp=now(),pid=queue.pid,worker_pid=owner,logs=str(directory),command=argv,detached=True)
        immutable(directory/'launch.json',launch)
        immutable(INDEX/'formal_queue_resume_001.json',launch)
        phase['name']='WAITING_FOR_FIRST_RESUMED_COMMIT'
        while not (window/'commit.json').exists():
            assert alive(owner), 'Resumed worker exited; inspect retained attempt failure'
            assert read(out/'status.json')['status']=='RUNNING',read(out/'status.json')
            time.sleep(10)
        commit=read(window/'commit.json')
        assert commit['state_before']==incident['previous_state']
        after=commit['state_after'];before=incident['previous_state']
        assert after['training_groups']==2432 and after['optimizer_steps']==608 and after['policy_windows']==304
        assert after['generated_groups']==before['generated_groups']+incident['inflight_generated_groups']
        assert after['output_tokens']==before['output_tokens']+incident['inflight_output_tokens']
        assert after['prompt_tokens']==before['prompt_tokens']+incident['inflight_prompt_tokens']
        for ref in incident['inflight_sources']:assert record(ref['path'])==ref
        native=read(window/'actor/resume_receipt.json')
        assert native['result']=='PASS' and native['restored_state']==before and native['restored_rng_digest']
        assert read(window/'actor_exit.json')['exit_code']==0
        immutable(INDEX/'vanilla_formal_resume_verified_001.json',dict(result='PASS',timestamp=now(),
            old_pid=read(out/'attempt_001/launch.json')['pid'],new_pid=owner,run_id=out.name,
            before=before,after=after,inflight_rollout_reused_exactly=True,inflight_optimizer_steps_before_recovery=0,
            code_and_config_unchanged=True,sources=[record(window/'commit.json'),record(window/'actor/resume_receipt.json'),
                record(INDEX/'vanilla_formal_incident_001.json')]))
        durable(INDEX/'formal_recovery_status.json',dict(status='RESUMED_WINDOW_RAW_IDENTITY_VERIFIED',timestamp=now(),
            worker_pid=owner,queue_pid=queue.pid,run_id=out.name,committed_training_groups=after['training_groups']))
    except BaseException as exc:
        immutable(INDEX/f'formal_recovery_failure_{time.time_ns()}.json',dict(timestamp=now(),error=repr(exc),traceback=traceback.format_exc()))
        durable(INDEX/'formal_recovery_status.json',dict(status='NEEDS_DIAGNOSIS',timestamp=now(),error=repr(exc)))
        raise
    finally:stop.set()


if __name__=='__main__':main()
