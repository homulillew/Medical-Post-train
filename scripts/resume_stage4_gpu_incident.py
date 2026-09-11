#!/usr/bin/env python3
"""Recover incident002, observe checkpoint adoption and one fresh guarded window."""
from pathlib import Path
import fcntl
import subprocess
import sys
import threading
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now,check_sources


def live(pid):
    p=Path(f'/proc/{pid}/stat');return p.exists() and p.read_text().split()[2]!='Z'


def main():
    index=ROOT/'experiments/stage4';out=Path(read(index/'formal_pair.json')['runs']['vanilla']['path'])
    lock=(index/'formal_incident_resume.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    incident=read(index/'vanilla_formal_incident_002.json');w=out/'windows'/incident['window']
    before=read(w/'state_before.json');phase={'status':'AUDITING_BOUNDARY_AND_PENDING_CHECKPOINT'};stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):durable(index/'gpu_incident_recovery_status.json',dict(timestamp=now(),pid=__import__('os').getpid(),**phase))
    thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
    try:
        check_sources(out)
        for ref in incident['preserved_sources']:assert record(ref['path'])==ref
        assert record(out/'checkpoint.json')==incident['previous_state']
        from stage4_runtime_v2 import install,launch
        install()
        from medical_posttrain.rl.recovery import recover_actor_transaction
        recovery=recover_actor_transaction(out)
        assert recovery['state']==before
        assert read(w/'checkpoint_adoption.json')['optimizer_steps_to_reexecute']==0
        immutable(index/'vanilla_formal_recovery_002.json',dict(timestamp=now(),incident=record(index/'vanilla_formal_incident_002.json'),
            recovery=record(Path(recovery['archive'])/'recovery.json'),action=recovery['action'],pending_checkpoint=record(w/'checkpoint/COMMITTED.json'),optimizer_steps_to_reexecute=0))
        phase['status']='STARTING_SAME_RUN_WITH_GPU_RELEASE_GUARD'
        owner=launch(out);phase['worker_pid']=owner
        while read(out/'status.json')['status']!='RUNNING':
            assert live(owner);time.sleep(1)
        # Attach the persistent queue without a second prelaunch reconciliation.
        directory=out/'attempt_003/queue';directory.mkdir()
        argv=[sys.executable,str(ROOT/'scripts/stage4_queue_runtime.py')]
        immutable(directory/'command.json',argv)
        with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
            q=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True)
        receipt=dict(timestamp=now(),pid=q.pid,worker_pid=owner,command=argv,logs=str(directory),runtime=record(ROOT/'scripts/stage4_queue_runtime.py'))
        immutable(directory/'launch.json',receipt);immutable(index/'formal_queue_resume_002.json',receipt)
        phase.update(status='WAITING_FOR_ADOPTED_CHECKPOINT_SYNC',queue_pid=q.pid)
        while not (w/'commit.json').exists():
            assert live(owner) and read(out/'status.json')['status']=='RUNNING','Worker stopped during adoption'
            time.sleep(5)
        commit=read(w/'commit.json');after=commit['state_after'];reloaded=read(w/'recovered_actor/result.json')
        assert commit['state_before']==before
        assert after['training_groups']==3224 and after['optimizer_steps']==806 and after['policy_windows']==403
        assert reloaded['result']=='PASS' and reloaded['optimizer_steps_added']==0
        assert read(w/'adoption_exit.json')['exit_code']==0
        for ref in incident['preserved_sources']:assert record(ref['path'])==ref
        immutable(index/'vanilla_formal_resume_verified_002.json',dict(result='PASS',timestamp=now(),
            run_id=out.name,worker_pid=owner,queue_pid=q.pid,adopted_window=incident['window'],
            committed_training_groups=3224,optimizer_steps=806,optimizer_steps_reexecuted=0,
            original_rollout_selection_update_checkpoint_unchanged=True,
            commit=record(w/'commit.json'),reload=record(w/'recovered_actor/result.json'),
            guard_receipts=[record(p) for p in (w/'gpu_release_observations').glob('*.json')]))
        phase['status']='ADOPTION_VERIFIED_WAITING_FOR_FRESH_WINDOW'
        fresh=out/'windows/0403'
        while not (fresh/'commit.json').exists():
            assert live(owner) and read(out/'status.json')['status']=='RUNNING','Worker stopped in fresh window'
            time.sleep(5)
        new=read(fresh/'commit.json');assert new['state_before']==after
        assert new['state_after']['training_groups']==3232 and new['state_after']['optimizer_steps']==808
        guards=[read(p) for p in (fresh/'gpu_release_observations').glob('*.json')]
        assert guards and all(g['result']=='RELEASED' and g['raw_nvml_values'] for g in guards)
        assert read(fresh/'actor/resume_receipt.json')['restored_state']==after
        immutable(index/'vanilla_formal_fresh_guard_verified_002.json',dict(result='PASS',timestamp=now(),
            window='0403',training_groups=3232,optimizer_steps=808,commit=record(fresh/'commit.json'),
            resume=record(fresh/'actor/resume_receipt.json'),guards=[record(p) for p in (fresh/'gpu_release_observations').glob('*.json')]))
        stop.set();thread.join()
        phase.update(status='RESUMED_AND_FRESH_WINDOW_VERIFIED',committed_training_groups=3232)
        durable(index/'gpu_incident_recovery_status.json',dict(timestamp=now(),**phase))
    except BaseException as exc:
        stop.set();thread.join()
        immutable(index/f'gpu_incident_recovery_failure_{time.time_ns()}.json',dict(timestamp=now(),error=repr(exc),traceback=traceback.format_exc()))
        durable(index/'gpu_incident_recovery_status.json',dict(status='NEEDS_DIAGNOSIS',timestamp=now(),error=repr(exc)))
        raise
    finally:stop.set()


if __name__=='__main__':main()
