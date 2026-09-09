#!/usr/bin/env python3
"""Persistent Vanilla -> raw verify -> Dynamic -> raw verify; no tuning or Stage5."""
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,sha256,immutable,durable,now


def live(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def run():
    from verify_stage4 import online_run
    from analyze_stage4 import analyze
    lock=(INDEX/'formal_queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=threading.Event()
    def heartbeat():
        while not stop.wait(10):durable(INDEX/'formal_queue_heartbeat.json',dict(pid=os.getpid(),timestamp=now()))
    threading.Thread(target=heartbeat,daemon=True).start()
    pair=read(INDEX/'formal_pair.json')
    assert pair['formal_frozen'] and pair['clean_git_state']['porcelain']==''
    try:
        for variant in ('vanilla','dynamic'):
            out=Path(pair['runs'][variant]['path']);receipt=INDEX/f'{variant}_formal_verification.json'
            for p,h in pair['execution_hashes'].items():assert sha256(ROOT/p)==h,p
            assert record(pair['frozen_config']['path'])==pair['frozen_config']
            assert read(out/'config.json')==dict(read(pair['frozen_config']['path']),sampling_mode=variant)
            if receipt.exists():
                assert read(receipt)['result']=='PASS' and read(receipt)['state']['training_groups']==5000
                continue
            status=read(out/'status.json')['status']
            assert status in ('PREPARED','INTERRUPTED','RUNNING','FULL_PASS'), (variant,status,'Diagnose retained failure before audited recovery')
            if status in ('PREPARED','INTERRUPTED'):
                from audit_stage4_boundary import reconcile
                reconcile(out)
                subprocess.run([sys.executable,str(ROOT/'scripts/run_stage4.py'),'launch','--run',str(out)],check=True)
                project=read(ROOT/'project_state.json');project['stages']['4']['status']='FULL_RUNNING'
                durable(ROOT/'project_state.json',project)
            while read(out/'status.json')['status']!='FULL_PASS':
                status=read(out/'status.json')['status']
                assert status=='RUNNING' or status=='PREPARED', (variant,status,'No automatic scientific changes')
                attempts=sorted(out.glob('attempt_*'));owner=read(attempts[-1]/'launch.json')['pid']
                assert live(owner), f'Worker {owner} exited; inspect transaction and inflight cost before recovery'
                counters=read(out/'checkpoint.json')['state'] if (out/'checkpoint.json').exists() else None
                if counters:counters={k:v for k,v in counters.items() if not k.endswith('exposure')}
                durable(INDEX/'formal_queue_status.json',dict(status='FULL_RUNNING',variant=variant,run_id=out.name,
                    owner_pid=owner,pid=os.getpid(),timestamp=now(),committed=counters,READY_FOR_STAGE5='NO'))
                time.sleep(30)
            # Wait for complete artifact sealing and process teardown before verification.
            owner=read(sorted(out.glob('attempt_*'))[-1]/'launch.json')['pid']
            while live(owner):time.sleep(5)
            exit_receipt=INDEX/out.name/'worker_exit_observed.json'
            if not exit_receipt.exists():
                immutable(exit_receipt,dict(pid=owner,observed_at=now(),alive=False,poll_interval_seconds=5,
                    launch=record(sorted(out.glob('attempt_*'))[-1]/'launch.json'),
                    note='Observed process termination after artifact sealing; timestamp is an observation, not an invented exact exit time'))
            durable(INDEX/'formal_queue_status.json',dict(status='RAW_VERIFYING',variant=variant,run_id=out.name,pid=os.getpid(),timestamp=now(),READY_FOR_STAGE5='NO'))
            verified=online_run(out,'formal')
            assert verified['state']['training_groups']==5000
            assert read(out/'final_reload/result.json')['result']=='PASS' and read(out/'final_reload/exit.json')['exit_code']==0
            immutable(receipt,verified)
            analyze(out)
            while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=100:time.sleep(5)
        state=read(ROOT/'project_state.json');state['stages']['4']['status']='FULL_PASS'
        durable(ROOT/'project_state.json',state)
        durable(INDEX/'formal_queue_status.json',dict(status='FORMAL_PAIR_ANALYSIS',timestamp=now(),READY_FOR_STAGE5='NO'))
        if not (INDEX/'formal_analysis_v1.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/review_stage4_formal.py')],check=True)
        assert read(INDEX/'formal_analysis_v1.json')['pair']==record(INDEX/'formal_pair.json')
        if not (INDEX/'formal_plots/manifest.json').exists():
            subprocess.run([str(ROOT/'.venv-analysis/bin/python'),str(ROOT/'scripts/plot_stage4.py'),
                '--pair',str(INDEX/'formal_pair.json')],check=True)
        durable(INDEX/'formal_queue_status.json',dict(status='BOTH_FORMALS_RAW_VERIFIED_AND_ANALYZED',timestamp=now(),
            next='Review measured pair analysis and full responses, complete manual cases/stage report/interview/handoff, then full stage verifier. Stage5 remains forbidden.',READY_FOR_STAGE5='NO'))
    except BaseException as exc:
        immutable(INDEX/f'formal_queue_failure_{time.time_ns()}.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
        durable(INDEX/'formal_queue_status.json',dict(status='NEEDS_DIAGNOSIS',timestamp=now(),error=repr(exc),
            next='Inspect owner, heartbeat, retained failure, checkpoint and paid inflight work before recovery. Never restart a fresh replacement implicitly.',READY_FOR_STAGE5='NO'))
        raise
    finally:stop.set()


if __name__=='__main__':run()
