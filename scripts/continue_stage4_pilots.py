#!/usr/bin/env python3
"""Detached bounded pilot queue; every transition requires raw verification.

The queue runs the prepared Vanilla then Dynamic pair and their real resume
checks. Formal preparation remains gated on reviewing both completed pilots.
"""
from pathlib import Path
import os
import subprocess
import sys
import time
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,immutable,durable,now


def live(pid):
    p=Path(f'/proc/{pid}/stat')
    return p.exists() and p.read_text().split()[2]!='Z'


def run():
    from verify_stage4 import online_run
    from analyze_stage4 import analyze
    pair=read(INDEX/'pilot_pair.json')
    deadline=time.monotonic()+24*3600
    durable(INDEX/'pilot_queue_status.json',dict(status='RUNNING',pid=os.getpid(),timestamp=now()))
    for variant in ('vanilla','dynamic'):
        path=Path(pair['runs'][variant]['path'])
        receipt=INDEX/f'{variant}_pilot_verification.json'
        if receipt.exists():
            assert read(receipt)['result']=='PASS'
            continue
        status=read(path/'status.json')['status']
        assert status in ('PREPARED','RUNNING','INTERRUPTED','PILOT_PASS'), (variant,status)
        if status in ('PREPARED','INTERRUPTED'):
            subprocess.run([sys.executable,str(ROOT/'scripts/run_stage4.py'),'launch','--run',str(path)],check=True)
        if not (path/'physical_resume_receipt.json').exists():
            subprocess.run([sys.executable,str(ROOT/'scripts/resume_stage4_smoke.py'),'--run',str(path)],check=True)
        while read(path/'status.json')['status']!='PILOT_PASS':
            assert read(path/'status.json')['status']!='FAILED', str(path)
            assert time.monotonic()<deadline, 'Pilot queue deadline reached; worker evidence retained'
            attempts=sorted(path.glob('attempt_*'))
            assert live(read(attempts[-1]/'launch.json')['pid']), 'Pilot parent exited without terminal status; inspect before recovery'
            durable(INDEX/'pilot_queue_status.json',dict(status='RUNNING',variant=variant,run_id=path.name,
                pid=os.getpid(),timestamp=now(),heartbeat=read(path/'heartbeat.json') if (path/'heartbeat.json').exists() else None))
            time.sleep(30)
        # Hashes/token decode/selection/loss/optimizer lineage/validation are all
        # replayed; worker's own PILOT_PASS is not sufficient to advance.
        verified=online_run(path,'pilot')
        immutable(receipt,verified)
        analyze(path)
        while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())>=100:
            assert time.monotonic()<deadline
            time.sleep(5)
    durable(INDEX/'pilot_queue_status.json',dict(status='BOTH_PILOTS_RAW_VERIFIED',timestamp=now(),
        next='Review stability/cases/resource evidence, then freeze shared formal config and code',
        formal_training_groups=0,READY_FOR_STAGE5='NO'))
    state=read(ROOT/'project_state.json')
    state['stages']['4']['status']='PILOT_PASS'
    durable(ROOT/'project_state.json',state)


if __name__=='__main__':
    try:
        run()
    except BaseException:
        durable(INDEX/'pilot_queue_status.json',dict(status='QUEUE_FAILED',timestamp=now(),traceback=traceback.format_exc(),
            action='Inspect retained worker/checkpoints; do not restart a new pilot automatically'))
        raise
