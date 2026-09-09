#!/usr/bin/env python3
"""Bounded external observer: physically terminate paused smoke, verify exit, relaunch."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,read,immutable,now


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    args=p.parse_args()
    out=Path(args.run)
    deadline=time.monotonic()+2400
    while not (out/'pause_ready.json').exists():
        assert read(out/'status.json')['status']!='FAILED', 'Smoke failed before pause'
        assert time.monotonic()<deadline, 'Timed out waiting for smoke checkpoint'
        time.sleep(5)
    pause=read(out/'pause_ready.json')
    subprocess.run([sys.executable,str(ROOT/'scripts/run_stage4.py'),'terminate','--run',str(out)],check=True)
    pid=pause['pid']
    while True:
        path=Path(f'/proc/{pid}/stat')
        dead=not path.exists() or path.read_text().split()[2]=='Z'
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if dead and memory<100:
            break
        assert time.monotonic()<deadline, 'Owned worker did not exit'
        time.sleep(2)
    immutable(out/'termination_observed.json',dict(old_pid=pid,dead=True,gpu_memory_mib=memory,timestamp=now()))
    subprocess.run([sys.executable,str(ROOT/'scripts/run_stage4.py'),'launch','--run',str(out)],check=True)
    while not (out/'attempt_002/resume_start.json').exists():
        assert read(out/'status.json')['status']!='FAILED'
        assert time.monotonic()<deadline
        time.sleep(2)
    resumed=read(out/'attempt_002/resume_start.json')
    assert resumed['pid']!=pid and resumed['state']==pause['state']
    assert resumed['next_encounters']==pause['next_encounters']
    immutable(out/'physical_resume_receipt.json',dict(result='PASS',old_pid=pid,new_pid=resumed['pid'],
        state=resumed['state'],next_encounters_match=True,timestamp=now(),
        scope='Fresh parent resumes committed controller state; actor reload and subsequent updates checked separately'))
    print('Physical resume observed',flush=True)


if __name__=='__main__':
    main()
