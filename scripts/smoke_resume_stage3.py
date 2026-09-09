#!/usr/bin/env python3
"""Observe a planned checkpoint, SIGTERM owned worker, verify exit, launch new process."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.evidence import write_json,now
from medical_posttrain.evidence.stage2 import read


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run)
    write_json(out/'resume_supervisor_command.json',[sys.executable,__file__,'--run',str(out)])
    deadline=time.monotonic()+600
    while not (out/'pause_ready.json').exists():
        assert read(out/'status.json')['status']!='FAILED'
        assert time.monotonic()<deadline,'Timed out waiting for smoke checkpoint'
        time.sleep(1)
    subprocess.run([sys.executable,'scripts/run_stage3.py','terminate','--run',str(out)],check=True)
    pid=read(out/'termination.json')['pid']
    for _ in range(60):
        stat=Path(f'/proc/{pid}/stat');dead=not stat.exists() or stat.read_text().split()[2]=='Z'
        mem=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        if dead and mem<100:break
        time.sleep(1)
    assert dead and mem<100,(dead,mem)
    write_json(out/'termination_observed.json',dict(old_pid=pid,old_process_exited=dead,gpu_memory_used_mib=mem,timestamp=now()))
    subprocess.run([sys.executable,'scripts/run_stage3.py','launch','--run',str(out)],check=True)

if __name__=='__main__':main()
