"""Persistently finish Stage 2 generation -> frozen scoring -> review packet.

Never declares the stage done; independent verification/report/review remain.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from medical_posttrain.evidence import write_json,now,sha256
from medical_posttrain.evidence.stage2 import read,record

p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
folder=root/'supervisor';folder.mkdir(exist_ok=False)
write_json(folder/'command.json',sys.argv)
write_json(folder/'source.json',record(__file__))
(folder/'source.py').write_bytes(Path(__file__).read_bytes())
start=time.monotonic()
try:
    while True:
        status=read(root/'status.json')['status']
        write_json(folder/'heartbeat.json',dict(pid=os.getpid(),timestamp=now(),worker_status=status))
        if status=='GENERATED':
            subprocess.run([sys.executable,'scripts/run_stage2.py','score','--run',str(root)],check=True)
            # launcher returns before worker changes status
            time.sleep(3)
            break
        if status in ('FAILED','INVALID','INTERRUPTED'):raise RuntimeError(f'Generation ended {status}; preserve evidence, manual recovery required')
        time.sleep(5)
    while True:
        status=read(root/'status.json')['status']
        write_json(folder/'heartbeat.json',dict(pid=os.getpid(),timestamp=now(),worker_status=status))
        if status=='PASS':break
        if status in ('FAILED','INVALID'):raise RuntimeError(f'Scoring ended {status}')
        time.sleep(5)
    subprocess.run([sys.executable,'scripts/analyze_stage2_cases.py'],check=True)
    write_json(folder/'status.json',dict(status='RUNTIME_CHECKS_COMPLETE',timestamp=now(),wall_seconds=time.monotonic()-start,
               next='Manual reading, retrospective report, calibration, artifact seal, verifier and final git push required; stage is not DONE'))
except BaseException:
    write_json(folder/'status.json',dict(status='FAILED',timestamp=now(),traceback=traceback.format_exc()));raise
