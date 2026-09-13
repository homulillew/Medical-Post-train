#!/usr/bin/env python3
"""Persist/publish diagnostic evidence after the existing supervisor ends; never rerun inference."""
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now
from frontier_diagnostic import INDEX


def main():
    launch=read(INDEX/'launch.json')
    def live(pid):
        f=Path(f'/proc/{pid}/stat')
        return f.exists() and f.read_text().split()[2]!='Z'
    while True:
        status=read(INDEX/'status.json')
        if status['status'] in ('DIAGNOSTIC_VERIFIED','FAILED'):break
        if not live(launch['pid']):
            immutable(INDEX/f'supervisor_missing_{time.time_ns()}.json',dict(timestamp=now(),launch=launch,last_status=status))
            durable(INDEX/'status.json',dict(status='FAILED',timestamp=now(),reason='Supervisor exited before terminal evidence; no inference retry'))
            break
        time.sleep(30)
    result=subprocess.run([str(ROOT/'.venv-analysis/bin/python'),str(ROOT/'scripts/build_postformal_handoff.py')],
                           cwd=ROOT,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    handoff=Path(result.stdout.strip().splitlines()[-1])
    # Audit index metadata and generated JSON. Raw payloads stay in the artifact root.
    paths=[p for p in INDEX.rglob('*') if p.is_file() and p.suffix in ('.json','.log')]
    paths += [handoff,ROOT/'experiments/stage4/postformal_handoff_current.json']
    for p in paths:
        assert p.stat().st_size<50*1024*1024,p
        if p.suffix=='.json':read(p)
    status=read(INDEX/'status.json')
    publication=dict(timestamp=now(),diagnostic_status=status['status'],handoff=record(handoff),stage4_done=False,
                     files=[str(p.relative_to(ROOT)) for p in paths])
    # Do not absorb another session's staged changes or force-push a changed remote.
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()
    if branch!='main' or subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode:
        publication['status']='PUBLISH_BLOCKED_EXISTING_INDEX_OR_BRANCH'
        immutable(INDEX/f'publication_{time.time_ns()}.json',publication)
        return
    receipt=INDEX/f'publication_{time.time_ns()}.json'
    publication['status']='READY_TO_COMMIT'
    immutable(receipt,publication)
    subprocess.run(['git','add','--',*[str(p.relative_to(ROOT)) for p in paths+[receipt]]],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','experiment: retain frontier diagnostic outcome and machine handoff'],cwd=ROOT,check=True)
    push=subprocess.run(['git','push','origin','main'],cwd=ROOT,capture_output=True,text=True)
    # This publication outcome is operational state; the pushed receipt has the exact manifest.
    durable(INDEX/'publication_status.json',dict(timestamp=now(),status='PUSHED' if push.returncode==0 else 'PUSH_FAILED',
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),returncode=push.returncode,
        output=push.stdout+push.stderr,diagnostic_status=status['status']))


if __name__=='__main__':main()
