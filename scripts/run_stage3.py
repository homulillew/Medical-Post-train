#!/usr/bin/env python3
"""Prepare, launch and physically terminate Stage 3 inference-only runs."""
import argparse
import os
from pathlib import Path
import signal
import sys
import traceback
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.evidence import now, sha256, write_json
from medical_posttrain.evidence.stage2 import read, record
from medical_posttrain.evidence.stage3 import prepare, launch, Run, select, INDEX, ROOT


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','launch','worker','terminate']);p.add_argument('--mode',choices=['smoke','formal']);p.add_argument('--run');p.add_argument('--attempt');a=p.parse_args()
    if a.action=='prepare':
        assert a.mode
        state=read(ROOT/'project_state.json')
        assert state['stages']['1']['status']=='DONE' and state['stages']['2']['status']=='DONE'
        assert all(state['stages'][str(n)]['status']=='NOT_STARTED' for n in (4,5,6))
        assert read(INDEX/'prerequisite-stage2.json')['result']=='PASS'
        cfg=read(ROOT/'configs/stages/s2_formal_1024.json')
        for k in ('planned_prompts','planned_responses','smoke_receipt','freeze_decision'):cfg.pop(k,None)
        cfg.update(stage=3,mode=a.mode,stream_domain='stage3:'+a.mode,target_accepted=256 if a.mode=='formal' else None,
                   smoke_prompts=32,request_batch_prompts=16,max_num_generation_batches=128 if a.mode=='formal' else 2,sampling_metric='acc')
        files=['src/medical_posttrain/sampling/dynamic.py','src/medical_posttrain/sampling/stage3.py','src/medical_posttrain/sampling/analysis.py',
               'src/medical_posttrain/evidence/stage3.py','scripts/run_stage3.py']
        cfg['controller_hashes']={path:sha256(ROOT/path) for path in files}
        cfg['prerequisite_stage2']=record(INDEX/'prerequisite-stage2.json')
        if a.mode=='formal':
            smoke=read(INDEX/'selected_runs.json')['smoke'];manifest=read(INDEX/smoke/'manifest.json');path=Path(manifest['artifact_root'])
            assert read(path/'summary.json')['status']=='SMOKE_PASS'
            assert read(INDEX/'smoke_verification.json')['result']=='PASS'
            cfg['smoke_receipt']=record(INDEX/'smoke_verification.json')
        out=prepare(a.mode,'SMOKE' if a.mode=='smoke' else 'FORMAL',cfg)
        write_json(ROOT/f'configs/stages/s3_{a.mode}.json',cfg)
        select(a.mode,out);print(out)
    elif a.action=='launch':
        out=Path(a.run);assert read(out/'status.json')['status'] not in ('SMOKE_PASS','FULL_PASS')
        import subprocess
        memory=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert memory<100, f'Prior GPU worker has not exited: {memory} MiB still allocated'
        launch(out,'integrate')
    elif a.action=='terminate':
        out=Path(a.run);pause=read(out/'pause_ready.json');pid=pause['pid']
        assert read(out/'checkpoint.json')==pause['state'] and os.getpgid(pid)==pid
        cmdline=Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
        assert 'run_stage3.py worker' in cmdline and str(out) in cmdline
        write_json(out/'termination.json',dict(timestamp=now(),pid=pid,signal='SIGTERM',command=['os.killpg',pid,'SIGTERM'],state=pause['state'],next_encounters=pause['next_encounters'],cmdline=cmdline))
        os.killpg(pid,signal.SIGTERM)
        write_json(out/'attempt_001/status.json',dict(status='INTERRUPTED',reason='Planned real process termination for resume test',timestamp=now()))
        write_json(out/'status.json',dict(status='INTERRUPTED',timestamp=now()))
        print('SIGTERM sent to owned worker process group',pid)
    else:
        run=Run(a.run,a.attempt)
        try:
            from medical_posttrain.sampling.stage3 import worker
            worker(run)
        except BaseException as exc:
            write_json(run.attempt/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
            write_json(run.attempt/'status.json',dict(status='FAILED',timestamp=now()))
            write_json(run.out/'status.json',dict(status='FAILED',timestamp=now()))
            traceback.print_exc();sys.stdout.flush();sys.stderr.flush()
            if os.getpgrp()==os.getpid():os.killpg(os.getpgrp(),signal.SIGTERM)
            raise

if __name__=='__main__':main()
