#!/usr/bin/env python3
"""Operational recovery boundary plus bounded actor-teardown synchronization."""
from pathlib import Path
import fcntl
import os
import runpy
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,check_sources,now


def install():
    from medical_posttrain.rl import online
    from stage4_resume_boundary import restore_boundary
    from stage4_gpu_release_guard import install as install_guard
    online.restore=restore_boundary
    install_guard()


def launch(out):
    out=Path(out)
    with (out/'launch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(out/'status.json')['status'] in ('PREPARED','INTERRUPTED')
        for attempt in out.glob('attempt_*'):
            if not (attempt/'launch.json').exists():continue
            pid=read(attempt/'launch.json')['pid'];stat=Path(f'/proc/{pid}/stat')
            assert not stat.exists() or stat.read_text().split()[2]=='Z'
        check_sources(out)
        used=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert used<100,f'GPU still owned: {used}MiB'
        logdir=out/f'attempt_{len(list(out.glob("attempt_*")))+1:03d}';logdir.mkdir()
        argv=[sys.executable,str(Path(__file__).resolve()),'worker','--run',str(out),'--attempt',str(logdir)]
        immutable(logdir/'command.json',argv)
        sources=[record(ROOT/p) for p in ('scripts/stage4_runtime_v2.py','scripts/stage4_gpu_release_guard.py','scripts/stage4_resume_boundary.py')]
        immutable(logdir/'operational_runtime.json',dict(timestamp=now(),
            override='online.restore and online.subprocess.Popen.wait for successful actor/adoption exit only',
            scope='Recovery I/O and <=60s GPU-release wait; both formal variants; original4GiB guard and all scientific settings retained',
            sources=sources,frozen_worker=record(ROOT/'scripts/run_stage4.py'),full_history_verifier=record(ROOT/'scripts/verify_stage4.py')))
        import zipfile
        with zipfile.ZipFile(logdir/'operational_source.zip','x',zipfile.ZIP_DEFLATED) as z:
            for ref in sources:z.write(ref['path'],str(Path(ref['path']).relative_to(ROOT)))
        env=dict(os.environ,TOKENIZERS_PARALLELISM='false');env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        with (logdir/'stdout.log').open('x') as stdout,(logdir/'stderr.log').open('x') as stderr:
            p=subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
        immutable(logdir/'launch.json',dict(pid=p.pid,timestamp=now(),detached=True))
        return p.pid


if __name__=='__main__':
    assert sys.argv[1]=='worker'
    attempt=Path(sys.argv[sys.argv.index('--attempt')+1]);runtime=read(attempt/'operational_runtime.json')
    for ref in runtime['sources']+[runtime['frozen_worker'],runtime['full_history_verifier']]:assert record(ref['path'])==ref
    install();runpy.run_path(str(ROOT/'scripts/run_stage4.py'),run_name='__main__')
