#!/usr/bin/env python3
"""Persistent Stage4 launcher; diagnostic is a separate bounded experiment."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import (ROOT,INDEX,read,prepare,inherited,validate,
    check_sources,immutable,durable,event,seal,record,now)


def diagnostic(out):
    from medical_posttrain.evidence.stage2 import jsonlines
    from medical_posttrain.sampling.dynamic import Stream
    from medical_posttrain.rl.rollout import Rollout,score
    from medical_posttrain.training.lora import MemoryMonitor
    cfg = read(out/'config.json')
    init = validate(cfg)
    pool = {r['prompt_id']:r for r in jsonlines(cfg['pool']['path'])}
    assert len(pool) == 15000 and all(r['split']=='train' for r in pool.values())
    stream = Stream(list(pool),cfg['seed'],cfg['stream_domain'])
    if cfg.get('trajectory_source'):
        source = Path(cfg['trajectory_source']['path'])
        assert record(source) == cfg['trajectory_source']
        groups = read(source)['groups']
    else:
        with MemoryMonitor() as memory:
            rollout = Rollout(out,cfg,init)
            groups = rollout.generate(out/'trajectories',stream,pool,0,8,out.name,read(out/'manifest.json')['config_sha256'])
            rollout.close()
            groups = score(out/'trajectories',groups,cfg,pool,rollout.tok)
            immutable(out/'generation_memory.json',memory.result())
    # Scorer tensors must leave GPU before fresh actor process starts.
    import gc,torch
    gc.collect()
    torch.cuda.empty_cache()
    summaries = []
    first_old = None
    for lr in cfg['diagnostic_lrs']:
        directory = out/f'lr_{lr:g}'
        directory.mkdir()
        child_cfg = dict(cfg,learning_rate=lr)
        immutable(directory/'config.json',child_cfg)
        argv = [sys.executable,str(ROOT/'scripts/run_stage4.py'),'actor-diagnostic','--run',str(out),
                '--directory',str(directory)]
        if first_old:
            argv += ['--fixed-old',str(first_old)]
        immutable(directory/'command.json',argv)
        with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
            proc = subprocess.Popen(argv,stdout=stdout,stderr=stderr,cwd=ROOT,
                env=dict(os.environ,PYTORCH_ALLOC_CONF='expandable_segments:True'))
            immutable(directory/'launch.json',dict(pid=proc.pid,timestamp=now()))
            code = proc.wait()
        immutable(directory/'exit.json',dict(exit_code=code,timestamp=now()))
        assert code == 0, f'Actor diagnostic failed: {directory}'
        first_old = first_old or directory/'update/old.npz'
        summaries.append(dict(lr=lr,**read(directory/'update/update.json')))
    seal(out,dict(status='DIAGNOSTIC_PASS',run_id=out.name,generated_groups=8,trajectories=32,
        generated_output_tokens=0 if cfg.get('trajectory_source') else sum(r['output_tokens'] for g in groups for r in g['responses']),
        trajectory_source=cfg.get('trajectory_source'),
        conditions=summaries,formal_training_groups=0,
        decision='LR selection requires review of these measured conditions, then online smoke'))


def actor_diagnostic(args):
    from medical_posttrain.rl.actor import Actor
    from medical_posttrain.training.lora import MemoryMonitor
    out,directory = Path(args.run),Path(args.directory)
    cfg = read(directory/'config.json')
    init = validate(cfg)
    with MemoryMonitor() as memory:
        actor = Actor(cfg,directory/'runtime',init['adapter_path'])
        groups = read(cfg['trajectory_source']['path'] if cfg.get('trajectory_source') else out/'trajectories/scored.json')['groups']
        summary = actor.update(groups,directory/'update',fixed_old=args.fixed_old,check_parity=True)
        adapter = actor.save(directory/'checkpoint',2,dict(kind='DIAGNOSTIC',summary=summary))
        immutable(directory/'result.json',dict(summary=summary,adapter=adapter,**memory.result()))
        actor.close()


def worker(out,attempt=None):
    out = Path(out)
    logdir = Path(attempt) if attempt else out
    stop = threading.Event()
    def heartbeat():
        while not stop.wait(10):
            durable(out/'heartbeat.json',dict(pid=os.getpid(),timestamp=now()))
    threading.Thread(target=heartbeat,daemon=True).start()
    try:
        check_sources(out)
        from medical_posttrain.runtime import environment
        immutable(logdir/'environment.json',environment())
        durable(out/'status.json',dict(status='RUNNING',pid=os.getpid(),timestamp=now()))
        with (ROOT.parent/'stage4-gpu.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if read(out/'config.json')['mode']=='diagnostic':
                diagnostic(out)
            else:
                from medical_posttrain.rl.online import run
                run(out,logdir)
    except BaseException as exc:
        immutable(logdir/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
        durable(out/'status.json',dict(status=getattr(exc,'status','FAILED'),timestamp=now()))
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Own process group only, including failed engine children.
        if os.getpid() == os.getpgrp():
            os.killpg(os.getpid(),signal.SIGTERM)
        raise
    finally:
        stop.set()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action',choices=['prepare-diagnostic','prepare-smoke','launch','inspect','worker','actor-diagnostic','actor-window','terminate','recover','adopt-actor'])
    p.add_argument('--run')
    p.add_argument('--directory')
    p.add_argument('--fixed-old')
    p.add_argument('--trajectory-source')
    p.add_argument('--variant',choices=['vanilla','dynamic'])
    p.add_argument('--attempt')
    a = p.parse_args()
    if a.action == 'prepare-diagnostic':
        cfg = inherited()
        cfg.update(mode='diagnostic',stream_domain='stage4:optimization-diagnostic',diagnostic_lrs=[1e-5,3e-6,1e-6])
        if a.trajectory_source:
            cfg['trajectory_source'] = record(a.trajectory_source)
        out = prepare('optimization_diagnostic','DIAGNOSTIC',cfg)
        durable(INDEX/'active_diagnostic.json',dict(run_id=out.name,path=str(out)))
        print(out)
    elif a.action == 'prepare-smoke':
        assert a.variant and read(INDEX/'diagnostic_verification.json')['result']=='PASS'
        cfg = inherited()
        cfg.update(mode='smoke',sampling_mode=a.variant,stream_domain='stage4:smoke:pair1',
                   target_training_groups=32,pause_after_windows=2,
                   diagnostic_receipt=record(INDEX/'diagnostic_verification.json'))
        out = prepare('smoke_'+a.variant,'SMOKE',cfg)
        durable(INDEX/f'active_smoke_{a.variant}.json',dict(run_id=out.name,path=str(out)))
        print(out)
    elif a.action == 'launch':
        out = Path(a.run)
        launch_lock=(out/'launch.lock').open('a')
        fcntl.flock(launch_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(out/'status.json')['status'] in ('PREPARED','INTERRUPTED')
        for attempt in out.glob('attempt_*'):
            if not (attempt/'launch.json').exists():
                continue
            pid=read(attempt/'launch.json')['pid']
            stat=Path(f'/proc/{pid}/stat')
            assert not stat.exists() or stat.read_text().split()[2]=='Z', f'Existing owner is alive: {pid}'
        check_sources(out)
        memory = int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())
        assert memory < 100, f'GPU still owned: {memory}MiB'
        argv = [sys.executable,str(ROOT/'scripts/run_stage4.py'),'worker','--run',str(out)]
        logdir = out
        if read(out/'config.json')['mode']!='diagnostic':
            logdir = out/f'attempt_{len(list(out.glob("attempt_*")))+1:03d}'
            logdir.mkdir()
            argv += ['--attempt',str(logdir)]
        immutable(logdir/'command.json',argv)
        env = dict(os.environ,TOKENIZERS_PARALLELISM='false')
        env.pop('PYTORCH_ALLOC_CONF',None)
        env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
        with (logdir/'stdout.log').open('x') as stdout,(logdir/'stderr.log').open('x') as stderr:
            proc = subprocess.Popen(argv,cwd=ROOT,stdout=stdout,stderr=stderr,start_new_session=True,env=env)
        immutable(logdir/'launch.json',dict(pid=proc.pid,timestamp=now(),detached=True))
        print(dict(run=str(out),pid=proc.pid))
        launch_lock.close()
    elif a.action == 'inspect':
        out = Path(a.run)
        print(read(out/'status.json'))
        if (out/'heartbeat.json').exists():
            print(read(out/'heartbeat.json'))
        if (out/'summary.json').exists():
            print(read(out/'summary.json'))
        elif (out/'checkpoint.json').exists():
            state=read(out/'checkpoint.json')['state']
            print({k:v for k,v in state.items() if not k.endswith('exposure')})
        attempts=sorted(out.glob('attempt_*'))
        if attempts:
            pid=read(attempts[-1]/'launch.json')['pid']
            stat=Path(f'/proc/{pid}/stat')
            print(dict(owner_pid=pid,owner_alive=stat.exists() and stat.read_text().split()[2]!='Z'))
    elif a.action == 'recover':
        from medical_posttrain.rl.recovery import recover_actor_transaction
        print(recover_actor_transaction(Path(a.run)))
    elif a.action == 'actor-diagnostic':
        actor_diagnostic(a)
    elif a.action == 'actor-window':
        from medical_posttrain.rl.online import actor_window
        out,window = Path(a.run),Path(a.directory)
        state = read(window/'state_before.json')
        checkpoint = out/'windows'/f'{state["policy_windows"]-1:04d}'/'checkpoint' if state['policy_windows'] else None
        actor_window(out,window,checkpoint,state)
    elif a.action == 'adopt-actor':
        from medical_posttrain.rl.online import adopt_actor
        adopt_actor(Path(a.run),Path(a.directory))
    elif a.action == 'terminate':
        out = Path(a.run)
        pause = read(out/'pause_ready.json')
        pid = pause['pid']
        cmdline = Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')
        assert 'run_stage4.py worker' in cmdline and str(out) in cmdline and os.getpgid(pid)==pid
        assert read(out/'checkpoint.json')['state']==pause['state']
        immutable(out/'termination.json',dict(pid=pid,signal='SIGTERM',timestamp=now(),cmdline=cmdline,
            state=pause['state'],next_encounters=pause['next_encounters']))
        os.killpg(pid,signal.SIGTERM)
        durable(out/'status.json',dict(status='INTERRUPTED',timestamp=now()))
        print(dict(terminated_pid=pid))
    else:
        worker(a.run,a.attempt)


if __name__ == '__main__':
    main()
