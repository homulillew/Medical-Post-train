"""Single-GPU online windows, native actor subprocess and physical resume."""
import gc
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .common import ROOT,INDEX,read,record,sha256,immutable,durable,event,validate,encounter,now
from .controller import select_groups,window_metrics,advance


def verify_checkpoint(path):
    marker = read(path/'COMMITTED.json')
    for ref in marker['files']:
        f = path/ref['path']
        assert f.stat().st_size==ref['bytes'] and sha256(f)==ref['sha256']
    return marker


def initial_state(initial):
    return dict(policy_version=initial['adapter_sha256'],cursor=0,policy_windows=0,
        optimizer_steps=0,training_groups=0,generated_groups=0,output_tokens=0,prompt_tokens=0,
        generated_exposure={},training_exposure={})


def retained(path,value):
    if path.exists():
        assert read(path)==value, f'Existing evidence differs: {path}'
    else:
        immutable(path,value)


def monitor(rollout,out,state,cfg):
    if cfg['mode']=='smoke':
        return
    from .validation import evaluate
    protocol=read(cfg['validation_protocol']['path'])
    if state['policy_windows'] in protocol['checkpoint_windows'] or state['training_groups']==cfg['target_training_groups']:
        event(out,'validation_started',policy_windows=state['policy_windows'],policy_version=state['policy_version'])
        result=evaluate(rollout,out,state,cfg['validation_protocol'])
        event(out,'validation_completed',policy_windows=state['policy_windows'],accuracy=result['accuracy'],
            validation_output_tokens=result['validation_output_tokens'])


def restore(out,initial):
    state = initial_state(initial)
    checkpoint = None
    for window in sorted((out/'windows').glob('*')):
        if not (window/'commit.json').exists():
            break
        commit = read(window/'commit.json')
        assert state==commit['state_before']
        for ref in commit['artifacts']:
            assert record(ref['path'])==ref
        checkpoint = window/'checkpoint'
        verify_checkpoint(checkpoint)
        assert sha256(checkpoint/'adapter/adapter_model.safetensors')==commit['state_after']['policy_version']
        state = commit['state_after']
    return state,checkpoint


def probe(rollout,prompt,directory,previous=None):
    from vllm import SamplingParams
    from medical_posttrain.verification.vllm_probe import serialize
    directory.mkdir(parents=True,exist_ok=False)
    params = SamplingParams(temperature=0,max_tokens=8,prompt_logprobs=0)
    start = time.monotonic()
    a = serialize(rollout.llm.generate([prompt],params,lora_request=rollout.request,use_tqdm=False)[0])
    b = serialize(rollout.llm.generate([prompt],params,lora_request=rollout.request,use_tqdm=False)[0])
    immutable(directory/'first.json',a)
    immutable(directory/'repeat.json',b)
    def delta(x,y):
        assert len(x['prompt_logprobs'])==len(y['prompt_logprobs'])
        return max(abs(u[k]-v[k]) for u,v in zip(x['prompt_logprobs'][1:],y['prompt_logprobs'][1:]) for k in u.keys()&v.keys())
    repeat = delta(a,b)
    assert a['token_ids']==b['token_ids'] and repeat<=1e-4
    change = delta(previous,a) if previous is not None else None
    if previous is not None:
        assert change>1e-6, 'Updated adapter has no measured effect in vLLM probe'
    receipt = dict(policy_version=rollout.policy,adapter_path=rollout.adapter,
        adapter_sha256=sha256(Path(rollout.adapter)/'adapter_model.safetensors'),
        lora_id=rollout.request.lora_int_id,repeat_error=repeat,previous_policy_delta=change,
        prompt_tokens=len(a['prompt_logprobs'])+len(b['prompt_logprobs']),
        output_tokens=len(a['token_ids'])+len(b['token_ids']),seconds=time.monotonic()-start,
        controls=[record(directory/'first.json'),record(directory/'repeat.json')])
    immutable(directory/'receipt.json',receipt)
    return a,receipt


def actor_window(out,window,checkpoint,state):
    from .actor import Actor
    from medical_posttrain.training.lora import MemoryMonitor,restore_rng
    import torch
    cfg = read(out/'config.json')
    initial = validate(cfg)
    adapter = checkpoint/'adapter' if checkpoint else Path(initial['adapter_path'])
    with MemoryMonitor() as memory:
        actor = Actor(cfg,window/'actor',adapter,checkpoint=checkpoint)
        if checkpoint:
            marker = verify_checkpoint(checkpoint)
            identity = read(window/'actor/loaded_identity.json')
            assert identity['trainable_digest']==marker['trainable_digest']
            assert identity['optimizer_digest']==marker['optimizer_digest']
            assert identity['scheduler']==marker['scheduler']
            assert identity['optimizer_steps']==[state['optimizer_steps']]
            restore_rng(torch.load(checkpoint/'rng.pt',weights_only=False))
            immutable(window/'actor/resume_receipt.json',dict(result='PASS',
                source_checkpoint=str(checkpoint),source_marker=record(checkpoint/'COMMITTED.json'),
                restored_identity=identity,restored_state=state,explicit_rng=record(checkpoint/'rng.pt')))
        groups = read(window/'selection.json')['groups']
        summary = actor.update(groups,window/'update',step_before=state['optimizer_steps'],
                               check_parity=(state['policy_windows']==0))
        adapter_ref = actor.save(window/'checkpoint',summary['optimizer_steps_after'],dict(
            state_before=state,selection=record(window/'selection.json'),
            config=record(out/'config.json'),run_id=out.name,update=record(window/'update/update.json')))
        immutable(window/'actor_result.json',dict(summary=summary,adapter=adapter_ref,**memory.result()))
        actor.close()


def run(out,attempt):
    import torch
    from medical_posttrain.evidence.stage2 import jsonlines
    from medical_posttrain.sampling.dynamic import Stream
    from medical_posttrain.reward.semantic import Encoder
    from medical_posttrain.training.lora import MemoryMonitor
    from .rollout import Rollout,score
    out,attempt = Path(out),Path(attempt)
    cfg = read(out/'config.json')
    initial = validate(cfg)
    pool = {r['prompt_id']:r for r in jsonlines(cfg['pool']['path'])}
    assert len(pool)==15000 and all(r['split']=='train' for r in pool.values())
    stream = Stream(list(pool),cfg['seed'],cfg['stream_domain'])
    (out/'windows').mkdir(exist_ok=True)
    state,checkpoint = restore(out,initial)
    immutable(attempt/'resume_start.json',dict(state=state,checkpoint=str(checkpoint) if checkpoint else None,
        pid=os.getpid(),next_encounters=[encounter(stream,i,out.name,state['policy_version']) for i in range(state['cursor'],state['cursor']+8)]))
    if cfg['mode']=='formal':
        assert cfg['target_training_groups']==5000
    assert state['training_groups']<=cfg['target_training_groups']
    with MemoryMonitor() as memory:
        load_initial = dict(initial,adapter_path=str(checkpoint/'adapter'),adapter_sha256=state['policy_version']) if checkpoint else initial
        rollout = Rollout(attempt,cfg,load_initial)
        probe_prompt = rollout.prompt(pool[stream.order(0)[0]])[0]
        previous,initial_sync = probe(rollout,probe_prompt,attempt/'initial_sync')
        monitor(rollout,out,state,cfg)
        rm = read(cfg['reward_manifest']['path'])
        encoder = Encoder(rm['encoder_id'],rm['encoder_revision'],device='cpu')
        while state['training_groups']<cfg['target_training_groups']:
            window = out/'windows'/f'{state["policy_windows"]:04d}'
            window.mkdir(exist_ok=True)
            complete_actor=all((window/p).exists() for p in ('actor_result.json','checkpoint/COMMITTED.json','actor_exit.json'))
            if (window/'actor').exists() and not complete_actor:
                raise RuntimeError('Incomplete actor transaction retained; require explicit audited rollback/recovery')
            if not (window/'state_before.json').exists():
                immutable(window/'state_before.json',state)
            assert read(window/'state_before.json')==state
            groups,decisions,selected = [],[],[]
            encoder.model.to('cuda')
            encoder.device='cuda'
            for batch_index in range(cfg['max_generation_batches']):
                directory = window/'batches'/f'{batch_index:03d}'
                raw = rollout.generate(directory,stream,pool,state['cursor']+len(groups),
                    cfg['request_batch_prompts'],out.name,read(out/'manifest.json')['config_sha256'])
                scored = score(directory,raw,cfg,pool,rollout.tok,encoder)
                accepted,dd = select_groups(scored,state['policy_version'],cfg['sampling_mode'],already=len(selected))
                if not (directory/'dispositions.json').exists():
                    immutable(directory/'dispositions.json',dd)
                assert read(directory/'dispositions.json')==dd
                groups.extend(scored)
                decisions.extend(dd)
                selected.extend(accepted)
                if len(selected)==8:
                    break
            if len(selected)!=8:
                immutable(window/'starvation.json',dict(selected=len(selected),generated=len(groups),decisions=decisions))
                raise RuntimeError('Bounded refill exhausted before eight training groups')
            metrics = window_metrics(groups,decisions)
            retained(window/'selection.json',dict(groups=selected,decisions=decisions,policy_version=state['policy_version']))
            retained(window/'rollout_metrics.json',metrics)
            encoder.model.to('cpu')
            encoder.device='cpu'
            gc.collect()
            torch.cuda.empty_cache()
            start = time.monotonic()
            rollout.llm.sleep(level=1)
            sleep_seconds = time.monotonic()-start
            sleeping_bytes = memory.nv.nvmlDeviceGetMemoryInfo(memory.handle).used
            assert sleeping_bytes<4*1024**3, 'Unexpected GPU residue before actor load'
            argv = [sys.executable,str(ROOT/'scripts/run_stage4.py'),'actor-window','--run',str(out),
                    '--directory',str(window)]
            if not complete_actor:
                import shutil
                assert shutil.disk_usage(out).free>100*1024**3, 'Evidence storage reserve exhausted'
                immutable(window/'actor_command.json',argv)
                start = time.monotonic()
                with (window/'actor.stdout.log').open('x') as stdout,(window/'actor.stderr.log').open('x') as stderr:
                    proc = subprocess.Popen(argv,stdout=stdout,stderr=stderr,cwd=ROOT,
                        env=dict(os.environ,PYTORCH_ALLOC_CONF='expandable_segments:True'))
                    immutable(window/'actor_launch.json',dict(pid=proc.pid,timestamp=now()))
                    code = proc.wait()
                actor_seconds = time.monotonic()-start
                immutable(window/'actor_exit.json',dict(exit_code=code,timestamp=now(),seconds=actor_seconds))
            else:
                saved_exit=read(window/'actor_exit.json')
                code,actor_seconds=saved_exit['exit_code'],saved_exit['seconds']
            assert code==0, f'Actor window failed: {window}'
            assert memory.nv.nvmlDeviceGetMemoryInfo(memory.handle).used<4*1024**3, 'Actor failed to release GPU'
            checkpoint = window/'checkpoint'
            verify_checkpoint(checkpoint)
            digest = sha256(checkpoint/'adapter/adapter_model.safetensors')
            after = advance(state,groups,decisions,digest,cfg['groups_per_window']//cfg['mini_prompts'])
            start = time.monotonic()
            rollout.llm.wake_up()
            wake_seconds = time.monotonic()-start
            rollout.set_policy(checkpoint/'adapter',digest,after['policy_windows']+1)
            previous,sync = probe(rollout,probe_prompt,window/'sync',previous)
            metrics.update(sleep_seconds=sleep_seconds,wake_seconds=wake_seconds,
                           actor_process_seconds=actor_seconds,sleeping_memory_bytes=sleeping_bytes,
                           sync_seconds=sync['seconds'],**memory.result())
            commit = dict(state_before=state,state_after=after,metrics=metrics,checkpoint=str(checkpoint),
                artifacts=[record(window/p) for p in ('selection.json','rollout_metrics.json','update/old_frozen.json',
                    'update/update.json','actor_result.json','checkpoint/COMMITTED.json','sync/receipt.json')])
            immutable(window/'commit.json',commit)
            state = after
            durable(out/'checkpoint.json',dict(state=state,checkpoint=str(checkpoint),commit=record(window/'commit.json')))
            event(out,'window_committed',policy_windows=state['policy_windows'],optimizer_steps=state['optimizer_steps'],
                training_groups=state['training_groups'],generated_groups=state['generated_groups'],
                output_tokens=state['output_tokens'],policy_version=digest)
            durable(INDEX/out.name/'progress.json',dict(state=state,last_window=metrics,timestamp=now()))
            monitor(rollout,out,state,cfg)
            if cfg.get('pause_after_windows')==state['policy_windows'] and attempt.name=='attempt_001':
                immutable(out/'pause_ready.json',dict(state=state,checkpoint=str(checkpoint),pid=os.getpid(),
                    next_encounters=[encounter(stream,i,out.name,state['policy_version']) for i in range(state['cursor'],state['cursor']+8)]))
                event(out,'pause_ready',pid=os.getpid())
                signal.pause()
                raise RuntimeError('External SIGTERM and fresh-process resume required')
        rollout.close()
    if cfg['mode']!='smoke':
        directory=out/'final_reload'
        if not (directory/'result.json').exists():
            directory.mkdir(exist_ok=True)
            argv=[sys.executable,str(ROOT/'scripts/reload_stage4.py'),'--run',str(out)]
            immutable(directory/'command.json',argv)
            with (directory/'stdout.log').open('x') as stdout,(directory/'stderr.log').open('x') as stderr:
                code=subprocess.call(argv,stdout=stdout,stderr=stderr,cwd=ROOT,
                    env=dict(os.environ,PYTORCH_ALLOC_CONF='expandable_segments:True'))
            immutable(directory/'exit.json',dict(exit_code=code,timestamp=now()))
        assert read(directory/'exit.json')['exit_code']==0 and read(directory/'result.json')['result']=='PASS'
    from .common import seal
    seal(out,dict(status='SMOKE_PASS' if cfg['mode']=='smoke' else 'PILOT_PASS' if cfg['mode']=='pilot' else 'FULL_PASS',
        run_id=out.name,state=state,mode=cfg['mode'],sampling_mode=cfg['sampling_mode']))
