"""Checkpoint fault barriers and separate physical-work accounting."""
import hashlib
import json
import os
from pathlib import Path
import signal
from .common import read,record,immutable,now


def rng_digest(state):
    import numpy as np
    import torch
    def convert(x):
        if isinstance(x,torch.Tensor):
            return dict(dtype=str(x.dtype),shape=list(x.shape),values=x.cpu().tolist())
        if isinstance(x,np.ndarray):
            return dict(dtype=str(x.dtype),shape=list(x.shape),values=x.tolist())
        if isinstance(x,dict):return {k:convert(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [convert(v) for v in x]
        return x
    return hashlib.sha256(json.dumps(convert(state),sort_keys=True).encode()).hexdigest()


def fault_point(cfg,window,scenario,**fields):
    spec=cfg.get('fault_injection')
    if not spec:return
    assert cfg['mode']=='recovery_test', 'Fault barriers are forbidden in formal training'
    window=Path(window);out=window.parent.parent
    if spec['scenario']!=scenario or int(window.name)!=spec['window']:return
    directory=out/'fault'
    if (directory/'termination.json').exists():return
    immutable(directory/'ready.json',dict(scenario=scenario,window=str(window),pid=os.getpid(),
        process_group=os.getpgrp(),timestamp=now(),state_before=read(window/'state_before.json'),**fields))
    signal.pause()
    raise RuntimeError('Fault barrier requires physical kill and a new process')


def cost_ledger(out):
    out=Path(out)
    groups=responses=prompt=output=steps=0
    seconds=0.
    sources=[]
    for p in sorted((out/'windows').glob('*/batches/*/raw.json')):
        d=read(p);sources.append(record(p));seconds+=d['generation_seconds']
        for g in d['groups']:
            groups+=1;prompt+=g['prompt_tokens']
            responses+=len(g['responses']);output+=sum(r['output_tokens'] for r in g['responses'])
    optimizer_seconds=0.
    for p in sorted((out/'windows').rglob('update/events.jsonl')):
        sources.append(record(p))
        for line in p.read_text().splitlines():
            e=json.loads(line)
            if e['event']=='optimizer_step':steps+=1;optimizer_seconds+=e['seconds']
    controls=0;control_prompt=0
    for p in sorted(out.rglob('*sync*/first.json'))+sorted(out.rglob('*sync*/repeat.json')):
        d=read(p);controls+=len(d['token_ids']);control_prompt+=len(d['prompt_logprobs'])
    return dict(known_generated_groups=groups,known_generated_trajectories=responses,
        known_prompt_tokens=prompt,known_output_tokens=output,physical_optimizer_steps=steps,
        known_generation_seconds=seconds,known_optimizer_seconds=optimizer_seconds,
        known_control_output_tokens=controls,known_control_prompt_tokens=control_prompt,sources=sources,
        semantics='Physical work includes orphan optimizer events; committed budget is tracked separately. Unrecorded crash tail time is not invented.')
