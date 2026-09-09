#!/usr/bin/env python3
"""Fresh-process native final checkpoint reload and finite teacher-forcing probe."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import read,record,immutable,validate


def main():
    import torch
    from medical_posttrain.rl.actor import Actor
    from medical_posttrain.rl.online import verify_checkpoint
    from medical_posttrain.training.lora import MemoryMonitor,restore_rng
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True,type=Path)
    out=p.parse_args().run
    cfg=read(out/'config.json')
    validate(cfg)
    state=read(out/'checkpoint.json')['state']
    assert state['training_groups']==cfg['target_training_groups']
    window=out/'windows'/f'{state["policy_windows"]-1:04d}'
    checkpoint=window/'checkpoint'
    marker=verify_checkpoint(checkpoint)
    directory=out/'final_reload'
    with MemoryMonitor() as memory:
        actor=Actor(cfg,directory/'native',checkpoint/'adapter',checkpoint)
        loaded=read(directory/'native/loaded_identity.json')
        assert loaded['trainable_digest']==marker['trainable_digest']
        assert loaded['optimizer_digest']==marker['optimizer_digest']
        assert loaded['scheduler']==marker['scheduler']
        assert loaded['optimizer_steps']==[state['optimizer_steps']]
        restore_rng(torch.load(checkpoint/'rng.pt',weights_only=False))
        rows=read(window/'selection.json')['groups'][0]['responses'][:2]
        probes=[]
        with torch.no_grad():
            for row in rows:
                lp,raw,_=actor.logprobs(row)
                assert torch.isfinite(lp).all() and torch.isfinite(raw).all()
                probes.append(dict(trajectory_id=row['trajectory_id'],response_tokens=len(lp),
                    current_logprobs=lp.cpu().tolist(),raw_logprobs=raw.cpu().tolist()))
        immutable(directory/'probes.json',probes)
        immutable(directory/'result.json',dict(result='PASS',loaded_identity=loaded,
            checkpoint=record(checkpoint/'COMMITTED.json'),finite_logprobs=True,
            response_tokens=sum(p['response_tokens'] for p in probes),probes=record(directory/'probes.json'),
            optimizer_steps_added=0,**memory.result()))
        actor.close()


if __name__=='__main__':
    main()
