#!/usr/bin/env python3
"""Six independent native GRPO optimizer diagnostics on the retained original batch."""
from pathlib import Path
import os,sys,subprocess,traceback
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,seal,now

def main():
    idx=ROOT/'experiments/stage4';p=read(idx/'grpo_diagnostic_preregistration_v1.json');out=Path(p['artifact_path'])
    for ref in p['diagnostic_sources'].values():assert record(ref['path'])==ref
    if len(sys.argv)>1 and sys.argv[1]=='actor':
        from types import SimpleNamespace
        from medical_posttrain.rl import actor
        from loss_objective_actor import Actor
        actor.Actor=Actor
        from run_stage4 import actor_diagnostic
        actor_diagnostic(SimpleNamespace(run=str(out),directory=sys.argv[2],fixed_old=sys.argv[3] if len(sys.argv)>3 else None))
        return
    try:
        from loss_objective_metrics import metrics,health,select
        from loss_objective_replay import update
        from verify_stage4 import raw_batch,checkpoint
        cfg=read(out/'config.json');groups=raw_batch(Path(p['original_batch']['path']).parent,cfg)
        conditions=[];old=None
        durable(out/'status.json',dict(status='RUNNING_DIAGNOSTIC',timestamp=now(),pid=os.getpid()))
        for c in p['grid']:
            d=out/f"lr_{c['learning_rate']:g}_clip_{c['clip']:g}";d.mkdir()
            cc=dict(cfg,learning_rate=c['learning_rate'],clip_ratio_low=c['clip'],clip_ratio_high=c['clip'])
            immutable(d/'config.json',cc)
            cmd=[sys.executable,str(Path(__file__)),'actor',str(d)]+([str(old)] if old else [])
            immutable(d/'command.json',cmd)
            env=dict(os.environ,HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
            env.pop('PYTORCH_ALLOC_CONF',None);env.pop('PYTORCH_CUDA_ALLOC_CONF',None)
            with (d/'stdout.log').open('x') as so,(d/'stderr.log').open('x') as se:
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=so,stderr=se,env=env)
                immutable(d/'launch.json',dict(pid=proc.pid,timestamp=now()));code=proc.wait()
            immutable(d/'exit.json',dict(exit_code=code,timestamp=now()))
            if code:
                conditions.append(dict(**c,health=dict(healthy=False,failed=['process_failure'],active_clipping=False),directory=str(d)))
            else:
                update(d/'update',groups,cc);checkpoint(d/'checkpoint')
                old=old or d/'update/old.npz'
                m=metrics(d/'update',cc);h=health(m,p['health_gates'])
                immutable(d/'optimization_health.json',dict(metrics=m,health=h))
                conditions.append(dict(**c,metrics=m,health=h,directory=str(d),native_replay='PASS'))
            durable(idx/'grpo_diagnostic_progress_v1.json',dict(status='RUNNING',completed=len(conditions),total=6,conditions=conditions,timestamp=now()))
        chosen=select(conditions,p['health_gates'])
        result=dict(status='DIAGNOSTIC_PASS',timestamp=now(),run_id=out.name,conditions=conditions,selected=dict(learning_rate=chosen['learning_rate'],clip=chosen['clip']),
            selection_rule=p['selection_rule'],preregistration=record(idx/'grpo_diagnostic_preregistration_v1.json'),new_rollouts=0,formal_training_groups=0)
        immutable(idx/'grpo_optimization_diagnostic_v1.json',result);seal(out,result)
        durable(idx/'grpo_diagnostic_progress_v1.json',dict(status='DIAGNOSTIC_PASS',completed=6,total=6,selected=result['selected'],timestamp=now()))
    except BaseException as exc:
        durable(out/'diagnostic_failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),timestamp=now()))
        durable(out/'status.json',dict(status='FAILED',timestamp=now()));raise
if __name__=='__main__':main()
