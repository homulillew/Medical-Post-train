#!/usr/bin/env python3
"""Prepare a fresh, matched512-group pilot pair after both raw smoke gates."""
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,immutable,durable,inherited,prepare,now


def main():
    receipts={v:record(INDEX/f'{v}_smoke_verification.json') for v in ('vanilla','dynamic')}
    for v,ref in receipts.items():
        r=read(ref['path'])
        assert r['result']=='PASS' and r['scope']=='SMOKE' and r['sampling_mode']==v and r['real_resume']
    assert not (INDEX/'pilot_pair.json').exists(), 'Existing pilot pair must be inspected, not replaced'
    cfg=inherited()
    cfg.update(mode='pilot',stream_domain='stage4:pilot:pair1',target_training_groups=512,
        pause_after_windows=2,validation_protocol=record(INDEX/'validation_protocol.json'),
        checkpoint_interval_windows=1,smoke_receipts=receipts)
    sample=Path(read(INDEX/'active_smoke_vanilla.json')['path'])/'windows/0000/checkpoint'
    checkpoint_bytes=sum(p.stat().st_size for p in sample.rglob('*') if p.is_file())
    free=shutil.disk_usage(sample).free
    projection=checkpoint_bytes*(128+1250)*1.1+100*1024**3
    assert free>projection, f'Insufficient evidence storage: {free} < {projection}'
    runs={}
    for variant in ('vanilla','dynamic'):
        out=prepare('pilot_'+variant,'PILOT',dict(cfg,sampling_mode=variant))
        runs[variant]=dict(run_id=out.name,path=str(out),config=record(out/'config.json'),manifest=record(out/'manifest.json'))
        durable(INDEX/f'active_pilot_{variant}.json',runs[variant])
    pair=dict(mode='PILOT',runs=runs,created_at=now(),shared_config=cfg,
        resource_preflight=dict(free_bytes=free,measured_checkpoint_bytes=checkpoint_bytes,
            pilot_and_formal_projection_with_reserve_bytes=projection),
        fresh_sft=True,formal_frozen=False,formal_training_groups=0,
        hypothesis='Assess stability, resource use and genuine clipping before formal comparison; validation superiority is not a pass condition')
    immutable(INDEX/'pilot_pair.json',pair)
    state=read(ROOT/'project_state.json')
    state['stages']['4']['status']='SMOKE_PASS'
    state['stages']['4']['run_ids']+= [r['run_id'] for r in runs.values()]
    durable(ROOT/'project_state.json',state)
    print({v:r['path'] for v,r in runs.items()})


if __name__=='__main__':
    main()
