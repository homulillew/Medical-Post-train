"""Predeclared diagnostic pauses; never alter reward, eligibility or budget."""
from .common import read,record,immutable


class DiagnosticPause(RuntimeError):
    status='PAUSED_DIAGNOSTIC'


class RefillBlocked(RuntimeError):
    status='BLOCKED'


def assess(out,state,cfg):
    import numpy as np
    rule=cfg.get('health_protocol')
    if not rule:return
    n=state['policy_windows']
    failures=[];sources=[]
    streak=rule['clip_consecutive_windows']
    if n>=streak:
        paths=[out/'windows'/f'{i:04d}'/'update/update.json' for i in range(n-streak,n)]
        clips=[read(p)['minibatches'][1]['clip_fraction'] for p in paths]
        if all(x>rule['clip_threshold'] for x in clips):
            failures.append(dict(reason='Persistent second-mini objective clipping',values=clips))
            sources.extend(record(p) for p in paths)
    block=rule['length_block_windows'];count=rule['length_consecutive_blocks']
    if n>=block*count and n%block==0:
        p95=[]
        for start in range(n-block*count,n,block):
            paths=[p for i in range(start,start+block) for p in (out/'windows'/f'{i:04d}'/'batches').glob('*/raw.json')]
            lengths=[r['output_tokens'] for p in paths for g in read(p)['groups'] for r in g['responses']]
            p95.append(float(np.percentile(lengths,95)))
        if all(x>=cfg['sampling']['max_tokens']*rule['length_cap_fraction'] for x in p95):
            failures.append(dict(reason='Persistent response length near cap',block_p95=p95))
            sources.extend(record(p) for i in range(n-block*count,n) for p in (out/'windows'/f'{i:04d}'/'batches').glob('*/raw.json'))
    if failures:
        receipt=out/'health'/f'{n:04d}.json'
        if not receipt.exists():immutable(receipt,dict(state=state,failures=failures,sources=sources,action='Pause for documented diagnosis before any further rollout'))
        raise DiagnosticPause(str(receipt))
