#!/usr/bin/env python3
"""Reconcile a stale convenience pointer from verified immutable window commits.

Only a dead owner may be reconciled. This never adopts an unsynced checkpoint:
that transaction remains the responsibility of the fault-tested recovery path.
"""
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,validate,now


def reconcile(out):
    from medical_posttrain.rl.online import restore
    out=Path(out)
    for attempt in out.glob('attempt_*'):
        pid=read(attempt/'launch.json')['pid'];proc=Path(f'/proc/{pid}/stat')
        assert not proc.exists() or proc.read_text().split()[2]=='Z', f'Owner {pid} still alive'
    state,checkpoint=restore(out,validate(read(out/'config.json')))
    if checkpoint is None:return state
    commit=checkpoint.parent/'commit.json'
    expected=dict(state=state,checkpoint=str(checkpoint),commit=record(commit))
    previous=read(out/'checkpoint.json') if (out/'checkpoint.json').exists() else None
    if previous!=expected:
        immutable(out/'pointer_reconciliations'/f'{time.time_ns()}.json',dict(timestamp=now(),
            previous=previous,canonical=expected,reason='Immutable synced commit is authoritative; repair only stale/missing convenience pointer',
            effective_budget_added=0,optimizer_steps_executed=0))
        durable(out/'checkpoint.json',expected)
    return state


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--run',required=True)
    s=reconcile(p.parse_args().run)
    print({k:v for k,v in s.items() if not k.endswith('exposure')})
