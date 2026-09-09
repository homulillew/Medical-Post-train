"""Audited recovery of incomplete actor transactions without discarding rollout."""
from pathlib import Path
import os
import subprocess

from .common import read,record,immutable,durable,now


def recover_actor_transaction(out):
    from .online import restore,verify_checkpoint
    from .common import validate
    out=Path(out)
    cfg=read(out/'config.json')
    state,checkpoint=restore(out,validate(cfg))
    assert read(out/'status.json')['status'] in ('FAILED','INTERRUPTED','RUNNING')
    # A stale RUNNING status is recoverable only after the real owner exited.
    for a in out.glob('attempt_*'):
        pid=read(a/'launch.json')['pid']
        proc=Path(f'/proc/{pid}/stat')
        assert not proc.exists() or proc.read_text().split()[2]=='Z', f'Owner still alive: {pid}'
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<100
    for reservation in (out/'validation').glob('*/reservation_*.json'):
        assert reservation.with_name(reservation.name.replace('reservation_','batch_')).exists(), 'Unreturned validation cost remains unknown'
    window=out/'windows'/f'{state["policy_windows"]:04d}'
    if not window.exists():
        journal=out/'recoveries'/f'{len(list((out/"recoveries").glob("*")))+1:03d}'
        journal.mkdir(parents=True)
        if (out/'final_reload').exists() and not (out/'final_reload/exit.json').exists():
            os.rename(out/'final_reload',journal/'partial_final_reload')
        elif (out/'final_reload/exit.json').exists() and read(out/'final_reload/exit.json')['exit_code']!=0:
            os.rename(out/'final_reload',journal/'failed_final_reload')
        immutable(journal/'recovery.json',dict(timestamp=now(),state=state,
            action='Resume committed boundary; any incomplete final reload is archived',committed_budget_added=0))
        durable(out/'status.json',dict(status='INTERRUPTED',timestamp=now(),recovery=record(journal/'recovery.json')))
        return dict(action='Resume committed boundary',state=state,archive=str(journal))
    assert not (window/'commit.json').exists()
    assert read(window/'state_before.json')==state
    for b in (window/'batches').glob('*'):
        assert (b/'raw.json').exists(), 'Unreturned generation cost is unknown; actor rollback cannot repair it'
    previous=read(out/'status.json')
    archive=window/'recovery'/f'{len(list((window/"recovery").glob("*")))+1:03d}'
    archive.mkdir(parents=True)
    keep_completed=all((window/p).exists() for p in ('checkpoint/COMMITTED.json','update/update.json'))
    if keep_completed:
        marker=verify_checkpoint(window/'checkpoint')
        assert marker['optimizer_step']==state['optimizer_steps']+2
        metadata=read(window/'checkpoint/controller.json')
        assert metadata['state_before']==state and metadata['run_id']==out.name
        for k in ('config','selection','update'):
            assert record(metadata[k]['path'])==metadata[k]
        if (window/'checkpoint_adoption.json').exists():
            assert read(window/'checkpoint_adoption.json')['source_marker']==record(window/'checkpoint/COMMITTED.json')
        else:
            immutable(window/'checkpoint_adoption.json',dict(timestamp=now(),state_before=state,
                source_marker=record(window/'checkpoint/COMMITTED.json'),
                action='Retain durable native checkpoint; fresh process must validate optimizer/scheduler/RNG before sync',
                optimizer_steps_to_reexecute=0))
        moved=['sync','recovered_actor','adoption_command.json','adoption_launch.json',
               'adoption_exit.json','adoption.stdout.log','adoption.stderr.log']
        if (window/'actor_result.json').exists() and read(window/'actor_result.json').get('recovered_from_committed_checkpoint'):
            moved.append('actor_result.json')
        action='Adopt complete renamed native actor transaction; no optimizer replay; validate reload then sync'
    else:
        moved=['actor','update','checkpoint','.tmp-checkpoint','actor_result.json','actor_command.json',
               'actor_launch.json','actor_exit.json','actor.stdout.log','actor.stderr.log','sync']
        action='Rollback uncommitted actor to previous trusted native checkpoint; reuse identical persisted rollout'
    refs=[]
    for name in moved:
        p=window/name
        if p.exists():
            refs.extend(record(f) for f in p.rglob('*') if f.is_file()) if p.is_dir() else refs.append(record(p))
            os.rename(p,archive/name)
    immutable(archive/'recovery.json',dict(timestamp=now(),action=action,previous_status=previous,
        state=state,source_checkpoint=str(checkpoint) if checkpoint else None,original_artifacts=refs,
        committed_budget_added=0,rollout_regenerated=False,
        note='Archived optimizer work/control cost remains in run evidence; it is excluded from committed training budget'))
    durable(out/'status.json',dict(status='INTERRUPTED',timestamp=now(),recovery=record(archive/'recovery.json')))
    return dict(action=action,archive=str(archive),state=state)
