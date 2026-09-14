"""Durable per-item execution; CPU-testable with an injected synthetic backend."""
from pathlib import Path
import time
from .core import freeze,read,ref,check_ref,digest,score,summarize
from .gates import reservation_guard

def execute_items(items,requests,backend,directory,checkpoint_id,run_id,resume=False,technical_retry=False,exam=True,batch_size=16):
    directory=Path(directory);responses=[]
    if [r['id'] for r in requests]!=[r['id'] for r in items]:raise ValueError('Request/gold ID mismatch')
    pending=[];saved_by_id={}
    for item,request in zip(items,requests):
        d=directory/digest(item['id'])[:24];action=reservation_guard(d,resume,technical_retry)
        if action=='REUSE_COMPLETE':
            saved=read(d/'response.json')
            if saved['checkpoint_id']!=checkpoint_id or saved['prompt_id']!=item['id']:raise ValueError('Resume identity changed')
            if read(d/'reservation.json')['request_sha256']!=digest(request):raise ValueError('Resume request changed')
            saved_by_id[item['id']]=saved;continue
        attempt=len(list(d.glob('attempt_*.json')))+1
        if action=='TECHNICAL_RETRY' and attempt>2:raise PermissionError('One explicit technical retry maximum')
        pending.append((item,request,d,action,attempt))
    for start in range(0,len(pending),batch_size):
        batch=pending[start:start+batch_size]
        for item,request,d,action,attempt in batch:
            reservation=dict(prompt_id=item['id'],checkpoint_id=checkpoint_id,run_id=run_id,request_sha256=digest(request),attempt=attempt,timestamp=time.time(),reason='explicit technical retry' if action=='TECHNICAL_RETRY' else 'first request')
            if action=='NEW':freeze(d/'reservation.json',reservation)
            freeze(d/f'attempt_{attempt:03d}.json',reservation)
        try:
            outputs=backend([x[1] for x in batch])
            if len(outputs)!=len(batch):raise ValueError('Incomplete batch response')
            for (item,request,d,action,attempt),raw in zip(batch,outputs):
                if raw['finish_reason'] not in ['stop','length'] or not isinstance(raw['raw_output'],str):raise ValueError('Incomplete/transport response')
                freeze(d/f'raw_attempt_{attempt:03d}.json',raw)
                response=score(raw,item,checkpoint_id,run_id) if exam else dict(raw,checkpoint_id=checkpoint_id,run_id=run_id,prompt_id=item['id'])
                freeze(d/'response.json',response);freeze(d/'receipt.json',dict(response=ref(d/'response.json'),raw=ref(d/f'raw_attempt_{attempt:03d}.json'),attempts=attempt))
                saved_by_id[item['id']]=response
        except BaseException as e:
            for item,request,d,action,attempt in batch:
                if not (d/'receipt.json').exists():freeze(d/f'error_{attempt:03d}.json',dict(error=repr(e),timestamp=time.time(),returned_cost='UNKNOWN unless raw attempt retained',silent_retry=False))
            raise
    responses=[saved_by_id[r['id']] for r in items]
    if not (directory/'complete.json').exists():freeze(directory/'complete.json',dict(n=len(responses),checkpoint_id=checkpoint_id,prompt_ids=[r['id'] for r in items],response_refs=[ref(directory/digest(r['id'])[:24]/'response.json') for r in items],summary=summarize(responses) if exam else None))
    return responses
