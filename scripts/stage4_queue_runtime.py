#!/usr/bin/env python3
"""Run the frozen queue with the same operational launcher for both variants."""
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


class QueueSubprocessProxy:
    def run(self,args,*pos,**kwargs):
        if isinstance(args,(tuple,list)) and len(args)==5 and args[1]==str(ROOT/'scripts/run_stage4.py') and args[2:4]==['launch','--run']:
            from stage4_runtime_v2 import launch
            launch(Path(args[4]));return subprocess.CompletedProcess(args,0)
        return subprocess.run(args,*pos,**kwargs)
    def __getattr__(self,name):return getattr(subprocess,name)


if __name__=='__main__':
    from medical_posttrain.rl.common import read,durable,immutable,record,now
    from stage4_runtime_v2 import install
    import continue_stage4_formal as queue
    install();queue.subprocess=QueueSubprocessProxy()
    try:queue.run()
    except BaseException as exc:
        state=read(ROOT/'project_state.json');previous=state['stages']['4']['status']
        state['stages']['4']['status']='FAILED'
        immutable(ROOT/'experiments/stage4'/f'queue_project_failure_{time.time_ns()}.json',
                  dict(timestamp=now(),previous_status=previous,new_status='FAILED',error=repr(exc),queue_status=record(ROOT/'experiments/stage4/formal_queue_status.json')))
        durable(ROOT/'project_state.json',state)
        raise
