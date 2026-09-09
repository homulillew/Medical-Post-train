#!/usr/bin/env python3
"""Bounded read-only worker monitoring, syncing only committed progress to the index."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.evidence import write_json,now
from medical_posttrain.evidence.stage2 import read
root=Path(__file__).resolve().parents[1]
run_id=read(root/'experiments/stage3/selected_runs.json')['formal']
out=Path(read(root/'experiments/stage3'/run_id/'manifest.json')['artifact_root'])
deadline=time.monotonic()+7200
while time.monotonic()<deadline:
    status=read(out/'status.json');cp=read(out/'checkpoint.json')
    state=read(root/'project_state.json')
    if state['stages']['3']['status']=='FULL_RUNNING':
        state['stages']['3']['progress']['accepted_integration_mixed_groups']=cp['accepted_mixed_groups'];write_json(root/'project_state.json',state)
    print(json.dumps(dict(timestamp=now(),status=status['status'],batches=cp['batches'],generated=cp['generated_groups'],accepted=cp['accepted_mixed_groups'],
        all_wrong=cp['all_wrong_groups'],all_correct=cp['all_correct_groups'],invalid=cp['invalid_groups'],overflow=cp['overflow_mixed_groups'],output_tokens=sum(cp['output_tokens_by_disposition'].values())),ensure_ascii=False),flush=True)
    if status['status']!='RUNNING':break
    time.sleep(45)
else:raise SystemExit('Monitor deadline exceeded; does not terminate or redefine the worker budget')
