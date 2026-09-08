"""Continue authorized Stage 1 runtime checks after the detached full epoch ends.

This does not mark VERIFIED/DONE, alter training, or start any later stage.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import psutil
from medical_posttrain.evidence import write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();root=Path(args.run)
    status_path=root/'postprocess_status.json'
    def status(value,**extra):
        row=dict(status=value,timestamp=datetime.now(timezone.utc).isoformat(),**extra);write_json(status_path,row);print(json.dumps(row),flush=True)
    try:
        status('WAITING_FOR_FORMAL_BUDGET')
        while True:
            state=json.loads((root/'status.json').read_text())
            if state['status']=='FULL_BUDGET_REACHED':break
            if state['status'] in ('FAILED','PAUSED'):raise RuntimeError('Training requires inspection/resume; do not run postprocessing on partial budget')
            time.sleep(30)
        summary=json.loads((root/'summary.json').read_text());assert summary['unique_examples']==20000 and summary['coverage_fraction']==1.
        launch=json.loads(sorted(root.glob('launch_*.json'))[-1].read_text())
        while psutil.pid_exists(launch['pid']):
            try:
                if psutil.Process(launch['pid']).status()==psutil.STATUS_ZOMBIE:break
            except psutil.NoSuchProcess:break
            time.sleep(2)
        status('FINAL_ADAPTER_RELOAD')
        command=[sys.executable,'scripts/check_stage1_adapter.py','--run',str(root),'--count','4']
        write_json(root/'reload_command.json',dict(command=command))
        with (root/'reload-check.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        status('PAIRED_HELDOUT_GENERATION')
        command=[sys.executable,'scripts/evaluate_stage1.py','--run',str(root),'--protocol','configs/stages/s1_evaluation.json']
        write_json(root/'evaluation_command.json',dict(command=command))
        with (root/'paired-generation.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        selected=json.loads(Path('experiments/stage1/selected_runs.json').read_text())
        evaluations=[]
        for path in Path('experiments/stage1').glob('s1_evaluation_*/manifest.json'):
            m=json.loads(path.read_text())
            if m['parent_run']==root.name and (path.parent/'summary.json').exists() and json.loads((path.parent/'summary.json').read_text())['status']=='PASS':evaluations.append(path.parent.name)
        assert len(evaluations)==1, 'Explicit review needed for multiple valid evaluation runs'
        selected['evaluation']=evaluations[0];write_json('experiments/stage1/selected_runs.json',selected)
        with (root/'plot.log').open('x') as log:subprocess.run(['/home/ubuntu/anaconda3/bin/python','scripts/plot_stage1.py','--run',str(root)],stdout=log,stderr=subprocess.STDOUT,check=True)
        status('RUNTIME_CHECKS_COMPLETE',evaluation_run=evaluations[0],next='Review actual cases, write report/interview/evidence ledger, seal artifacts, run stage verifier. Stage not DONE yet.')
    except BaseException:
        status('FAILED',traceback=traceback.format_exc());raise

if __name__=='__main__':main()
