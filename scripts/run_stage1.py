"""Prepare, detach, inspect or resume a real Stage 1 run."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import uuid
import zipfile
from medical_posttrain.evidence import sha256,write_json
from medical_posttrain.runtime import environment

ROOT=Path(__file__).resolve().parents[1]
BULK=Path('/data/WSH/medical-post-train-artifacts/runs')

def launch(path,resume=None):
    status=json.loads((path/'status.json').read_text()) if (path/'status.json').exists() else {}
    if status.get('status')=='RUNNING':
        try:os.kill(status['pid'],0)
        except ProcessLookupError:pass
        else:raise RuntimeError('Worker still running')
    index=len(list(path.glob('worker_*.stdout.log')))+1
    command=[sys.executable,str(ROOT/'scripts/run_stage1.py'),'worker','--run',str(path)]
    if resume:command+=['--checkpoint',str(resume)]
    env=os.environ.copy();env['PATH']=str(Path(sys.executable).parent)+os.pathsep+env.get('PATH','');env['TOKENIZERS_PARALLELISM']='false'
    config=json.loads((path/'config.json').read_text())
    if config.get('pytorch_alloc_conf'):env['PYTORCH_ALLOC_CONF']=config['pytorch_alloc_conf']
    with (path/f'worker_{index:03d}.stdout.log').open('x') as out,(path/f'worker_{index:03d}.stderr.log').open('x') as err:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=out,stderr=err,start_new_session=True)
    write_json(path/f'launch_{index:03d}.json',dict(pid=p.pid,command=command,started=datetime.now(timezone.utc).isoformat(),resume=str(resume) if resume else None,pytorch_alloc_conf=env.get('PYTORCH_ALLOC_CONF')))
    print(json.dumps(dict(run=str(path),pid=p.pid)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['prepare','resume','worker','inspect']);parser.add_argument('--config');parser.add_argument('--run');parser.add_argument('--checkpoint');args=parser.parse_args()
    if args.action=='prepare':
        config=json.loads(Path(args.config).read_text());assert config['run_class'] in ('SMOKE','PILOT','FORMAL')
        assert config['lora_rank']==32 and config['lora_alpha'] in (32,64) and config['planned_epochs']==1
        assert config['model_revision']=='b968826d9c46dd6066d109eabc6255188de91218'
        assert sha256(config['data_verification'])==config['data_verification_sha256']
        data_receipt=json.loads(Path(config['data_verification']).read_text())
        assert data_receipt['status']=='PASS' and data_receipt['counts']['train']==dict(medical_o1=10000,huatuo=10000)
        dirty=subprocess.check_output(['git','status','--porcelain'],text=True,cwd=ROOT)
        if config['run_class']=='FORMAL':
            assert not dirty,'Formal source/config must be committed before preparation'
            assert config['planned_examples']==20000 and not config.get('pause_after_updates')
            for key in ('smoke_receipt','pilot_receipt'):
                receipt=json.loads(Path(config[key]).read_text());assert receipt['status']=='PASS'
        else:assert config['planned_examples']==(128 if config['run_class']=='SMOKE' else 1024)
        run_id='s1_'+config['run_class'].lower()+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
        path=BULK/run_id;path.mkdir(exist_ok=False);git=ROOT/'experiments/stage1'/run_id;git.mkdir(parents=True,exist_ok=False)
        write_json(path/'config.json',config)
        manifest=dict(run_id=run_id,stage=1,run_class=config['run_class'],purpose='Medical SFT one-epoch coverage experiment',created=datetime.now(timezone.utc).isoformat(),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip(),dirty_state=dirty,config_sha256=sha256(path/'config.json'),artifact_root=str(path),planned_budget=dict(examples=config['planned_examples'],epochs=1),stop_conditions=['complete planned unique-example epoch','nonfinite loss/grad','invalid data/config','unrecoverable resource error'],checkpoint_interval=config['checkpoint_interval'],wall_checkpoint_seconds=900,source_hashes={str(p.relative_to(ROOT)):sha256(p) for base in ('src','scripts','configs','env') for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py','.json','.yaml','.lock','.in')})
        write_json(path/'manifest.json',manifest);write_json(git/'manifest.json',manifest);write_json(git/'config.json',config);write_json(path/'environment.json',environment())
        with zipfile.ZipFile(path/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for file in manifest['source_hashes']:z.write(ROOT/file,file)
        launch(path)
    elif args.action=='worker':
        from medical_posttrain.training.stage1 import worker
        worker(args.run,args.checkpoint)
    elif args.action=='resume':
        path=Path(args.run)
        assert json.loads((path/'status.json').read_text())['status'] in ('PAUSED','FAILED','RUNNING')
        checkpoint=args.checkpoint or json.loads((path/'latest_checkpoint.json').read_text())['path']
        assert json.loads((Path(checkpoint)/'integrity.json').read_text())['sample_cursor']<=json.loads((path/'config.json').read_text())['planned_examples']
        launch(path,checkpoint)
    else:
        path=Path(args.run)
        for file in ('status.json','progress.json','heartbeat.json','latest_checkpoint.json'):
            if (path/file).exists():print(file,(path/file).read_text())

if __name__=='__main__':main()
