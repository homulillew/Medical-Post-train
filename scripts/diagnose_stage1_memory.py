"""Allocator-only replay of 32 pilot updates plus longest real-sample batch."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import random
import subprocess
import time
import traceback
import uuid
from medical_posttrain.evidence import sha256,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',required=True);args=parser.parse_args()
    os.environ['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    import torch
    from transformers import get_cosine_schedule_with_warmup
    from medical_posttrain.training.stage1 import load,read_rows,update
    from medical_posttrain.training.lora import MemoryMonitor,parameters,digest
    from medical_posttrain.runtime import environment
    parent=Path(args.pilot);config=json.loads((parent/'config.json').read_text())
    run_id='s1_memory_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=parent.parent/run_id;out.mkdir(exist_ok=False);git=Path('experiments/stage1')/run_id;git.mkdir(exist_ok=False)
    write_json(git/'manifest.json',dict(run_id=run_id,stage=1,run_class='DIAGNOSTIC',purpose='Allocator-only pilot replay and worst real length check',parent_pilot=parent.name,config=config,pytorch_alloc_conf=os.environ['PYTORCH_ALLOC_CONF'],planned_updates=33,planned_examples=528,artifact_root=str(out),script_sha256=sha256(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()))
    write_json(out/'environment.json',environment());write_json(out/'status.json',dict(status='RUNNING'))
    start=time.monotonic()
    try:
        model,tokenizer=load(config);rows=read_rows(config['train_path']);random.Random(config['seed']).shuffle(rows)
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=0,betas=(.9,.999),eps=1e-8)
        scheduler=get_cosine_schedule_with_warmup(opt,2,64)
        reference=[json.loads(l) for l in (parent/'attempt_001/metrics.jsonl').open() if '"event": "update"' in l]
        metrics=[]
        with MemoryMonitor() as memory, (out/'metrics.jsonl').open('x') as f:
            for step in range(32):
                row=update(model,tokenizer,opt,scheduler,rows[step*16:(step+1)*16],config)
                row=dict(event='allocator_replay',global_step=step+1,loss_abs_difference_from_default=abs(row['loss']-reference[step]['loss']),**row,**memory.result())
                assert row['sample_ids']==reference[step]['sample_ids']
                metrics.append(row);f.write(json.dumps(row)+'\n');f.flush();print(json.dumps({k:v for k,v in row.items() if k!='sample_ids'}),flush=True)
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            # Compare to actual checkpoint parameter tensors, with PEFT's canonical
            # state-dict key mapping rather than guessing adapter parameter names.
            actual=parameters(model)
            saved=load_file(str(parent/'checkpoints/step_000032/adapter/adapter_model.safetensors'))
            set_peft_model_state_dict(model,saved)
            expected=parameters(model)
            error=max((actual[k]-expected[k]).abs().max().item() for k in actual)
            for name,p in model.named_parameters():
                if p.requires_grad:p.data.copy_(actual[name].to(p.device))
            assert error<=1e-5, f'Allocator replay changed parameters: {error}'
            worst=sorted(rows,key=lambda r:r['total_tokens'],reverse=True)[:16]
            worst_row=update(model,tokenizer,opt,scheduler,worst,config)
            worst_row=dict(event='longest_real_examples',maximum_sequence=max(r['total_tokens'] for r in worst),**worst_row,**memory.result())
            f.write(json.dumps(worst_row)+'\n');f.flush()
            assert memory.result()['remaining_at_peak_bytes']>=3*2**30,'Less than 3 GiB measured headroom'
        summary=dict(run_id=run_id,status='PASS',replayed_updates=32,diagnostic_updates=1,allocator='expandable_segments:True',parameter_max_abs_error=error,max_loss_abs_difference=max(r['loss_abs_difference_from_default'] for r in metrics),replay_tokens=sum(r['processed_tokens'] for r in metrics),replay_seconds=sum(r['update_seconds'] for r in metrics),longest_real_batch=worst_row,wall_seconds=time.monotonic()-start,**memory.result())
        write_json(git/'summary.json',summary);write_json(git/'artifacts_manifest.json',[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='status.json']);write_json(out/'status.json',dict(status='PASS'));print(json.dumps(summary),flush=True)
    except BaseException:
        write_json(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
