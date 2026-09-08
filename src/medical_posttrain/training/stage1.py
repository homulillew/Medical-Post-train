"""Single-GPU, full-epoch BF16 LoRA training with auditable cursor recovery."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import threading
import time
import traceback
import zipfile

from medical_posttrain.data.stage1 import collate
from medical_posttrain.evidence import sha256, write_json
from medical_posttrain.training.lora import TARGETS, MemoryMonitor, seed_all, parameters, digest, rng_state, restore_rng

def utc(): return datetime.now(timezone.utc).isoformat()

def read_rows(path):
    with open(path) as f: return [json.loads(line) for line in f]

def json_digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

class Run:
    def __init__(self,path):
        self.path=Path(path); self.config=json.loads((self.path/'config.json').read_text())
        self.manifest=json.loads((self.path/'manifest.json').read_text())
        assert sha256(self.path/'config.json')==self.manifest['config_sha256']
        self.attempt=self.path/f'attempt_{len(list(self.path.glob("attempt_*")))+1:03d}'
        self.attempt.mkdir(exist_ok=False)
        self.stop=False; self.heartbeat_stop=threading.Event()
        self.latest_progress={}
        write_json(self.attempt/'provenance.json',dict(timestamp=utc(),pid=os.getpid(),source_hashes={p:sha256(p) for p in self.manifest['source_hashes']},parent_config_sha256=self.manifest['config_sha256']))
        with zipfile.ZipFile(self.attempt/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p in self.manifest['source_hashes']:archive.write(p,p)

    def metric(self,**row):
        row=dict(timestamp=utc(),attempt=self.attempt.name,**row)
        with (self.attempt/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush()
        print(json.dumps(row,allow_nan=False),flush=True)
        return row

    def start_heartbeat(self):
        def heartbeat():
            while not self.heartbeat_stop.wait(5):
                write_json(self.path/'heartbeat.json',dict(timestamp=utc(),pid=os.getpid(),attempt=self.attempt.name,**self.latest_progress))
        self.thread=threading.Thread(target=heartbeat,daemon=True); self.thread.start()
        for sig in (signal.SIGTERM,signal.SIGINT):
            signal.signal(sig,lambda *a:setattr(self,'stop',True))

def load(config, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from peft import LoraConfig,get_peft_model,PeftModel
    seed_all(config['seed'])
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(config['model'],dtype=torch.bfloat16,attn_implementation='sdpa',device_map='cuda',local_files_only=True)
    assert model.config.model_type=='qwen3' and model.config.hidden_size==4096
    assert next(model.parameters()).dtype==torch.bfloat16
    if adapter: model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    else: model=get_peft_model(model,LoraConfig(r=32,lora_alpha=config['lora_alpha'],lora_dropout=0,target_modules=TARGETS,task_type='CAUSAL_LM'))
    assert model.peft_config['default'].r==32 and model.peft_config['default'].lora_alpha==config['lora_alpha']
    assert set(model.peft_config['default'].target_modules)==set(TARGETS)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==87293952
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    return model,tokenizer

def tensor_state_digest(value):
    """Stable optimizer/RNG identity, independent of torch.save container bytes."""
    import torch
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if isinstance(x,torch.Tensor):
            y=x.detach().cpu().contiguous();h.update(str((str(y.dtype),tuple(y.shape))).encode());h.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,np.ndarray): h.update(x.tobytes())
        elif isinstance(x,dict):
            for k in sorted(x,key=str):h.update(str(k).encode());visit(x[k])
        elif isinstance(x,(list,tuple)):
            for v in x:visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()

def validate(model,tokenizer,rows,run,step,scope):
    import torch
    model.eval(); numerator=0.;tokens=0;by_source={}
    start=time.monotonic()
    with torch.no_grad():
        for row in rows:
            data={k:v.to('cuda') for k,v in collate([row],tokenizer.pad_token_id).items()}
            n=int((data['labels'][:,1:]!=-100).sum())
            loss=float(model(**data).loss)
            if not math.isfinite(loss) or loss<0:raise ValueError('Invalid validation loss')
            numerator+=loss*n;tokens+=n
            entry=by_source.setdefault(row['source'],dict(loss_sum=0.,tokens=0,examples=0))
            entry['loss_sum']+=loss*n;entry['tokens']+=n;entry['examples']+=1
    model.train()
    return run.metric(event='validation',global_step=step,scope=scope,loss=numerator/tokens,examples=len(rows),supervised_tokens=tokens,seconds=time.monotonic()-start,by_source={k:dict(loss=v['loss_sum']/v['tokens'],tokens=v['tokens'],examples=v['examples']) for k,v in by_source.items()})

def update(model,tokenizer,opt,scheduler,rows,config):
    import torch
    model.train(); opt.zero_grad(set_to_none=True)
    start=time.monotonic();total=sum(r['total_tokens'] for r in rows)
    supervised=sum(sum(x!=-100 for x in r['labels'][1:]) for r in rows)
    numerator=0.;lr=opt.param_groups[0]['lr'];padded=0
    execution_rows=sorted(rows,key=lambda r:r['total_tokens']) if config.get('sort_within_effective_batch') else rows
    device=next(model.parameters()).device
    for i in range(0,len(rows),config['microbatch']):
        group=execution_rows[i:i+config['microbatch']]
        data={k:v.to(device) for k,v in collate(group,tokenizer.pad_token_id).items()}
        n=int((data['labels'][:,1:]!=-100).sum());padded+=data['input_ids'].numel()
        loss=model(**data).loss
        if not torch.isfinite(loss) or loss.item()<0:raise ValueError('Invalid training loss')
        (loss*(n/supervised)).backward();numerator+=loss.item()*n
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['max_grad_norm'])
    if not torch.isfinite(grad): raise ValueError('Nonfinite gradient norm')
    opt.step();scheduler.step()
    if device.type=='cuda':torch.cuda.synchronize()
    elapsed=time.monotonic()-start
    return dict(loss=numerator/supervised,grad_norm=float(grad),learning_rate=lr,next_learning_rate=opt.param_groups[0]['lr'],examples=len(rows),sample_ids=[r['sample_id'] for r in rows],processed_tokens=total,supervised_tokens=supervised,padded_tokens=padded,update_seconds=elapsed,tokens_per_second=total/elapsed)

def checkpoint(run,model,opt,scheduler,state,tag):
    import torch
    start=time.monotonic();path=run.path/'checkpoints'/tag
    tmp=path.with_name(path.name+'.'+run.attempt.name+'.incomplete');tmp.mkdir(parents=True,exist_ok=False)
    adapter=tmp/'adapter';model.save_pretrained(adapter,safe_serialization=True)
    saved=dict(**state,optimizer=opt.state_dict(),scheduler=scheduler.state_dict(),rng=rng_state(),trainable_digest=digest(parameters(model)),config_sha256=run.manifest['config_sha256'])
    torch.save(saved,tmp/'state.pt')
    write_json(tmp/'integrity.json',dict(files=[dict(path=str(p.relative_to(tmp)),sha256=sha256(p),bytes=p.stat().st_size) for p in tmp.rglob('*') if p.is_file()],global_step=state['global_step'],sample_cursor=state['sample_cursor'],covered_ids_sha256=json_digest(state['covered_ids']),trainable_digest=saved['trainable_digest'],optimizer_digest=tensor_state_digest(saved['optimizer']),scheduler_digest=tensor_state_digest(saved['scheduler']),rng_digest=tensor_state_digest(saved['rng']),config_sha256=run.manifest['config_sha256']))
    write_json(tmp/'COMPLETE.json',dict(timestamp=utc(),integrity_sha256=sha256(tmp/'integrity.json')))
    # fsync checkpoint files and directory before publishing the atomic marker path.
    for p in tmp.rglob('*'):
        if p.is_file():
            with p.open('rb') as f:os.fsync(f.fileno())
    os.replace(tmp,path)
    directory_fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(directory_fd)
    finally:os.close(directory_fd)
    write_json(run.path/'latest_checkpoint.json',dict(path=str(path),global_step=state['global_step'],sample_cursor=state['sample_cursor'],integrity_sha256=sha256(path/'integrity.json')))
    run.metric(event='checkpoint',path=str(path),global_step=state['global_step'],sample_cursor=state['sample_cursor'],seconds=time.monotonic()-start,adapter_sha256=sha256(path/'adapter/adapter_model.safetensors'))
    # Sufficient local space: retain all valid checkpoints, including previous/final.
    return path

def check_checkpoint(path):
    path=Path(path); marker=json.loads((path/'COMPLETE.json').read_text())
    assert sha256(path/'integrity.json')==marker['integrity_sha256']
    integrity=json.loads((path/'integrity.json').read_text())
    for item in integrity['files']:
        assert (path/item['path']).stat().st_size==item['bytes'] and sha256(path/item['path'])==item['sha256']
    return integrity

def worker(path,resume=None):
    import torch
    from transformers import get_cosine_schedule_with_warmup
    run=Run(path);cfg=run.config;run.start_heartbeat();start=time.monotonic()
    write_json(run.path/'status.json',dict(status='RUNNING',attempt=run.attempt.name,pid=os.getpid(),started=utc(),resume=resume))
    try:
        if cfg['run_class']=='FORMAL':
            assert all(sha256(p)==expected for p,expected in run.manifest['source_hashes'].items()),'Formal code/environment inputs changed; do not splice implementations'
            assert os.environ.get('PYTORCH_ALLOC_CONF')==cfg.get('pytorch_alloc_conf')
        for name in ('train','validation'):
            assert sha256(cfg[name+'_path'])==cfg[name+'_sha256']
        rows=read_rows(cfg['train_path']); val=read_rows(cfg['validation_path'])
        assert len(rows)==20000 and len({r['sample_id'] for r in rows})==20000
        assert all(r['total_tokens']<=cfg['max_sequence_length'] for r in rows+val)
        random.Random(cfg['seed']).shuffle(rows)
        rows=rows[:cfg['planned_examples']]
        order=[r['sample_id'] for r in rows];effective=cfg['microbatch']*cfg['gradient_accumulation']
        total_steps=math.ceil(len(rows)/effective)
        if cfg['run_class']=='FORMAL':assert len(rows)==20000 and cfg['planned_epochs']==1
        integrity=check_checkpoint(resume) if resume else None
        model,tokenizer=load(cfg,str(Path(resume)/'adapter') if resume else None)
        run.metric(event='load',seconds=time.monotonic()-start,trainable_parameters=87293952,base_dtype='bfloat16',adapter_dtypes=sorted({str(p.dtype) for p in model.parameters() if p.requires_grad}),initialization='RESUME' if resume else 'FRESH_BASE_NEW_LORA')
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=0,betas=(.9,.999),eps=1e-8)
        scheduler=get_cosine_schedule_with_warmup(opt,math.ceil(total_steps*cfg['warmup_ratio']),total_steps)
        initial_digest=digest(parameters(model))
        if resume:
            state=torch.load(Path(resume)/'state.pt',map_location='cpu',weights_only=False)
            assert state.pop('config_sha256')==run.manifest['config_sha256']
            assert state.pop('trainable_digest')==initial_digest
            saved_opt=state.pop('optimizer');saved_sched=state.pop('scheduler');saved_rng=state.pop('rng')
            opt.load_state_dict(saved_opt);scheduler.load_state_dict(saved_sched);restore_rng(saved_rng)
            assert tensor_state_digest(opt.state_dict())==integrity['optimizer_digest']
            assert tensor_state_digest(scheduler.state_dict())==integrity['scheduler_digest']
            assert tensor_state_digest(rng_state())==integrity['rng_digest']
            assert state['covered_ids']==order[:state['sample_cursor']]
            assert len(set(state['covered_ids']))==state['sample_cursor']
            run.metric(event='resume_restore',checkpoint=resume,global_step=state['global_step'],sample_cursor=state['sample_cursor'],optimizer_digest=integrity['optimizer_digest'],scheduler_digest=integrity['scheduler_digest'],rng_digest=integrity['rng_digest'],adapter_digest=initial_digest,coverage_sha256=json_digest(state['covered_ids']))
        else:
            state=dict(global_step=0,sample_cursor=0,covered_ids=[],canonical_updates=[],validation=[],initial_digest=initial_digest,order_sha256=json_digest(order),processed_tokens=0,supervised_tokens=0,active_seconds=0.)
            state['validation'].append(validate(model,tokenizer,val if cfg['run_class']=='FORMAL' else val[:64]+val[500:564],run,0,'initial'))
        assert state['order_sha256']==json_digest(order)
        last_save=time.monotonic();reference=run.path/'resume_reference.pt';ckpt=Path(resume) if resume else None
        with MemoryMonitor() as memory:
            while state['sample_cursor']<len(rows):
                cursor=state['sample_cursor'];group=rows[cursor:cursor+effective]
                row=update(model,tokenizer,opt,scheduler,group,cfg)
                state['global_step']+=1;state['sample_cursor']+=len(group);state['covered_ids'].extend(r['sample_id'] for r in group)
                state['processed_tokens']+=row['processed_tokens'];state['supervised_tokens']+=row['supervised_tokens'];state['active_seconds']+=row['update_seconds']
                row=run.metric(event='update',global_step=state['global_step'],sample_cursor=state['sample_cursor'],coverage_fraction=state['sample_cursor']/len(rows),**row,**memory.result())
                state['canonical_updates'].append(row)
                run.latest_progress=dict(global_step=state['global_step'],sample_cursor=state['sample_cursor'],coverage_fraction=state['sample_cursor']/len(rows))
                write_json(run.path/'progress.json',dict(**run.latest_progress,processed_tokens=state['processed_tokens'],supervised_tokens=state['supervised_tokens']))
                if resume and reference.exists() and state['global_step']==cfg.get('pause_after_updates',0)+1:
                    ref=torch.load(reference,map_location='cpu',weights_only=False)
                    actual=parameters(model)
                    max_error=max((actual[k]-v).abs().max().item() for k,v in ref['parameters'].items())
                    assert all(torch.allclose(actual[k],v,atol=1e-6,rtol=1e-4) for k,v in ref['parameters'].items())
                    assert abs(row['loss']-ref['loss'])<1e-5 and scheduler.state_dict()==ref['scheduler']
                    assert row['sample_ids']==ref['sample_ids'] and state['sample_cursor']==ref['sample_cursor']
                    receipt=run.metric(event='real_data_resume_parity',checkpoint=resume,global_step=state['global_step'],sample_cursor=state['sample_cursor'],parameter_max_abs_error=max_error,loss_abs_error=abs(row['loss']-ref['loss']),same_next_sample_ids=True,scheduler_equal=True,reference_parameters_sha256=sha256(reference))
                    write_json(run.path/'resume_receipt.json',receipt)
                pause=(not resume and cfg.get('pause_after_updates')==state['global_step'])
                final=state['sample_cursor']==len(rows)
                save=pause or final or run.stop or state['global_step']%cfg['checkpoint_interval']==0 or time.monotonic()-last_save>=900
                if save:
                    if not pause:
                        eval_rows=val if final and cfg['run_class']=='FORMAL' else val[:64]+val[500:564]
                        state['validation'].append(validate(model,tokenizer,eval_rows,run,state['global_step'],'final' if final else 'monitor'))
                    ckpt=checkpoint(run,model,opt,scheduler,state,'final' if final else f'step_{state["global_step"]:06d}')
                    last_save=time.monotonic()
                if pause:
                    group=rows[state['sample_cursor']:state['sample_cursor']+effective]
                    reference_row=update(model,tokenizer,opt,scheduler,group,cfg)
                    torch.save(dict(parameters=parameters(model),loss=reference_row['loss'],scheduler=scheduler.state_dict(),sample_ids=reference_row['sample_ids'],sample_cursor=state['sample_cursor']+len(group)),reference)
                    run.metric(event='reference_only_update_excluded_from_coverage',global_step=state['global_step']+1,**reference_row)
                    write_json(run.path/'status.json',dict(status='PAUSED',reason='Planned real-data process termination and resume parity',checkpoint=str(ckpt),ended=utc()))
                    return
                if run.stop and not final:
                    write_json(run.path/'status.json',dict(status='PAUSED',reason='Signal at safe update boundary',checkpoint=str(ckpt),ended=utc()))
                    return
        assert state['covered_ids']==order and len(set(order))==len(rows)
        final_digest=digest(parameters(model));assert final_digest!=state['initial_digest']
        canonical_text=''.join(json.dumps(row,allow_nan=False)+'\n' for row in state['canonical_updates'])
        if (run.path/'canonical_metrics.jsonl').exists():assert (run.path/'canonical_metrics.jsonl').read_text()==canonical_text
        else:(run.path/'canonical_metrics.jsonl').write_text(canonical_text)
        write_json(run.path/'coverage.json',dict(planned_ids=order,covered_ids=state['covered_ids'],missing_ids=[],fraction=1.,epoch=1,processed_tokens=state['processed_tokens'],supervised_tokens=state['supervised_tokens']))
        summary=dict(run_id=run.manifest['run_id'],run_class=cfg['run_class'],status='FULL_BUDGET_REACHED' if cfg['run_class']=='FORMAL' else 'TRAINING_PASS',examples=len(rows),unique_examples=len(set(order)),global_step=state['global_step'],coverage_fraction=1.,epoch=1,processed_tokens=state['processed_tokens'],supervised_tokens=state['supervised_tokens'],update_seconds=state['active_seconds'],effective_tokens_per_update_second=state['processed_tokens']/state['active_seconds'],attempt_wall_seconds=time.monotonic()-start,final_checkpoint=str(ckpt),final_adapter=str(ckpt/'adapter'),adapter_sha256=sha256(ckpt/'adapter/adapter_model.safetensors'),initial_trainable_digest=state['initial_digest'],final_trainable_digest=final_digest,validation=state['validation'],**memory.result())
        write_json(run.path/'summary.json',summary)
        write_json(run.path/'status.json',dict(status=summary['status'],ended=utc()))
    except BaseException:
        write_json(run.attempt/'failure.json',dict(timestamp=utc(),traceback=traceback.format_exc()))
        write_json(run.path/'status.json',dict(status='FAILED',attempt=run.attempt.name,ended=utc(),reason='See attempt failure.json; retained checkpoints may permit recovery'))
        raise
    finally:
        run.heartbeat_stop.set();run.thread.join()
