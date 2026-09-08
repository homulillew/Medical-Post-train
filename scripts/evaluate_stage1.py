"""Paired heldout generation with the verified vLLM LoRA numeric settings."""
from collections import Counter
from datetime import datetime,timezone
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
import uuid
from medical_posttrain.evidence import sha256,write_json

def measure(text,ids,finish,tokenizer):
    think=re.search(r'<think>([\s\S]*?)</think>',text)
    answer=re.search(r'<answer>([\s\S]*?)</answer>',text)
    strict=re.fullmatch(r'\s*<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>\s*',text)
    format_valid=bool(strict and text.count('<think>')==1 and text.count('</think>')==1 and text.count('<answer>')==1 and text.count('</answer>')==1 and answer and answer[1].strip())
    # An unclosed think block is retained as reasoning to measure runaway thinking.
    reasoning=think[1].strip() if think else text.split('<think>',1)[1].strip() if '<think>' in text and '</think>' not in text else ''
    ans=answer[1].strip() if answer else ''
    lines=[line.strip() for line in re.split(r'[。！？\n]',text) if len(line.strip())>=12]
    repetitions=Counter(lines)
    return dict(total_tokens=len(ids),reasoning_tokens=len(tokenizer.encode(reasoning,add_special_tokens=False)),answer_tokens=len(tokenizer.encode(ans,add_special_tokens=False)),think_closed=bool(think),answer_closed=bool(answer and ans),format_valid=format_valid,empty_answer=bool(answer and not ans),truncated=finish=='length',repetition_max=max(repetitions.values(),default=0),reasoning_measure='closed block or open-think remainder; excludes tags',answer_measure='closed answer block only; untagged Base answers are not zero-length semantic answers')

def aggregate(rows):
    import numpy as np
    result={}
    for source in ('all','medical_o1','huatuo'):
        selected=[r for r in rows if source=='all' or r['source']==source]
        item=dict(count=len(selected))
        for k in ('think_closed','answer_closed','format_valid','empty_answer','truncated'):
            item[k+'_fraction']=sum(r[k] for r in selected)/len(selected)
        for k in ('total_tokens','reasoning_tokens','answer_tokens'):
            values=[r[k] for r in selected];item[k]=dict(mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p90=float(np.percentile(values,90)),p95=float(np.percentile(values,95)),max=max(values),total=sum(values))
        item['reasoning_present_fraction']=sum(r['reasoning_tokens']>0 for r in selected)/len(selected)
        item['repetition_ge3_fraction']=sum(r['repetition_max']>=3 for r in selected)/len(selected)
        result[source]=item
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--protocol',default='configs/stages/s1_evaluation.json');args=parser.parse_args()
    parent=Path(args.run);config=json.loads((parent/'config.json').read_text());sft=json.loads((parent/'summary.json').read_text());protocol=json.loads(Path(args.protocol).read_text())
    assert config['run_class']=='FORMAL' and sft['coverage_fraction']>=.99
    run_id='s1_evaluation_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=parent.parent/run_id;out.mkdir(exist_ok=False);git=Path('experiments/stage1')/run_id;git.mkdir(exist_ok=False)
    manifest=dict(run_id=run_id,stage=1,run_class='EVALUATION',parent_run=sft['run_id'],parent_adapter_sha256=sft['adapter_sha256'],protocol=protocol,protocol_sha256=sha256(args.protocol),validation_sha256=config['validation_sha256'],script_sha256=sha256(__file__),git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),artifact_root=str(out),started=datetime.now(timezone.utc).isoformat())
    write_json(git/'manifest.json',manifest);write_json(out/'status.json',dict(status='RUNNING'))
    for key,value in dict(VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_USE_V2_MODEL_RUNNER='0',VLLM_BATCH_INVARIANT='1').items():os.environ[key]=value
    try:
        from transformers import AutoTokenizer
        from vllm import LLM,SamplingParams
        from vllm.lora.request import LoRARequest
        from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
        from medical_posttrain.training.lora import MemoryMonitor
        from medical_posttrain.verification.vllm_probe import serialize
        assert get_lora_op_configs('shrink',1,23,4096,32,3)['split_k']==1
        tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
        assert sha256(config['validation_path'])==config['validation_sha256']
        val=[json.loads(line) for line in open(config['validation_path'])]
        count=protocol['prompts_per_source'];selected=val[:count]+val[500:500+count]
        prompts=[tokenizer.apply_chat_template(r['messages'][:-1],tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in selected]
        write_json(out/'prompts.json',[dict(sample_id=r['sample_id'],source=r['source'],messages=r['messages'][:-1],prompt_ids=tokenizer.encode(p,add_special_tokens=False),reference=r['messages'][-1]['content']) for r,p in zip(selected,prompts)])
        adapter=Path(sft['final_adapter']);assert sha256(adapter/'adapter_model.safetensors')==sft['adapter_sha256']
        request=LoRARequest('medical_sft_'+sft['adapter_sha256'][:12],1,str(adapter))
        llm_args=dict(model=config['model'],tokenizer=config['model'],dtype='bfloat16',tensor_parallel_size=1,max_model_len=protocol['max_model_len'],max_num_seqs=protocol['max_num_seqs'],gpu_memory_utilization=protocol['gpu_memory_utilization'],enable_lora=True,max_lora_rank=32,enforce_eager=True,seed=protocol['seed'],enable_prefix_caching=False,generation_config='vllm')
        write_json(out/'vllm_args.json',llm_args)
        results={};timings={};start=time.monotonic()
        with MemoryMonitor() as memory:
            llm=LLM(**llm_args);cold=time.monotonic()-start
            control=SamplingParams(temperature=0,max_tokens=8,prompt_logprobs=5)
            controls={}
            for name,req in [('base',None),('sft',request),('base_negative',None),('sft_repeat',request)]:
                controls[name]=serialize(llm.generate(prompts[:1],control,lora_request=req,use_tqdm=False)[0]);write_json(out/f'identity_{name}.json',controls[name])
            assert controls['base']['token_ids']==controls['base_negative']['token_ids']
            assert controls['sft']['token_ids']==controls['sft_repeat']['token_ids']
            def delta(a,b):return max(abs(x[k]-y[k]) for x,y in zip(controls[a]['prompt_logprobs'][1:],controls[b]['prompt_logprobs'][1:]) for k in x.keys()&y.keys())
            assert delta('base','sft')>1e-5 and delta('sft','sft_repeat')<=1e-4
            limits=[protocol['primary_response_cap']]
            for limit in limits:
                for model,req in [('base',None),('sft',request)]:
                    key=f'{model}_{limit}';records=[];begin=time.monotonic()
                    params=SamplingParams(temperature=protocol['temperature'],top_p=protocol['top_p'],top_k=protocol['top_k'],max_tokens=limit,seed=protocol['seed'])
                    # Bounded chunks persist completed outputs even if a later chunk fails.
                    with (out/f'{key}.jsonl').open('x') as f:
                        for offset in range(0,len(prompts),16):
                            outputs=llm.generate(prompts[offset:offset+16],params,lora_request=req,use_tqdm=False)
                            for source,result in zip(selected[offset:offset+16],outputs):
                                generated=result.outputs[0]
                                row=dict(sample_id=source['sample_id'],source=source['source'],model=model,max_response_tokens=limit,prompt_ids=result.prompt_token_ids,output_ids=list(generated.token_ids),text=generated.text,finish_reason=generated.finish_reason,stop_reason=generated.stop_reason,adapter_sha256=sft['adapter_sha256'] if model=='sft' else None,**measure(generated.text,generated.token_ids,generated.finish_reason,tokenizer))
                                records.append(row);f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n');f.flush()
                            print(json.dumps(dict(event='generation_progress',condition=key,completed=len(records),planned=len(prompts))),flush=True)
                    elapsed=time.monotonic()-begin;timings[key]=dict(seconds=elapsed,generated_tokens=sum(r['total_tokens'] for r in records),output_tokens_per_second=sum(r['total_tokens'] for r in records)/elapsed)
                    results[key]=aggregate(records)
                if limit==protocol['primary_response_cap']:
                    primary=results[f'sft_{limit}']['all']
                    if primary['answer_closed_fraction']<.95 or primary['truncated_fraction']>.05:limits.append(protocol['secondary_response_cap'])
            summary=dict(run_id=run_id,status='PASS',prompt_count=len(prompts),limits=limits,results=results,timings=timings,cold_load_seconds=cold,wall_seconds=time.monotonic()-start,identity=dict(adapter_sha256=sft['adapter_sha256'],base_sft_prompt_logprob_delta=delta('base','sft'),repeat_sft_prompt_logprob_error=delta('sft','sft_repeat')),**memory.result())
        write_json(git/'summary.json',summary)
        write_json(git/'artifacts_manifest.json',[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file() and p.name!='status.json'])
        write_json(out/'status.json',dict(status='PASS',ended=datetime.now(timezone.utc).isoformat()));print(json.dumps(summary),flush=True)
    except BaseException:
        write_json(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

if __name__=='__main__':main()
