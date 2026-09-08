import gc
import json
import os
from pathlib import Path
import time
from medical_posttrain.evidence import sha256,write_json
from medical_posttrain.training.lora import MemoryMonitor

def run(ev):
    # Forking a process with initialized CUDA is not supported by vLLM.
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER']='0'
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='1' if ev.config['vllm_runner']=='v2' else '0'
    os.environ['VLLM_BATCH_INVARIANT']='1' if ev.config['batch_invariant'] else '0'
    print('Explicit sampler selection: VLLM_USE_FLASHINFER_SAMPLER=0; native vLLM sampler, no CUDA 12 FlashInfer JIT',flush=True)
    from vllm import LLM,SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(ev.config['model'],local_files_only=True)
    args=dict(model=ev.config['model'],tokenizer=ev.config['model'],dtype='bfloat16',tensor_parallel_size=1,
              max_model_len=4096,max_num_seqs=16,gpu_memory_utilization=.65,enable_lora=True,max_lora_rank=32,
              enable_sleep_mode=True,enforce_eager=True,seed=ev.config['seed'],enable_prefix_caching=False)
    write_json(ev.path/'vllm_args.json',args)
    write_json(ev.path/'execution_environment.json',{k:os.environ[k] for k in ['VLLM_WORKER_MULTIPROC_METHOD','VLLM_USE_FLASHINFER_SAMPLER','VLLM_USE_V2_MODEL_RUNNER','VLLM_BATCH_INVARIANT']})
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
    shrink=get_lora_op_configs('shrink',1,23,4096,32,3)
    write_json(ev.path/'lora_shrink_config.json',shrink)
    if ev.config['batch_invariant']: assert shrink['split_k']==1
    t=time.monotonic()
    with MemoryMonitor() as mm:
        llm=LLM(**args)
        cold=time.monotonic()-t
        if ev.config['purpose']=='length':
            return length(ev,llm,tok,SamplingParams,mm,cold)
        if not ev.config['adapter']: raise ValueError('vLLM identity requires explicit trained adapter')
        adapter=Path(ev.config['adapter'])
        ah=sha256(adapter/'adapter_model.safetensors')
        request=LoRARequest('stage0_adapter_'+ah[:12],1,str(adapter))
        prompt=tok.apply_chat_template([{'role':'user','content':'请选择 C，并用 <answer>C</answer> 表示最终答案。'}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
        assert prompt.endswith('<|im_start|>assistant\n')
        params=SamplingParams(temperature=0,max_tokens=32,logprobs=5,prompt_logprobs=5)
        outputs={}; elapsed={}
        for name,req in [('base',None),('adapter',request),('base_negative',None)]:
            t=time.monotonic(); result=llm.generate([prompt],params,lora_request=req,use_tqdm=False)[0]; elapsed[name]=time.monotonic()-t
            outputs[name]=serialize(result)
        assert outputs['base']['token_ids']==outputs['base_negative']['token_ids']
        # Same prompt token logprobs provide a matched-token positive control even if generation text agrees.
        deltas=[]
        for a,b in zip(outputs['base']['prompt_logprobs'][1:],outputs['adapter']['prompt_logprobs'][1:]):
            for key in a.keys()&b.keys(): deltas.append(abs(a[key]-b[key]))
        assert deltas and max(deltas)>1e-5,'Adapter produced no measurable logprob change'
        write_json(ev.path/'identity_outputs.json',outputs)
        ev.metric(event='vllm_identity',cold_load_seconds=cold,adapter_sha256=ah,max_matched_prompt_logprob_delta=max(deltas),seconds=elapsed,output_tokens_per_second={k:len(v['token_ids'])/elapsed[k] for k,v in outputs.items()},**mm.result())
        warm=[]
        for index in (1,2):
            warm.append(serialize(llm.generate([prompt],params,lora_request=request,use_tqdm=False)[0]))
            write_json(ev.path/f'adapter_warm_{index}.json',warm[-1])
        warm_errors=[abs(a[k]-b[k]) for a,b in zip(warm[0]['prompt_logprobs'][1:],warm[1]['prompt_logprobs'][1:]) for k in a.keys()&b.keys()]
        cold_errors=[abs(a[k]-b[k]) for a,b in zip(outputs['adapter']['prompt_logprobs'][1:],warm[1]['prompt_logprobs'][1:]) for k in a.keys()&b.keys()]
        ev.metric(event='cold_warm_control',warm_tokens_equal=warm[0]['token_ids']==warm[1]['token_ids'],cold_warm_tokens_equal=outputs['adapter']['token_ids']==warm[1]['token_ids'],warm_max_prompt_logprob_error=max(warm_errors),cold_warm_max_prompt_logprob_error=max(cold_errors))
        if max(cold_errors)>1e-4:
            ev.case('cold_warm_adapter_mismatch','Adapter logprobs changed before any sleep; cold/warm control isolates the boundary',cold=outputs['adapter'],warm=warm[1],max_error=max(cold_errors))
        assert warm[0]['token_ids']==warm[1]['token_ids'] and max(warm_errors)<=1e-4,'Adapter is unstable even before sleeping'
        t=time.monotonic(); llm.sleep(level=1); sleep=time.monotonic()-t
        sleep_mem=mm.nv.nvmlDeviceGetMemoryInfo(mm.handle).used
        # Sleeping rollout allows an actual BF16 LoRA actor to use the same GPU.
        import subprocess,sys
        switch_config=dict(purpose='lora',model=ev.config['model'],adapter=str(adapter))
        # A dedicated subprocess runs one backward; no full SFT launcher or formal stage is entered.
        argv=[sys.executable,'-m','medical_posttrain.verification.switch_actor',str(ev.path)]
        t=time.monotonic()
        p=subprocess.run(argv,stdout=open(ev.path/'switch_actor.stdout.log','w'),stderr=open(ev.path/'switch_actor.stderr.log','w'),timeout=600)
        assert p.returncode==0,'Actor switch subprocess failed; see switch_actor.stderr.log'
        switch=time.monotonic()-t
        t=time.monotonic(); llm.wake_up(); wake=time.monotonic()-t
        after=serialize(llm.generate([prompt],params,lora_request=request,use_tqdm=False)[0])
        write_json(ev.path/'after_wake.json',after)
        parity=[abs(a[k]-b[k]) for a,b in zip(after['prompt_logprobs'][1:],warm[1]['prompt_logprobs'][1:]) for k in a.keys()&b.keys()]
        ev.metric(event='wake_identity',token_ids_equal=after['token_ids']==warm[1]['token_ids'],max_matched_prompt_logprob_error=max(parity),sleep_seconds=sleep,wake_seconds=wake,sleep_memory_bytes=sleep_mem,actor_process_seconds=switch)
        if after['token_ids']!=warm[1]['token_ids']:
            ev.case('sleep_wake_output_mismatch','Greedy tokens changed across sleep/wake',before=warm[1],after=after,max_matched_prompt_logprob_error=max(parity))
        assert after['token_ids']==warm[1]['token_ids'] and max(parity)<=1e-4,'sleep/wake changed adapter identity'
        updated=Path(ev.config['artifact_root'])/ev.run_id/'actor_child'/'updated_adapter'
        updated_hash=sha256(updated/'adapter_model.safetensors')
        assert updated_hash!=ah
        updated_request=LoRARequest('stage0_updated_'+updated_hash[:12],2,str(updated))
        changed=serialize(llm.generate([prompt],params,lora_request=updated_request,use_tqdm=False)[0])
        sync_deltas=[abs(a[k]-b[k]) for a,b in zip(after['prompt_logprobs'][1:],changed['prompt_logprobs'][1:]) for k in a.keys()&b.keys()]
        assert max(sync_deltas)>1e-5,'Updated actor adapter was not observable in rollout'
        write_json(ev.path/'updated_adapter_outputs.json',changed)
        for p in updated.glob('*'):
            if p.is_file():ev.artifact(p,'updated_actor_adapter')
        ev.metric(event='adapter_sync',old_adapter_sha256=ah,new_adapter_sha256=updated_hash,max_matched_prompt_logprob_delta=max(sync_deltas))
        ev.metric(event='sleep_wake_switch',sleep_seconds=sleep,wake_seconds=wake,sleep_memory_bytes=sleep_mem,actor_process_seconds=switch,**mm.result())
        # Repeat boundaries without additional optimizer updates, isolating sleep state.
        for cycle in (2,3):
            before_cycle=serialize(llm.generate([prompt],params,lora_request=request,use_tqdm=False)[0])
            t=time.monotonic();llm.sleep(level=1);cycle_sleep=time.monotonic()-t
            t=time.monotonic();llm.wake_up();cycle_wake=time.monotonic()-t
            after_cycle=serialize(llm.generate([prompt],params,lora_request=request,use_tqdm=False)[0])
            errors=[abs(a[k]-b[k]) for a,b in zip(before_cycle['prompt_logprobs'][1:],after_cycle['prompt_logprobs'][1:]) for k in a.keys()&b.keys()]
            write_json(ev.path/f'cycle_{cycle}.json',dict(before=before_cycle,after=after_cycle))
            equal=before_cycle['token_ids']==after_cycle['token_ids']
            ev.metric(event='repeat_sleep_wake',cycle=cycle,token_ids_equal=equal,max_matched_prompt_logprob_error=max(errors),sleep_seconds=cycle_sleep,wake_seconds=cycle_wake)
            if not equal or max(errors)>1e-4:ev.case('repeat_wake_mismatch','Repeated identity check failed',cycle=cycle,before=before_cycle,after=after_cycle)
            assert equal and max(errors)<=1e-4,'Repeated sleep/wake identity failed'
        # Bounded offline batch timing, not an API/concurrency serving benchmark.
        for condition,req in [('base',None),('adapter',request)]:
            timing_params=SamplingParams(temperature=0,max_tokens=32)
            t=time.monotonic(); batch_outputs=llm.generate([prompt]*16,timing_params,lora_request=req,use_tqdm=False);seconds=time.monotonic()-t
            total=sum(len(r.outputs[0].token_ids) for r in batch_outputs)
            write_json(ev.bulk/f'batch16_{condition}.json',[serialize(r) for r in batch_outputs]);ev.artifact(ev.bulk/f'batch16_{condition}.json','bounded_offline_batch_timing')
            ev.metric(event='offline_batch_timing',condition=condition,sequences=16,max_output_tokens=32,generated_tokens=total,seconds=seconds,aggregate_output_tokens_per_second=total/seconds)
    return dict(gates={'vllm_load':True,'vllm_lora_identity':True,'actor_rollout_switch':True},cold_load_seconds=cold,adapter_sha256=ah,sleep_seconds=sleep,wake_seconds=wake,max_prompt_logprob_delta=max(deltas),runner=ev.config['vllm_runner'],batch_invariant=ev.config['batch_invariant'],verified_sleep_cycles=3)

def serialize(r):
    o=r.outputs[0]
    def logs(rows):
        return [None if row is None else {str(k):v.logprob for k,v in row.items()} for row in rows] if rows else None
    return dict(text=o.text,token_ids=list(o.token_ids),finish_reason=o.finish_reason,logprobs=logs(o.logprobs),prompt_logprobs=logs(r.prompt_logprobs))

def length(ev,llm,tok,SamplingParams,mm,cold):
    from medical_posttrain.data.format import parse_answer
    manifest=json.loads(Path('experiments/stage0/bootstrap/cmexam-train32-manifest.json').read_text())
    source=Path(manifest['path'])
    assert sha256(source)==manifest['sha256']
    data=json.loads(source.read_text())
    prompts=[tok.apply_chat_template([{'role':'user','content':row['Question']+'\n'+row['Options']+'\n请推理，并将最终选项字母写在 <answer>...</answer> 内。'}],tokenize=False,add_generation_prompt=True,enable_thinking=True) for row in data['rows']]
    assert len(prompts)==32
    result=[]
    for limit in (512,1024):
        params=SamplingParams(n=4,temperature=.6,top_p=1.,top_k=-1,max_tokens=limit,seed=ev.config['seed'])
        t=time.monotonic(); outputs=llm.generate(prompts,params,use_tqdm=False); seconds=time.monotonic()-t
        rows=[]
        for i,r in enumerate(outputs):
            for j,o in enumerate(r.outputs):
                parsed=parse_answer(o.text)
                row=dict(prompt_id=f'cmexam_train_row_{i}',group_member=j,response=o.text,token_ids=list(o.token_ids),answer=data['rows'][i]['Answer'],parsed=parsed,closed='<answer>' in o.text and '</answer>' in o.text,format_valid=parsed is not None,correct=parsed==data['rows'][i]['Answer'],truncated=o.finish_reason=='length',finish_reason=o.finish_reason)
                rows.append(row)
        assert len(rows)==128
        write_json(ev.bulk/f'outputs_{limit}.json',rows); ev.artifact(ev.bulk/f'outputs_{limit}.json','train_only_length_responses')
        metrics=dict(limit=limit,prompts=32,responses=128,closure_rate=sum(x['closed'] for x in rows)/128,truncation_rate=sum(x['truncated'] for x in rows)/128,format_validity=sum(x['format_valid'] for x in rows)/128,accuracy=sum(x['correct'] for x in rows)/128,generated_tokens=sum(len(x['token_ids']) for x in rows),average_output_tokens=sum(len(x['token_ids']) for x in rows)/128,seconds=seconds)
        metrics['output_tokens_per_second']=metrics['generated_tokens']/seconds
        result.append(ev.metric(event='output_length',**metrics,**mm.result()))
        for row in sorted(rows,key=lambda x:len(x['token_ids']),reverse=True)[:2]:
            ev.case(f'length_{limit}',f"limit={limit}; truncation={row['truncated']}; parsed={row['parsed']}",category='BAD_CASE' if row['truncated'] else 'BOUNDARY_CASE',response=row)
    return dict(gates={'response_length':True},conditions=result,cold_load_seconds=cold,proposal_1024=result[0]['closure_rate']<.95 or result[0]['truncation_rate']>.05)
