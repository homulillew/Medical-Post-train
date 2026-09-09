"""Durable fixed-policy G=4 inference. No optimizer or refill controller."""
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
from medical_posttrain.evidence import now, sha256, write_json
from medical_posttrain.evidence.stage2 import read, jsonlines, record, dump_lines
from medical_posttrain.data.exam import messages
from medical_posttrain.reward.parser import reasoning_text, parse
from medical_posttrain.reward.hybrid import reward


def validate_frozen(config):
    for path,digest in config['execution_hashes'].items():
        assert sha256(path)==digest, f'Frozen pipeline source changed: {path}'
    for key in ('pool','selection','reward_manifest','initialization'):
        assert sha256(config[key]['path'])==config[key]['sha256'],key
    initial=read(config['initialization']['path'])
    assert initial['base_revision']=='b968826d9c46dd6066d109eabc6255188de91218'
    assert initial['adapter_sha256']=='1601e97891e51940bd4b575d8811a77d8278cbeb296c044b7004e41da6d9ea64'
    assert sha256(Path(initial['adapter_path'])/'adapter_model.safetensors')==initial['adapter_sha256']
    return initial


def request_seed(seed, prompt_id):
    return int.from_bytes(hashlib.sha256(f'{seed}:{prompt_id}'.encode()).digest()[:4],'big')


def atomic_group(path, row):
    assert not path.exists(), 'Never overwrite a completed group'
    tmp=path.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(row,f,ensure_ascii=False,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
    fd=os.open(path.parent,os.O_DIRECTORY);os.fsync(fd);os.close(fd)


def rollout(run):
    config=run.config;initial=validate_frozen(config)
    os.environ.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_FLASHINFER_SAMPLER='0',
                      VLLM_USE_V2_MODEL_RUNNER='0', VLLM_BATCH_INVARIANT='1')
    os.environ['PATH']=str(Path(__import__('sys').executable).parent)+os.pathsep+os.environ.get('PATH','')
    write_json(run.attempt/'runtime_environment.json',{k:os.environ[k] for k in ('VLLM_WORKER_MULTIPROC_METHOD','VLLM_USE_FLASHINFER_SAMPLER','VLLM_USE_V2_MODEL_RUNNER','VLLM_BATCH_INVARIANT')})
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
    from medical_posttrain.verification.vllm_probe import serialize
    from medical_posttrain.training.lora import MemoryMonitor
    shrink=get_lora_op_configs('shrink',1,23,4096,32,3)
    assert shrink['split_k']==1
    write_json(run.attempt/'lora_shrink_config.json',shrink)
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    pool={r['prompt_id']:r for r in jsonlines(config['pool']['path'])}
    ids=read(config['selection']['path']);assert len(ids)==config['planned_prompts']==len(set(ids))
    selected=[pool[sid] for sid in ids]
    prompts=[tokenizer.apply_chat_template(messages(r),tokenize=False,add_generation_prompt=True,enable_thinking=True) for r in selected]
    prompt_ids=[tokenizer.encode(p,add_special_tokens=False) for p in prompts]
    assert all(len(p)+config['sampling']['max_tokens']<=config['engine']['max_model_len'] for p in prompt_ids), 'Prompt too long; no silent truncation'
    prompt_path=run.out/'prompts.jsonl'
    expected=[dict(prompt_id=r['prompt_id'],messages=messages(r),prompt_ids=p) for r,p in zip(selected,prompt_ids)]
    if prompt_path.exists():assert jsonlines(prompt_path)==expected
    else:dump_lines(prompt_path,expected)
    groups=run.out/'raw_groups';groups.mkdir(exist_ok=True)
    completed={}
    for path in groups.glob('*.json'):
        group=read(path);assert group['prompt_id'] in ids and group['policy_version']==config['policy_version']
        assert len(group['responses'])==4 and len({r['trajectory_id'] for r in group['responses']})==4
        assert group['prompt_id'] not in completed
        completed[group['prompt_id']]=path
    request=LoRARequest('medical_sft_'+initial['adapter_sha256'][:12],1,initial['adapter_path'])
    args=dict(model=config['model'],tokenizer=config['model'],**config['engine'])
    write_json(run.attempt/'vllm_args.json',args)
    attempt_start=time.monotonic(); generation_seconds=0; successful_tokens=0
    with MemoryMonitor() as memory:
        llm=LLM(**args);cold=time.monotonic()-attempt_start
        control=SamplingParams(temperature=0,max_tokens=8,prompt_logprobs=5)
        controls={};control_cost=[]
        for name,req in [('base',None),('sft',request),('base_negative',None),('sft_repeat',request)]:
            start=time.monotonic();result=llm.generate(prompts[:1],control,lora_request=req,use_tqdm=False)[0]
            controls[name]=serialize(result)
            control_cost.append(dict(condition=name,prompt_tokens=len(result.prompt_token_ids),output_tokens=len(result.outputs[0].token_ids),seconds=time.monotonic()-start))
            write_json(run.attempt/f'identity_{name}.json',controls[name])
        def delta(a,b):
            return max(abs(x[k]-y[k]) for x,y in zip(controls[a]['prompt_logprobs'][1:],controls[b]['prompt_logprobs'][1:]) for k in x.keys()&y.keys())
        assert controls['base']['token_ids']==controls['base_negative']['token_ids']
        assert delta('base','base_negative')<=1e-4 and delta('base','sft')>1e-5
        assert controls['sft']['token_ids']==controls['sft_repeat']['token_ids'] and delta('sft','sft_repeat')<=1e-4
        before_sleep=time.monotonic();llm.sleep(level=1);sleep_seconds=time.monotonic()-before_sleep
        sleeping_memory=memory.nv.nvmlDeviceGetMemoryInfo(memory.handle).used
        before_wake=time.monotonic();llm.wake_up();wake_seconds=time.monotonic()-before_wake
        start=time.monotonic();after=llm.generate(prompts[:1],control,lora_request=request,use_tqdm=False)[0]
        controls['after_wake']=serialize(after)
        control_cost.append(dict(condition='after_wake',prompt_tokens=len(after.prompt_token_ids),output_tokens=len(after.outputs[0].token_ids),seconds=time.monotonic()-start))
        write_json(run.attempt/'identity_after_wake.json',controls['after_wake'])
        assert controls['sft']['token_ids']==controls['after_wake']['token_ids'] and delta('sft','after_wake')<=1e-4
        identity=dict(adapter_sha256=initial['adapter_sha256'],base_sft_delta=delta('base','sft'),base_negative_error=delta('base','base_negative'),
                      repeat_error=delta('sft','sft_repeat'),wake_error=delta('sft','after_wake'),sleep_seconds=sleep_seconds,
                      wake_seconds=wake_seconds,sleeping_memory_bytes=sleeping_memory,controls=control_cost,optimizer_updates=0)
        write_json(run.attempt/'identity_receipt.json',identity)
        pending=[i for i,r in enumerate(selected) if r['prompt_id'] not in completed]
        for offset in range(0,len(pending),config['request_batch_prompts']):
            batch=pending[offset:offset+config['request_batch_prompts']]
            validate_frozen(config)
            params=[SamplingParams(**config['sampling'],seed=request_seed(config['seed'],ids[i])) for i in batch]
            request_id=f'{run.attempt.name}:batch:{offset:04d}'
            run.metric(event='request_started',request_id=request_id,prompt_ids=[ids[i] for i in batch],
                       prompt_tokens=sum(len(prompt_ids[i]) for i in batch),planned_responses=len(batch)*4,
                       seeds=[request_seed(config['seed'],ids[i]) for i in batch])
            start=time.monotonic()
            outputs=llm.generate([prompts[i] for i in batch],params,lora_request=request,use_tqdm=False)
            elapsed=time.monotonic()-start;generation_seconds+=elapsed
            assert len(outputs)==len(batch)
            batch_tokens=0
            for i,result in zip(batch,outputs):
                source=selected[i];sid=source['prompt_id'];gid=run.run_id+':'+sid
                assert result.prompt_token_ids==prompt_ids[i]
                assert len(result.outputs)==4 and {o.index for o in result.outputs}==set(range(4))
                records=[]
                for output in sorted(result.outputs,key=lambda o:o.index):
                    assert output.finish_reason in ('stop','length') and len(output.token_ids)>0
                    row=dict(run_id=run.run_id,policy_version=config['policy_version'],config_sha256=run.manifest['config_sha256'],
                             reward_version=config['reward_manifest']['sha256'],prompt_id=sid,group_id=gid,
                             trajectory_id=gid+':'+str(output.index),member_index=output.index,request_id=request_id,
                             question=source['question'],options=source['options'],ground_truth=source['answer_set'],
                             raw_output=output.text,token_ids=list(output.token_ids),prompt_tokens=len(result.prompt_token_ids),
                             output_tokens=len(output.token_ids),finish_reason=output.finish_reason,stop_reason=output.stop_reason,
                             generation_time=elapsed,generation_time_scope='shared batch wall time, not individual request latency',
                             request_seed=request_seed(config['seed'],sid),adapter_sha256=initial['adapter_sha256'])
                    records.append(row);batch_tokens+=len(output.token_ids)
                atomic_group(groups/f'{i:04d}.json',dict(prompt_id=sid,group_id=gid,policy_version=config['policy_version'],responses=records))
                completed[sid]=groups/f'{i:04d}.json'
            successful_tokens+=batch_tokens
            run.metric(event='request_completed',request_id=request_id,completed_prompts=len(completed),completed_responses=len(completed)*4,
                       batch_output_tokens=batch_tokens,seconds=elapsed,output_tokens_per_second=batch_tokens/elapsed)
            write_json(run.out/'progress.json',dict(unique_prompts=len(completed),completed_responses=len(completed)*4,planned_prompts=len(ids),optimizer_updates=0,timestamp=now()))
        validate_frozen(config)
        assert set(completed)==set(ids)
        summary=dict(status='GENERATED',planned_prompts=len(ids),unique_prompts=len(completed),completed_responses=4*len(completed),
                     optimizer_updates=0,cold_load_seconds=cold,generation_seconds=generation_seconds,
                     generated_output_tokens_this_attempt=successful_tokens,
                     wall_seconds=time.monotonic()-attempt_start,identity=identity,**memory.result())
        write_json(run.attempt/'generation_summary.json',summary)
        write_json(run.out/'generation_summary.json',summary)
    write_json(run.attempt/'status.json',dict(status='PASS',ended=now(),wall_seconds=time.monotonic()-run.start))
    write_json(run.out/'status.json',dict(status='GENERATED',ended=now(),next='Frozen semantic scoring and raw analysis required'))


def score_run(run):
    import numpy as np
    from transformers import AutoTokenizer
    from medical_posttrain.reward.semantic import Encoder, CHUNK_POLICY
    from medical_posttrain.training.lora import MemoryMonitor
    from medical_posttrain.rollout.statistics import summarize, group_summary
    config=run.config;validate_frozen(config)
    frozen=read(config['reward_manifest']['path'])
    assert frozen['chunk_policy']==CHUNK_POLICY
    pool={r['prompt_id']:r for r in jsonlines(config['pool']['path'])}
    ids=read(config['selection']['path'])
    groups=[read(p) for p in sorted((run.out/'raw_groups').glob('*.json'))]
    assert [g['prompt_id'] for g in groups]==ids
    rows=[r for g in groups for r in g['responses']]
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    # Retain raw wrong-answer semantic values as well as correct ones.
    texts=list(dict.fromkeys([reasoning_text(r['raw_output']) for r in rows]+[pool[sid]['reference_explanation'] for sid in ids]))
    index={t:i for i,t in enumerate(texts)}
    start=time.monotonic()
    with MemoryMonitor() as memory:
        encoder=Encoder(frozen['encoder_id'],frozen['encoder_revision'],device='cuda')
        vectors,metadata=encoder.encode(texts,batch_size=16)
        encoding_seconds=time.monotonic()-start
        np.save(run.out/'semantic_vectors.npy',vectors)
        write_json(run.out/'semantic_encoding.json',dict(texts=texts,metadata=metadata,encoder_id=encoder.model_id,revision=encoder.revision,chunk_policy=CHUNK_POLICY))
        scored=[]
        for row in rows:
            text=row['raw_output'];reasoning=reasoning_text(text);reference=pool[row['prompt_id']]['reference_explanation']
            cosine=float(vectors[index[reasoning]]@vectors[index[reference]])
            semantic=float(np.clip(cosine,0,1)) if reasoning.strip() and reference.strip() else 0.
            rr=reward(text,row['ground_truth'],semantic,''.join(row['options']),row['finish_reason'])
            parsed=rr['parser'];span=parsed['matched_span'];answer=text[span[0]:span[1]].strip() if span else ''
            scored.append(dict(**row,parsed_answer=parsed['answer_set'],parse_method='strict' if parsed['strict_match'] else 'fallback' if parsed['fallback_match'] else 'none',
                               format_valid=parsed['valid_format'],ambiguous=parsed['ambiguous'],parse_error=parsed['error_type'],parsed=parsed,
                               acc=rr['acc'],semantic=semantic,semantic_cosine=cosine,semantic_contribution=rr['semantic_contribution'],format=rr['format'],total_reward=rr['score'],
                               semantic_reason='empty_reasoning' if not reasoning.strip() else 'missing_reference' if not reference.strip() else 'encoded',
                               reasoning_tokens=len(tokenizer.encode(reasoning,add_special_tokens=False)),answer_tokens=len(tokenizer.encode(answer,add_special_tokens=False)),
                               total_tokens=row['output_tokens'],answer_closed='<answer>' in text and '</answer>' in text,
                               think_closed='<think>' in text and '</think>' in text,
                               reasoning_embedding_index=index[reasoning],reference_embedding_index=index[reference]))
        dump_lines(run.out/'trajectories.jsonl',scored)
        derived=[group_summary(scored[i:i+4]) for i in range(0,len(scored),4)]
        dump_lines(run.out/'groups.jsonl',derived)
        summary=summarize(scored,derived,run.out)
        summary.update(status='PASS',optimizer_updates=0,semantic_encoding_seconds=encoding_seconds,
                       semantic_memory=memory.result(),reward_version=config['reward_manifest']['sha256'])
    validate_frozen(config)
    write_json(run.git/'groups.json',derived)
    run.finish(summary)
