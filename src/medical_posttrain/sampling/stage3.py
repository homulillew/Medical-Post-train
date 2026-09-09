"""Stage 3 inference-only transactional refill worker; no trainer or optimizer."""
import os
from pathlib import Path
import signal
import time
from medical_posttrain.evidence import now, sha256, write_json
from medical_posttrain.evidence.stage2 import read, jsonlines, record, dump_lines
from medical_posttrain.rollout.profile import validate_frozen, atomic_group
from medical_posttrain.data.exam import messages
from medical_posttrain.reward.parser import reasoning_text
from medical_posttrain.reward.hybrid import reward
from .dynamic import Stream, initial_state, apply_batch, stop_reason


def frozen(config):
    initial=validate_frozen(config)
    for path,digest in config['controller_hashes'].items():
        assert sha256(path)==digest, f'Frozen controller changed: {path}'
    return initial


def restore(out, config, pool_size):
    state=initial_state(config)
    for batch in sorted((out/'batches').glob('*')):
        if not (batch/'commit.json').exists():
            break
        commit=read(batch/'commit.json')
        assert commit['state_before']==state
        for artifact in commit['artifacts']:
            assert sha256(artifact['path'])==artifact['sha256']
        state,decisions=apply_batch(state,read(batch/'scored.json')['groups'],config,pool_size)
        assert state==commit['state_after'] and decisions==commit['decisions']
    checkpoint=out/'checkpoint.json'
    if checkpoint.exists():
        cached=read(checkpoint)
        # A crash between durable commit and replaceable checkpoint is recoverable.
        assert cached==state or (cached['batches']+1==state['batches'] and read(out/'batches'/f"{cached['batches']:04d}"/'commit.json')['state_before']==cached)
    write_json(checkpoint,state)
    return state


def score_groups(groups, encoder, tokenizer, pool, batch):
    import numpy as np
    from medical_posttrain.reward.semantic import CHUNK_POLICY
    rows=[r for g in groups for r in g['responses']]
    texts=list(dict.fromkeys([reasoning_text(r['raw_output']) for r in rows]+[pool[g['prompt_id']]['reference_explanation'] for g in groups]))
    index={t:i for i,t in enumerate(texts)}
    vectors,metadata=encoder.encode(texts,batch_size=16)
    np.save(batch/'semantic_vectors.npy',vectors)
    write_json(batch/'semantic_encoding.json',dict(texts=texts,metadata=metadata,encoder_id=encoder.model_id,revision=encoder.revision,chunk_policy=CHUNK_POLICY))
    result=[]
    for group in groups:
        scored=[]
        for row in group['responses']:
            text=row['raw_output'];reasoning=reasoning_text(text);reference=pool[row['prompt_id']]['reference_explanation']
            cosine=float(vectors[index[reasoning]]@vectors[index[reference]])
            sem=float(np.clip(cosine,0,1)) if reasoning.strip() and reference.strip() else 0.
            rr=reward(text,row['ground_truth'],sem,''.join(row['options']),row['finish_reason'])
            parsed=rr['parser'];span=parsed['matched_span'];answer=text[span[0]:span[1]].strip() if span else ''
            scored.append(dict(**row,parsed_answer=parsed['answer_set'],parse_method='strict' if parsed['strict_match'] else 'fallback' if parsed['fallback_match'] else 'none',
                format_valid=parsed['valid_format'],ambiguous=parsed['ambiguous'],parse_error=parsed['error_type'],parsed=parsed,
                acc=rr['acc'],semantic=sem,semantic_cosine=cosine,semantic_contribution=rr['semantic_contribution'],format=rr['format'],total_reward=rr['score'],
                semantic_reason='empty_reasoning' if not reasoning.strip() else 'missing_reference' if not reference.strip() else 'encoded',
                reasoning_tokens=len(tokenizer.encode(reasoning,add_special_tokens=False)),answer_tokens=len(tokenizer.encode(answer,add_special_tokens=False)),
                total_tokens=row['output_tokens'],answer_closed='<answer>' in text and '</answer>' in text,think_closed='<think>' in text and '</think>' in text,
                reasoning_embedding_index=index[reasoning],reference_embedding_index=index[reference]))
        result.append(dict(group, responses=scored))
    atomic_group(batch/'scored.json',dict(groups=result))
    return result


def worker(run):
    config=run.config;initial=frozen(config)
    os.environ.update(VLLM_WORKER_MULTIPROC_METHOD='spawn',VLLM_USE_FLASHINFER_SAMPLER='0',VLLM_USE_V2_MODEL_RUNNER='0',VLLM_BATCH_INVARIANT='1')
    os.environ['PATH']=str(Path(__import__('sys').executable).parent)+os.pathsep+os.environ.get('PATH','')
    write_json(run.attempt/'runtime_environment.json',{k:os.environ[k] for k in ('VLLM_WORKER_MULTIPROC_METHOD','VLLM_USE_FLASHINFER_SAMPLER','VLLM_USE_V2_MODEL_RUNNER','VLLM_BATCH_INVARIANT')})
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
    from medical_posttrain.verification.vllm_probe import serialize
    from medical_posttrain.training.lora import MemoryMonitor
    from medical_posttrain.reward.semantic import Encoder
    shrink=get_lora_op_configs('shrink',1,23,4096,32,3);assert shrink['split_k']==1
    write_json(run.attempt/'lora_shrink_config.json',shrink)
    pool={r['prompt_id']:r for r in jsonlines(config['pool']['path'])}
    stream=Stream(list(pool),config['seed'],config['stream_domain'])
    (run.out/'batches').mkdir(exist_ok=True)
    state=restore(run.out,config,len(pool))
    next_encounters=[stream.encounter(i,run.run_id,config['policy_version']) for i in range(state['encounter_index'],state['encounter_index']+config['request_batch_prompts'])]
    write_json(run.attempt/'resume_start.json',dict(pid=os.getpid(),state=state,next_encounters=next_encounters,checkpoint=record(run.out/'checkpoint.json'),timestamp=now()))
    if state['batches']:
        assert (run.out/'termination.json').exists(), 'Resumption requires a retained interruption record'
        assert read(run.out/'pause_ready.json')['state']==state
        assert read(run.out/'pause_ready.json')['next_encounters']==next_encounters
    assert stop_reason(state,config) is None, 'Completed runs are immutable'
    tokenizer=AutoTokenizer.from_pretrained(config['model'],local_files_only=True)
    def prompt(encounter):
        source=pool[encounter['prompt_id']]
        text=tokenizer.apply_chat_template(messages(source),tokenize=False,add_generation_prompt=True,enable_thinking=True)
        ids=tokenizer.encode(text,add_special_tokens=False)
        assert len(ids)+config['sampling']['max_tokens']<=config['engine']['max_model_len']
        return text,ids
    control_prompt=prompt(stream.encounter(0,run.run_id,config['policy_version']))[0]
    request=LoRARequest('medical_sft_'+initial['adapter_sha256'][:12],1,initial['adapter_path'])
    args=dict(model=config['model'],tokenizer=config['model'],**config['engine']);write_json(run.attempt/'vllm_args.json',args)
    attempt_start=time.monotonic()
    with MemoryMonitor() as memory:
        llm=LLM(**args);cold=time.monotonic()-attempt_start
        control=SamplingParams(temperature=0,max_tokens=8,prompt_logprobs=5)
        controls={};control_cost=[]
        for name,req in [('base',None),('sft',request),('base_negative',None),('sft_repeat',request)]:
            start=time.monotonic();result=llm.generate([control_prompt],control,lora_request=req,use_tqdm=False)[0]
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
        start=time.monotonic();after=llm.generate([control_prompt],control,lora_request=request,use_tqdm=False)[0]
        controls['after_wake']=serialize(after)
        control_cost.append(dict(condition='after_wake',prompt_tokens=len(after.prompt_token_ids),output_tokens=len(after.outputs[0].token_ids),seconds=time.monotonic()-start))
        write_json(run.attempt/'identity_after_wake.json',controls['after_wake'])
        assert controls['sft']['token_ids']==controls['after_wake']['token_ids'] and delta('sft','after_wake')<=1e-4
        identity=dict(adapter_sha256=initial['adapter_sha256'],base_sft_delta=delta('base','sft'),base_negative_error=delta('base','base_negative'),
            repeat_error=delta('sft','sft_repeat'),wake_error=delta('sft','after_wake'),sleep_seconds=sleep_seconds,wake_seconds=wake_seconds,
            sleeping_memory_bytes=sleeping_memory,controls=control_cost,optimizer_updates=0)
        write_json(run.attempt/'identity_receipt.json',identity)
        reward_config=read(config['reward_manifest']['path']);start=time.monotonic()
        encoder=Encoder(reward_config['encoder_id'],reward_config['encoder_revision'],device='cuda')
        encoder_load=time.monotonic()-start
        while stop_reason(state,config) is None:
            frozen(config)
            batch=run.out/'batches'/f"{state['batches']:04d}";batch.mkdir(exist_ok=True)
            size=config['request_batch_prompts']
            if config['mode']=='smoke':size=min(size,config['smoke_prompts']-state['generated_groups'])
            encounters=[stream.encounter(i,run.run_id,config['policy_version']) for i in range(state['encounter_index'],state['encounter_index']+size)]
            prompts=[prompt(e) for e in encounters]
            reservation=dict(state_before=state,encounters=encounters,prompt_ids=[p[1] for p in prompts],attempt=run.attempt.name)
            if (batch/'reservation.json').exists():
                previous=read(batch/'reservation.json');assert previous['state_before']==state and previous['encounters']==encounters
                assert (batch/'raw.json').exists(), 'Interrupted generation has UNKNOWN output cost: retain FAILED attempt, no silent retry'
            else:
                atomic_group(batch/'reservation.json',reservation)
            if (batch/'raw.json').exists():
                raw=read(batch/'raw.json');groups=raw['groups']
            else:
                run.metric(event='generation_started',batch=state['batches'],encounters=encounters)
                params=[SamplingParams(**config['sampling'],seed=e['request_seed']) for e in encounters]
                start=time.monotonic()
                outputs=llm.generate([p[0] for p in prompts],params,lora_request=request,use_tqdm=False)
                seconds=time.monotonic()-start
                assert len(outputs)==len(encounters)
                groups=[]
                for e,p,result in zip(encounters,prompts,outputs):
                    assert result.prompt_token_ids==p[1]
                    source=pool[e['prompt_id']];rows=[]
                    for output in sorted(result.outputs,key=lambda o:o.index):
                        rows.append(dict(run_id=run.run_id,policy_version=config['policy_version'],config_sha256=run.manifest['config_sha256'],
                            reward_version=config['reward_manifest']['sha256'],prompt_id=e['prompt_id'],group_id=e['group_id'],
                            encounter_index=e['encounter_index'],trajectory_id=e['group_id']+':'+str(output.index),member_index=output.index,
                            question=source['question'],options=source['options'],ground_truth=source['answer_set'],raw_output=output.text,
                            token_ids=list(output.token_ids),prompt_tokens=len(p[1]),output_tokens=len(output.token_ids),finish_reason=output.finish_reason,
                            stop_reason=output.stop_reason,request_seed=e['request_seed'],adapter_sha256=initial['adapter_sha256']))
                    groups.append(dict(**e,prompt_tokens=len(p[1]),responses=rows))
                raw=dict(groups=groups,generation_seconds=seconds,attempt=run.attempt.name,timestamp=now())
                atomic_group(batch/'raw.json',raw)
                run.metric(event='generation_completed',batch=state['batches'],seconds=seconds,output_tokens=sum(r['output_tokens'] for g in groups for r in g['responses']))
            start=time.monotonic()
            scored=read(batch/'scored.json')['groups'] if (batch/'scored.json').exists() else score_groups(groups,encoder,tokenizer,pool,batch)
            after,decisions=apply_batch(state,scored,config,len(pool))
            commit=dict(state_before=state,state_after=after,decisions=decisions,scoring_seconds=time.monotonic()-start,
                        artifacts=[record(batch/name) for name in ('reservation.json','raw.json','scored.json','semantic_vectors.npy','semantic_encoding.json')])
            atomic_group(batch/'commit.json',commit)
            state=after;write_json(run.out/'checkpoint.json',state)
            run.metric(event='batch_committed',batch=state['batches'],generated=state['generated_groups'],accepted=state['accepted_mixed_groups'],costs=state['output_tokens_by_disposition'])
            write_json(run.attempt/'runtime_progress.json',dict(wall_seconds=time.monotonic()-attempt_start,cold_load_seconds=cold,encoder_load_seconds=encoder_load,**memory.result()))
            if config['mode']=='smoke' and state['batches']==1 and run.attempt.name=='attempt_001':
                next_encounters=[stream.encounter(i,run.run_id,config['policy_version']) for i in range(state['encounter_index'],state['encounter_index']+config['request_batch_prompts'])]
                atomic_group(run.out/'pause_ready.json',dict(state=state,next_encounters=next_encounters,pid=os.getpid(),timestamp=now(),checkpoint=record(run.out/'checkpoint.json')))
                run.metric(event='pause_ready_for_external_termination',accepted=state['accepted_mixed_groups'])
                signal.pause()
                raise RuntimeError('Paused process must be terminated and resumed in a new process')
        frozen(config)
        runtime=dict(wall_seconds=time.monotonic()-attempt_start,cold_load_seconds=cold,encoder_load_seconds=encoder_load,identity=identity,**memory.result())
        write_json(run.attempt/'runtime_summary.json',runtime)
    llm.llm_engine.engine_core.shutdown(timeout=30)
    from .analysis import summarize
    summary=summarize(run.out)
    summary.update(status=stop_reason(state,config),optimizer_updates=0)
    run.finish(summary)
