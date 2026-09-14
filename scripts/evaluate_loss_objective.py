#!/usr/bin/env python3
"""Five matched greedy policies on a new sealed auxiliary validation set only."""
import argparse
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now
from analyze_frontier import paired_bootstrap
from medical_posttrain.reward.parser import parse


def paired(a,b,seed=20260914):
    assert [r['prompt_id'] for r in a]==[r['prompt_id'] for r in b]
    gains=[x['prompt_id'] for x,y in zip(a,b) if not x['acc'] and y['acc']]
    losses=[x['prompt_id'] for x,y in zip(a,b) if x['acc'] and not y['acc']]
    n=len(gains)+len(losses)
    return dict(wrong_to_correct=len(gains),correct_to_wrong=len(losses),gained_ids=gains,regressed_ids=losses,
                exact_mcnemar=min(1.,2*sum(math.comb(n,k) for k in range(min(len(gains),len(losses))+1))/2**n) if n else 1.,
                bootstrap=paired_bootstrap([r['acc'] for r in a],[r['acc'] for r in b],seed=seed))


def main():
    import fcntl
    import subprocess
    gpu_lock=(ROOT.parent/'stage4-gpu.lock').open('a')
    fcntl.flock(gpu_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip())<100
    from loss_objective_runtime import check
    from medical_posttrain.rl.rollout import Rollout
    from vllm import SamplingParams
    from tokenizers.decoders import DecodeStream
    import numpy as np
    p=check();idx=ROOT/'experiments/stage4';out=Path(p['evaluation_artifact_path'])
    assert all(read(idx/f'grpo_{arm}512_verification_v1.json')['result']=='PASS' for arm in ['vanilla','dynamic'])
    manifest=read(p['evaluation_manifest']['path']);assert record(p['evaluation_manifest']['path'])==p['evaluation_manifest']
    assert record(manifest['dataset']['path'])==manifest['dataset']
    rows=read(manifest['dataset']['path']);assert len(rows)==manifest['count']
    policies=dict(p['evaluation_policies'])
    for arm in ['vanilla','dynamic']:
        adapter=Path(p['runs'][arm]['path'])/'windows/0063/checkpoint/adapter'
        policies[arm+'_grpo']=dict(adapter_path=str(adapter),adapter_sha256=record(adapter/'adapter_model.safetensors')['sha256'])
    directory=out/'loss_objective_eval';directory.mkdir()
    immutable(directory/'actual_policies.json',policies)
    llm=Rollout(directory,p['config'],policies['sft'])
    allpred={};summaries={};sources=[]
    try:
        for number,name in enumerate(p['evaluation_order'],start=1):
            policy=policies[name];llm.set_policy(policy['adapter_path'],policy['adapter_sha256'],number)
            predictions=[];sec=0.
            for start in range(0,len(rows),16):
                batch=rows[start:start+16];prompts=[llm.prompt(r) for r in batch]
                dest=directory/name/f'batch_{start//16:03d}.json'
                immutable(dest.with_name(dest.stem+'_reservation.json'),dict(prompt_ids=[r['prompt_id'] for r in batch],
                    policy_version=llm.policy,decoding=p['evaluation_decoding'],timestamp=now()))
                began=time.monotonic()
                outputs=llm.llm.generate([q[0] for q in prompts],SamplingParams(**p['evaluation_decoding']),
                    lora_request=llm.request,use_tqdm=False)
                elapsed=time.monotonic()-began;sec+=elapsed
                assert len(outputs)==len(batch)
                saved=[]
                for r,q,result in zip(batch,prompts,outputs):
                    assert result.prompt_token_ids==q[1] and len(result.outputs)==1
                    o=result.outputs[0];assert o.finish_reason in ('stop','length') and len(o.token_ids)>0
                    decoder=DecodeStream(ids=q[1],skip_special_tokens=True)
                    assert ''.join(decoder.step(llm.tok._tokenizer,t) or '' for t in o.token_ids)==o.text
                    parsed=parse(o.text,''.join(r['options']),o.finish_reason)
                    saved.append(dict(prompt_id=r['prompt_id'],policy_version=llm.policy,prompt_token_ids=q[1],response_ids=list(o.token_ids),
                        output=o.text,finish_reason=o.finish_reason,answer_set=r['answer_set'],parsed_answer=parsed.answer_set,
                        acc=int(parsed.answer_set==r['answer_set']),strict_format=parsed.valid_format))
                immutable(dest,dict(predictions=saved,seconds=elapsed));sources.append(record(dest));predictions.extend(saved)
                durable(idx/'loss_objective_status_v1.json',dict(status='ABLATION_EVALUATING',policy=name,completed=len(predictions),timestamp=now(),READY_FOR_STAGE5='NO'))
            assert [r['prompt_id'] for r in predictions]==[r['prompt_id'] for r in rows]
            lengths=[len(r['response_ids']) for r in predictions]
            summaries[name]=dict(n=len(rows),correct=sum(r['acc'] for r in predictions),accuracy=float(np.mean([r['acc'] for r in predictions])),
                strict_format=float(np.mean([r['strict_format'] for r in predictions])),unparseable=float(np.mean([r['parsed_answer'] is None for r in predictions])),
                truncation=float(np.mean([r['finish_reason']=='length' for r in predictions])),
                mean_output_length=float(np.mean(lengths)),p95_output_length=float(np.percentile(lengths,95)),
                output_tokens=sum(lengths),seconds=sec,policy=policy)
            allpred[name]=predictions
    finally:llm.close()
    comparisons={label:paired(allpred[a],allpred[b],p['bootstrap_seed']) for label,a,b in [
        ('vanilla_gspo_minus_grpo','vanilla_grpo','vanilla_gspo'),
        ('dynamic_gspo_minus_grpo','dynamic_grpo','dynamic_gspo'),
        ('dynamic_minus_vanilla_grpo','vanilla_grpo','dynamic_grpo'),
        ('dynamic_minus_vanilla_gspo','vanilla_gspo','dynamic_gspo')]}
    contrast=np.array([dg['acc']-vg['acc']-dr['acc']+vr['acc'] for dg,vg,dr,vr in zip(allpred['dynamic_gspo'],allpred['vanilla_gspo'],allpred['dynamic_grpo'],allpred['vanilla_grpo'])])
    rng=np.random.default_rng(p['bootstrap_seed']);samples=np.array([contrast[rng.integers(0,len(contrast),len(contrast))].mean() for _ in range(10000)])
    interaction=dict(delta_pp=float(contrast.mean()*100),ci95_pp=(np.percentile(samples,[2.5,97.5])*100).tolist(),unit='paired prompt',interpretation='descriptive exploratory single-seed interaction',resamples=10000,seed=p['bootstrap_seed'])
    immutable(idx/'loss_objective_eval_analysis_v1.json',dict(timestamp=now(),scope='VALIDATION_AUXILIARY_ABLATION',
        manifest=p['evaluation_manifest'],summary=summaries,paired=comparisons,interaction=interaction,
        bootstrap=dict(seed=p['bootstrap_seed'],resamples=10000,unit='paired prompt',confidence=.95,interval='percentile',multiple_comparisons='Exploratory unadjusted'),
        raw_sources=sources,selection1024_used=False,final_test_used=False))
    assert read(ROOT/'project_state.json')==p['project_state_snapshot']
    immutable(idx/'loss_objective_eval_verification_v1.json',dict(result='PASS',timestamp=now(),
        complete_policy_count=5,prompts_per_policy=len(rows),total_responses=5*len(rows),
        checks=['frozen disjoint dataset and final fixed checkpoints','matched prompt ordering','exact raw token decode',
                'complete one output per request','frozen parser','no resampling valid wrong answers','unchanged project state'],
        analysis=record(idx/'loss_objective_eval_analysis_v1.json'),sources=sources))


if __name__=='__main__':main()
