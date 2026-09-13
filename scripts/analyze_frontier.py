#!/usr/bin/env python3
"""Replay diagnostic raw outputs and produce paired prompt-level mechanism evidence."""
from collections import Counter
import csv
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, now, encounter
from medical_posttrain.reward.parser import parse
from medical_posttrain.sampling.dynamic import classify, Stream

CLASSES = ['all_wrong','mixed_parsed_wrong','mixed_unparseable_only','mixed_both','all_correct']


def paired_bootstrap(a, b, seed=20260913, resamples=10000):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    assert a.shape == b.shape and a.ndim == 1
    if not len(a):
        return dict(n=0, delta=None, ci95=None)
    delta = b-a
    rng = np.random.default_rng(seed)
    draws = np.concatenate([delta[rng.integers(0,len(a),size=(min(250,resamples-i),len(a)))].mean(axis=1)
                            for i in range(0,resamples,250)])
    return dict(n=len(a), a_mean=float(a.mean()), b_mean=float(b.mean()), delta=float(delta.mean()),
                ci95=[float(x) for x in np.quantile(draws,[.025,.975])])


def classify_group(group, policy):
    scored=[]
    for r in group['responses']:
        p = parse(r['raw_output'], ''.join(r['options']), r['finish_reason'])
        scored.append(dict(r, acc=int(p.answer_set==r['ground_truth']), parsed_answer=p.answer_set,
                           strict_format=p.valid_format, parse=p.to_dict()))
    result=classify(scored, policy)
    assert result['valid'], result
    label=result['mixed_subtype'] or result['classification']
    counts=Counter(r['parsed_answer'] for r in scored)
    parsed=Counter({k:v for k,v in counts.items() if k is not None})
    winners=[] if not parsed else [k for k,v in parsed.items() if v==max(parsed.values())]
    return dict(prompt_id=group['prompt_id'], classification=label, correct_count=sum(r['acc'] for r in scored),
                accuracy=sum(r['acc'] for r in scored)/4, any_correct=int(any(r['acc'] for r in scored)),
                all_correct=int(all(r['acc'] for r in scored)), all_wrong=int(not any(r['acc'] for r in scored)),
                mixed=int(result['eligible']), strict_format=sum(r['strict_format'] for r in scored)/4,
                unparseable=sum(r['parsed_answer'] is None for r in scored)/4,
                truncation=sum(r['finish_reason']=='length' for r in scored)/4,
                response_lengths=[r['output_tokens'] for r in scored],
                majority_accuracy=int(len(winners)==1 and winners[0]==scored[0]['ground_truth']),
                consistency=int(len(counts)==1 and None not in counts),
                answer_entropy_bits=-sum((n/4)*math.log2(n/4) for n in counts.values()),
                parsed_answers=[r['parsed_answer'] for r in scored], acc_vector=[r['acc'] for r in scored])


def matrix(a,b):
    return {c:{d:sum(x['classification']==c and y['classification']==d for x,y in zip(a,b))
               for d in CLASSES} for c in CLASSES}


def main():
    from frontier_diagnostic import INDEX, check_prereg, gates
    p=check_prereg();gates()
    artifact=Path(p['artifact_path'])
    from transformers import AutoTokenizer
    from tokenizers.decoders import DecodeStream
    tok=AutoTokenizer.from_pretrained(p['config']['model'],local_files_only=True)
    from medical_posttrain.data.exam import messages
    dataset=read(read(p['manifest']['path'])['dataset']['path'])
    pool={r['prompt_id']:r for r in dataset}
    stream=Stream(list(pool),p['seed'],p['stream_domain'])
    sources=[];policies={};summaries={};raw_artifacts=[]
    for name in p['policy_order']:
        out=artifact/name
        assert read(artifact/(name+'_process')/'exit.json')['exit_code']==0
        assert read(out/'generation_complete.json')['responses']==4000
        policy=p['policies'][name]['adapter_sha256']
        assert record(Path(p['policies'][name]['adapter_path'])/'adapter_model.safetensors')['sha256']==policy
        groups=[];physical_prompt_tokens=0;outputs=0
        paths=sorted((out/'batches').glob('*/raw.json'))
        assert len(paths)==125
        for file in paths:
            raw=read(file);reservation=read(file.parent/'reservation.json')
            assert len(raw['groups'])==8
            for g in raw['groups']:
                idx=len(groups)
                expected=encounter(stream,idx,p['run_id']+'_'+name,policy)
                assert all(g[k]==v for k,v in expected.items())
                assert reservation['encounters'][idx%8]==expected
                source=pool[g['prompt_id']]
                prompt=tok.apply_chat_template(messages(source),tokenize=False,add_generation_prompt=True,enable_thinking=True)
                prompt_ids=tok.encode(prompt,add_special_tokens=False)
                assert g['prompt_tokens']==len(prompt_ids)
                assert len(g['responses'])==4
                for i,r in enumerate(g['responses']):
                    assert r['member_index']==i and r['policy_version']==policy
                    assert r['prompt_id']==g['prompt_id'] and r['group_id']==g['group_id']
                    assert r['ground_truth']==source['answer_set'] and r['options']==source['options']
                    assert r['prompt_token_ids']==prompt_ids and r['request_seed']==expected['request_seed']
                    assert r['config_sha256']==record(INDEX/'preregistration.json')['sha256']
                    assert r['output_tokens']==len(r['token_ids']) and 0<r['output_tokens']<=1024
                    assert r['finish_reason'] in ('stop','length')
                    decoder=DecodeStream(ids=prompt_ids,skip_special_tokens=True)
                    decoded=''.join(decoder.step(tok._tokenizer,t) or '' for t in r['token_ids'])
                    assert decoded==r['raw_output'], ('token_decode',name,idx,i)
                    assert len(r['rollout_raw_logprobs'])==r['output_tokens']
                    assert np.isfinite(r['rollout_raw_logprobs']).all()
                    outputs+=r['output_tokens']
                physical_prompt_tokens+=g['prompt_tokens']
                groups.append(classify_group(g,policy))
            sources.extend([record(file),record(file.parent/'reservation.json')])
        assert len(groups)==1000 and len({g['prompt_id'] for g in groups})==1000
        groups=sorted(groups,key=lambda r:r['prompt_id'])
        policies[name]=groups
        lengths=[n for g in groups for n in g['response_lengths']]
        summary={k:float(np.mean([g[k] for g in groups])) for k in
                 ['accuracy','any_correct','all_correct','mixed','all_wrong','strict_format','unparseable','truncation',
                  'majority_accuracy','consistency','answer_entropy_bits']}
        summary.update(class_counts=dict(Counter(g['classification'] for g in groups)),
                       response_length=dict(mean=float(np.mean(lengths)),p95=float(np.percentile(lengths,95))),
                       physical_prompt_tokens=physical_prompt_tokens,output_tokens=outputs,responses=4000,
                       policy_version=policy)
        summaries[name]=summary
        immutable(INDEX/f'{name}_prompt_metrics.json',groups)
        raw_artifacts.extend(record(f) for f in out.rglob('*') if f.is_file())
    s,v,d=[policies[n] for n in p['policy_order']]
    assert [x['prompt_id'] for x in s]==[x['prompt_id'] for x in v]==[x['prompt_id'] for x in d]
    boot=p['bootstrap']
    compare=lambda a,b: paired_bootstrap(a,b,boot['seed'],boot['resamples'])
    paired={key:compare([x[key] for x in v],[x[key] for x in d]) for key in
            ['accuracy','any_correct','all_correct','mixed','all_wrong','strict_format','unparseable','truncation','majority_accuracy']}
    subsets={}
    for label in ['mixed',*CLASSES]:
        ix=[i for i,x in enumerate(s) if (x['mixed'] if label=='mixed' else x['classification']==label)]
        subsets[label]={target:compare([v[i][target] for i in ix],[d[i][target] for i in ix])
                        for target in ['all_correct','mixed','all_wrong','accuracy']}
    result=dict(timestamp=now(),run_id=p['run_id'],scope=p['scope'],preregistration=record(INDEX/'preregistration.json'),
                optimizer_updates=0,policy_summaries=summaries,paired_dynamic_minus_vanilla=paired,
                sft_subgroups=subsets,transitions=dict(sft_to_vanilla=matrix(s,v),sft_to_dynamic=matrix(s,d),vanilla_to_dynamic=matrix(v,d)),
                bootstrap=boot,hypotheses=p['hypotheses'],
                limitations=['Validation diagnostic only; exploratory intervals and no training-seed replication.',
                             'Same request seeds do not ensure bitwise common random numbers.',
                             'Stochastic G4 classes may change without a stable latent capability change.',
                             'No checkpoint selection, model training, monitor512, selection1024 or test used.'])
    immutable(INDEX/'analysis.json',result)
    immutable(INDEX/'raw_artifacts_manifest.json',raw_artifacts)
    immutable(INDEX/'verification.json',dict(result='PASS',timestamp=now(),scope='FRONTIER_DIAGNOSTIC_RAW_REPLAY',
                responses=12000,prompts_per_policy=1000,optimizer_updates=0,
                checks=['preregistration/source/adapter hashes','disjoint frozen dataset','complete unique matched prompt IDs',
                        'exact encounter seeds','prompt token identity','token decode','frozen parser/classification','finite logprobs',
                        'all raw batches and successful process exits','no optimizer instantiated'],
                sources=sources,analysis=record(INDEX/'analysis.json'),verifier=record(Path(__file__))))
    print({k: v['accuracy'] for k,v in summaries.items()})


if __name__=='__main__':
    main()
