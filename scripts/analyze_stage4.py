#!/usr/bin/env python3
"""Rebuild compact per-window curves from immutable committed Stage4 evidence."""
import argparse
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,durable,now


def analyze(path):
    cfg=read(path/'config.json')
    windows=[]
    costs=Counter()
    cases=[]
    for w in sorted((path/'windows').glob('*')):
        if not (w/'commit.json').exists():
            continue
        commit=read(w/'commit.json')
        s=commit['state_after']
        m=commit['metrics']
        opt=read(w/'update/update.json')
        count=m['valid_generated_groups']
        mixed=m['group_counts'].get('mixed',0)
        costs.update(m['output_tokens_by_disposition'])
        row=dict(policy_windows=s['policy_windows'],optimizer_steps=s['optimizer_steps'],
            training_groups=s['training_groups'],generated_groups=s['generated_groups'],
            cumulative_output_tokens=s['output_tokens'],cumulative_prompt_tokens=s['prompt_tokens'],
            cumulative_total_tokens=s['output_tokens']+s['prompt_tokens'],
            window_amplification=m['sampling_amplification'],cumulative_amplification=s['generated_groups']/s['training_groups'],
            p_all_wrong=m['group_counts'].get('all_wrong',0)/count,
            p_mixed=mixed/count,p_all_correct=m['group_counts'].get('all_correct',0)/count,
            mixed_unparseable_only_share=m['mixed_subtypes'].get('mixed_unparseable_only',0)/mixed if mixed else None,
            acc=m['acc'],reward=m['reward'],strict_format=m['strict_format'],unparseable=m['unparseable'],
            semantic=m['semantic'],semantic_contribution=m['semantic_contribution'],
            think_closure=m['think_closure'],answer_closure=m['answer_closure'],
            response_length=m['response_length'],truncation=m['truncation'],
            optimization=opt['minibatches'],parameter_delta_l2=opt['parameter_delta_l2'],
            system={k:m.get(k) for k in ('sleep_seconds','wake_seconds','actor_process_seconds','sync_seconds','nvml_peak_bytes')})
        windows.append(row)
        cases.extend(dict(window=w.name,mini=m['mini_index'],ratio=m['ratio'],clip=m['clip_fraction'],
            grad_norm=m['grad_norm'],entropy=m['entropy'],loss=m['policy_loss'],
            source=record(w/'update/update.json')) for m in opt['minibatches'])
    if not windows:
        return dict(status='NO_COMMITTED_WINDOWS',run_id=path.name)
    validations=[read(p) for p in sorted((path/'validation').glob('*/summary.json'))]
    state=read(path/'checkpoint.json')['state']
    exposure={}
    for kind in ('generated','training'):
        values=state[kind+'_exposure'].values()
        exposure[kind]=dict(unique_prompts=len(values),repeat_histogram=dict(Counter(map(str,values))),max_exposure=max(values))
    extremes={
        'largest_ratio_drift':max(cases,key=lambda c:max(abs(c['ratio']['min']-1),abs(c['ratio']['max']-1))),
        'highest_clip':max(cases,key=lambda c:c['clip']),
        'highest_grad_norm':max(cases,key=lambda c:c['grad_norm']),
        'lowest_entropy':min(cases,key=lambda c:c['entropy']),
        'largest_absolute_loss':max(cases,key=lambda c:abs(c['loss']))}
    result=dict(run_id=path.name,mode=cfg['mode'],sampling_mode=cfg['sampling_mode'],
        status=read(path/'status.json')['status'],timestamp=now(),windows=windows,validation=validations,
        output_tokens_by_disposition=dict(costs),exposure=exposure,optimization_cases=extremes,
        metrics_scope='Committed online windows only; original run ledger retains failures and uncommitted costs separately',
        formal_completed=False if cfg['mode']!='formal' else state['training_groups']==5000)
    durable(INDEX/path.name/'curves.json',result)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--run',required=True)
    a=p.parse_args()
    r=analyze(Path(a.run))
    print({k:v for k,v in r.items() if k not in ('windows','validation','optimization_cases')})


if __name__=='__main__':
    main()
