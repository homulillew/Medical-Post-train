#!/usr/bin/env python3
"""Measured formal comparison after both raw verifiers; no checkpoint selection."""
from collections import Counter,defaultdict
from pathlib import Path
import sys
import statistics
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import INDEX,read,record,immutable,now
from medical_posttrain.rl.transactions import cost_ledger
from review_stage4_pilots import predictions,paired
from analyze_stage4 import analyze


def inspect_run(path):
    cfg=read(path/'config.json');state=read(path/'checkpoint.json')['state']
    assert cfg['mode']=='formal' and state['training_groups']==5000
    curves=analyze(path);assert curves['formal_completed'] and len(curves['windows'])==625
    counts=Counter();dispositions=Counter();subtypes=Counter();selected=Counter();nonzero=Counter();phases=Counter()
    history=defaultdict(list);blocks=[];block=[];sources=[];extremes={};invalid=0
    for i in range(625):
        w=path/'windows'/f'{i:04d}';commit=read(w/'commit.json');m=commit['metrics'];update=read(w/'update/update.json')
        counts.update(m['group_counts']);dispositions.update(m['output_tokens_by_disposition']);subtypes.update(m['mixed_subtypes'])
        invalid+=m['generated_groups']-m['valid_generated_groups']
        selected_groups=read(w/'selection.json')['groups'];selected_ids={g['group_id'] for g in selected_groups}
        with np.load(w/'update/old.npz') as old:
            for j,g in enumerate(selected_groups):
                c=sum(r['acc'] for r in g['responses']);kind='all_wrong' if c==0 else 'all_correct' if c==4 else 'mixed'
                selected[kind]+=1;nonzero[kind]+=int(np.max(np.abs(old['advantages'][j*4:j*4+4]))>1e-6)
        lengths=[];block_counts=Counter();phase=Counter()
        for batch in sorted((w/'batches').glob('*')):
            raw=read(batch/'raw.json');phase['generation_seconds']+=raw['generation_seconds']
            phase['reward_seconds']+=read(batch/'reward_runtime.json')['seconds']
            scored=read(batch/'scored.json')['groups']
            for g in scored:
                c=sum(r['acc'] for r in g['responses']);kind='all_wrong' if c==0 else 'all_correct' if c==4 else 'mixed'
                block_counts[kind]+=1
                history[g['prompt_id']].append(dict(window=i+1,group_id=g['group_id'],candidate_epoch=g['candidate_epoch'],
                    encounter_index=g['encounter_index'],classification=kind,acc_vector=[r['acc'] for r in g['responses']],
                    selected=g['group_id'] in selected_ids,source_path=str(batch/'scored.json')))
                for r in g['responses']:
                    lengths.append(r['output_tokens'])
                    if 'longest_response' not in extremes or r['output_tokens']>extremes['longest_response']['response']['output_tokens']:
                        extremes['longest_response']=dict(window=i+1,response=r,source=record(batch/'scored.json'))
                    if not r['acc'] and ('wrong_high_semantic' not in extremes or r['semantic']>extremes['wrong_high_semantic']['response']['semantic']):
                        extremes['wrong_high_semantic']=dict(window=i+1,response=r,source=record(batch/'scored.json'))
        phase.update({k:m[k] for k in ('actor_process_seconds','sleep_seconds','wake_seconds','sync_seconds')})
        phases.update(phase)
        # These subphases are nested in actor_process_seconds, not additive to it.
        phases['actor_nested_old_logprob_seconds']+=update['old_logprob_seconds']
        phases['actor_nested_optimizer_seconds']+=sum(z['seconds'] for z in update['minibatches'])
        phases['actor_nested_checkpoint_seconds']+=read(w/'checkpoint/COMMITTED.json')['seconds']
        block.append(dict(window=i+1,counts=block_counts,lengths=lengths,unparseable=m['unparseable'],
            entropy=statistics.mean(z['entropy'] for z in update['minibatches']),clip=update['minibatches'][1]['clip_fraction']))
        sources.extend([record(w/'commit.json'),record(w/'update/update.json')])
        if len(block)==16 or i==624:
            n=sum(sum(z['counts'].values()) for z in block);cc=sum((z['counts'] for z in block),Counter())
            ll=[v for z in block for v in z['lengths']]
            blocks.append(dict(windows=[block[0]['window'],block[-1]['window']],training_groups=len(block)*8,generated=n,
                amplification=n/(len(block)*8),group_counts=dict(cc),prefilter={k:cc[k]/n for k in ('all_wrong','mixed','all_correct')},
                output_length=dict(mean=float(np.mean(ll)),p95=float(np.percentile(ll,95)),p99=float(np.percentile(ll,99))),
                entropy=float(np.mean([z['entropy'] for z in block])),second_mini_clip=float(np.mean([z['clip'] for z in block]))))
            block=[]
    assert sum(counts.values())+invalid==state['generated_groups'] and sum(dispositions.values())==state['output_tokens']
    assert sum(selected.values())==5000
    costs=cost_ledger(path)
    assert costs['known_generated_groups']==state['generated_groups'] and costs['known_output_tokens']==state['output_tokens']
    validations=curves['validation']
    phases['validation_seconds']=sum(v['seconds'] for v in validations)
    repeated={pid:h for pid,h in history.items() if len(h)>1}
    transitions=Counter(f'{a["classification"]}->{b["classification"]}' for h in repeated.values() for a,b in zip(h,h[1:]))
    immutable(INDEX/path.name/'formal_prompt_history.json',dict(run_id=path.name,repeated_prompts=repeated,
        transitions=dict(transitions),interpretation='Repeated stochastic G4 encounters under changed policy/context exposure; changes are observed transitions, not proof of causally learned medical knowledge'))
    return dict(run_id=path.name,state={k:v for k,v in state.items() if not k.endswith('exposure')},
        group_counts=dict(counts),invalid_generated_groups=invalid,mixed_subtypes=dict(subtypes),selected_classes=dict(selected),nonzero_advantage_groups=dict(nonzero),
        output_tokens_by_disposition=dict(dispositions),selected_output_token_fraction=dispositions['selected']/state['output_tokens'],
        physical_cost=costs,logical_trajectory_prompt_tokens=4*state['prompt_tokens'],committed_phase_seconds=dict(phases),
        phase_accounting='Actor nested old-logprob/optimizer/checkpoint durations are included in actor_process_seconds; do not sum twice. Crash-tail time absent from durable events remains unknown.',
        validation_output_tokens=sum(v['validation_output_tokens'] for v in validations),
        validation_prompt_tokens=sum(v['validation_prompt_tokens'] for v in validations),
        exposure=curves['exposure'],candidate_epochs_encountered=sorted({x['candidate_epoch'] for h in history.values() for x in h}),
        prompt_history=record(INDEX/path.name/'formal_prompt_history.json'),blocks=blocks,validation=validations,
        optimization_extremes=curves['optimization_cases'],automatic_cases=extremes,sources=sources)


def main():
    pair=read(INDEX/'formal_pair.json');paths={v:Path(r['path']) for v,r in pair['runs'].items()}
    for v in paths:
        receipt=read(INDEX/f'{v}_formal_verification.json');assert receipt['result']=='PASS' and receipt['state']['training_groups']==5000
    runs={v:inspect_run(p) for v,p in paths.items()}
    initial={};final={};raw_sources=[]
    for v,p in paths.items():
        initial[v],ss=predictions(p,0);raw_sources+=ss
        final[v],ss=predictions(p,625);raw_sources+=ss
    token_points={}
    for v,p in paths.items():
        token_points[v]={t:dict(window=int(f.parent.name),actual=read(f)['actual_generated_total_tokens'])
            for f in sorted((p/'validation').glob('*/trigger.json')) for t in read(f)['generated_total_token_milestones']}
    common=sorted(set(token_points['vanilla']) & set(token_points['dynamic']));matched=[]
    for t in common:
        a,b=token_points['vanilla'][t],token_points['dynamic'][t]
        va,ss=predictions(paths['vanilla'],a['window']);raw_sources+=ss
        db,ss=predictions(paths['dynamic'],b['window']);raw_sources+=ss
        matched.append(dict(threshold=t,vanilla=a,dynamic=b,vanilla_accuracy=sum(r['acc'] for r in va)/512,
            dynamic_accuracy=sum(r['acc'] for r in db)/512,paired_descriptive=paired(va,db)))
    result=dict(scope='FORMAL_MONITOR512_DESCRIPTIVE',timestamp=now(),pair=record(INDEX/'formal_pair.json'),runs=runs,
        initial_predictions_exact=initial['vanilla']==initial['dynamic'],
        paired=dict(sft_to_vanilla=paired(initial['vanilla'],final['vanilla']),sft_to_dynamic=paired(initial['dynamic'],final['dynamic']),
            vanilla_to_dynamic=paired(final['vanilla'],final['dynamic'])),shared_token_thresholds=matched,
        validation_raw_sources=raw_sources,
        limits=['One training seed; paired question tests are descriptive and unadjusted for multiple looks.',
            'Monitor512 only; no selection1024 or test evaluation.',
            'Shared thresholds have measured window-level overshoot, not identical token totals.',
            'No missing accuracy interpolation or best-checkpoint selection.',
            'Population group changes do not establish within-prompt learning. Repeated-prompt histories are explicitly retained.'])
    immutable(INDEX/'formal_analysis_v1.json',result)
    index={}
    for v,path in paths.items():
        def entry(n):
            if n==0:
                init=read(read(path/'config.json')['initialization']['path'])
                adapter=Path(init['adapter_path'])/'adapter_model.safetensors'
            else:adapter=path/'windows'/f'{n-1:04d}'/'checkpoint/adapter/adapter_model.safetensors'
            return dict(policy_windows=n,training_groups=n*8,optimizer_steps=n*2,adapter=record(adapter),
                monitor=record(path/'validation'/f'{n:04d}'/'summary.json'),
                trigger=record(path/'validation'/f'{n:04d}'/'trigger.json'))
        index[v]=dict(final=entry(625),validation_checkpoints=[entry(int(p.name)) for p in sorted((path/'validation').glob('*'))],
            selection_performed=False)
    immutable(INDEX/'formal_checkpoint_index.json',index)
    print({v:dict(training_groups=r['state']['training_groups'],generated_groups=r['state']['generated_groups'],validation_points=len(r['validation'])) for v,r in runs.items()})


if __name__=='__main__':main()
