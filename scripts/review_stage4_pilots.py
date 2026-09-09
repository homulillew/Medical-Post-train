#!/usr/bin/env python3
"""Descriptive paired pilot analysis; never selects a test checkpoint or formal budget."""
from pathlib import Path
from collections import Counter
import math
import statistics
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import INDEX,read,record,immutable,now


def predictions(path,window):
    directory=path/'validation'/f'{window:04d}'
    files=sorted(directory.glob('batch_*.json'))
    assert len(files)==32
    rows=[r for p in files for r in read(p)['predictions']]
    assert len(rows)==512 and len({r['prompt_id'] for r in rows})==512
    return rows,[record(p) for p in files]


def paired(a,b):
    assert [r['prompt_id'] for r in a]==[r['prompt_id'] for r in b]
    gains=[(x,y) for x,y in zip(a,b) if not x['acc'] and y['acc']]
    regressions=[(x,y) for x,y in zip(a,b) if x['acc'] and not y['acc']]
    n=len(gains)+len(regressions)
    # Two-sided exact discordance test, conditional on observed discordant
    # questions. Descriptive, unadjusted; not inference over training seeds.
    p=min(1.,2*sum(math.comb(n,k) for k in range(min(len(gains),len(regressions))+1))/2**n) if n else 1.
    return dict(corrected=len(gains),regressed=len(regressions),net_correct=len(gains)-len(regressions),
        accuracy_delta=(len(gains)-len(regressions))/512,exact_p_unadjusted=p,
        corrected_from_unparseable=sum(x['parsed_answer'] is None for x,y in gains),
        regressed_to_unparseable=sum(y['parsed_answer'] is None for x,y in regressions),
        corrected_ids=[x['prompt_id'] for x,y in gains],regressed_ids=[x['prompt_id'] for x,y in regressions])


def run(path):
    windows=sorted((path/'windows').glob('*'))
    assert len(windows)==64
    state=read(path/'checkpoint.json')['state']
    assert state['training_groups']==512 and state['optimizer_steps']==128
    counts,subtypes,dispositions,selected_classes,nonzero=Counter(),Counter(),Counter(),Counter(),Counter()
    rows,sources=[],[]
    for w in windows:
        commit=read(w/'commit.json'); m=commit['metrics']; opt=read(w/'update/update.json')
        assert all(np.isfinite(z[k]) for z in opt['minibatches'] for k in ('policy_loss','grad_norm','entropy'))
        assert opt['parameter_delta_l2']>0
        groups=read(w/'selection.json')['groups']
        with np.load(w/'update/old.npz') as saved:
            for i,g in enumerate(groups):
                c=sum(r['acc'] for r in g['responses'])
                kind='all_wrong' if c==0 else 'all_correct' if c==4 else 'mixed'
                selected_classes[kind]+=1
                nonzero[kind]+=int(np.max(np.abs(saved['advantages'][i*4:i*4+4]))>1e-6)
        counts.update(m['group_counts']);subtypes.update(m['mixed_subtypes']);dispositions.update(m['output_tokens_by_disposition'])
        gen=sum(read(b/'raw.json')['generation_seconds'] for b in (w/'batches').glob('*'))
        reward=sum(read(b/'reward_runtime.json')['seconds'] for b in (w/'batches').glob('*'))
        seconds=gen+reward+sum(m[k] for k in ('actor_process_seconds','sleep_seconds','wake_seconds','sync_seconds'))
        rows.append(dict(window=int(w.name)+1,generated=m['generated_groups'],mixed=m['group_counts'].get('mixed',0),
            all_correct=m['group_counts'].get('all_correct',0),unparseable=m['unparseable'],
            length=m['response_length']['mean'],clip_second=opt['minibatches'][1]['clip_fraction'],
            clip_first=opt['minibatches'][0]['clip_fraction'],entropy=statistics.mean(z['entropy'] for z in opt['minibatches']),
            max_gradient=max(z['grad_norm'] for z in opt['minibatches']),
            ratio_min=min(z['ratio']['min'] for z in opt['minibatches']),
            ratio_max=max(z['ratio']['max'] for z in opt['minibatches']),seconds=seconds))
        sources.extend([record(w/'commit.json'),record(w/'update/update.json')])
    blocks=[]
    for i in range(0,64,16):
        block=rows[i:i+16];n=sum(z['generated'] for z in block)
        blocks.append(dict(windows=[i+1,i+16],generated=n,amplification=n/128,
            prefilter_mixed=sum(z['mixed'] for z in block)/n,
            prefilter_all_correct=sum(z['all_correct'] for z in block)/n,
            unparseable=sum(z['unparseable']*z['generated'] for z in block)/n,
            mean_output_length=sum(z['length']*z['generated'] for z in block)/n,
            entropy=statistics.mean(z['entropy'] for z in block),
            second_mini_clip=statistics.mean(z['clip_second'] for z in block)))
    vals=[read(path/'validation'/f'{i:04d}'/'summary.json') for i in (0,64)]
    assert sum(counts.values())==state['generated_groups'] and sum(dispositions.values())==state['output_tokens']
    return dict(run_id=path.name,training_groups=512,windows=64,optimizer_steps=128,
        generated_groups=state['generated_groups'],output_tokens=state['output_tokens'],prompt_tokens=state['prompt_tokens'],
        total_tokens=state['output_tokens']+state['prompt_tokens'],amplification=state['generated_groups']/512,
        group_counts=dict(counts),mixed_subtypes=dict(subtypes),output_tokens_by_disposition=dict(dispositions),
        selected_classes=dict(selected_classes),nonzero_advantage_groups=dict(nonzero),
        selected_output_token_fraction=dispositions['selected']/state['output_tokens'],
        training_phase_hours=sum(z['seconds'] for z in rows)/3600,
        conditional_formal_training_phase_hours=sum(z['seconds'] for z in rows)/64*625/3600,
        mean_second_mini_clip=statistics.mean(z['clip_second'] for z in rows),
        max_second_mini_clip=max(z['clip_second'] for z in rows),
        first_mini_clips_all_zero=all(z['clip_first']==0 for z in rows),
        ratio_min=min(z['ratio_min'] for z in rows),ratio_max=max(z['ratio_max'] for z in rows),
        max_gradient=max(z['max_gradient'] for z in rows),
        blocks=blocks,validation=vals,sources=sources)


def main():
    pair=read(INDEX/'pilot_pair.json')
    paths={v:Path(r['path']) for v,r in pair['runs'].items()}
    for v in paths:
        assert read(INDEX/f'{v}_pilot_verification.json')['result']=='PASS'
    a,ar=predictions(paths['vanilla'],0);b,br=predictions(paths['dynamic'],0)
    assert a==b, 'Independent initial greedy monitor controls differ'
    v,vr=predictions(paths['vanilla'],64);d,dr=predictions(paths['dynamic'],64)
    runs={k:run(p) for k,p in paths.items()}
    result=dict(timestamp=now(),scope='PILOT_ANALYSIS_ONLY',runs=runs,
        initial_predictions_exact=True,paired=dict(sft_to_vanilla=paired(a,v),sft_to_dynamic=paired(a,d),vanilla_to_dynamic=paired(v,d)),
        ratios=dict(output_tokens=runs['dynamic']['output_tokens']/runs['vanilla']['output_tokens'],
            total_tokens=runs['dynamic']['total_tokens']/runs['vanilla']['total_tokens'],
            training_phase_time=runs['dynamic']['training_phase_hours']/runs['vanilla']['training_phase_hours']),
        validation_raw_sources=ar+br+vr+dr,
        limits=['One training seed per variant; question-wise paired tests do not measure seed variability.',
            'Monitor512 only, no test or selection1024 use. Three paired tests are exploratory and unadjusted.',
            'Only initial/final validation points; no matched-generated-token checkpoint comparison exists.',
            'Prefilter block trends use changing prompts; do not identify within-prompt policy-frontier movement.',
            'Fault-injection gate and formal config/code freeze are still required; no formal run is complete.'])
    immutable(INDEX/'pilot_analysis_v1.json',result)
    print({v:{k:r[k] for k in ('selected_classes','nonzero_advantage_groups','mean_second_mini_clip','max_second_mini_clip','max_gradient','training_phase_hours')} for v,r in runs.items()})
    print('cost_ratios',result['ratios'])


if __name__=='__main__':
    main()
