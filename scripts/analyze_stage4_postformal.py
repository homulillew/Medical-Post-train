#!/usr/bin/env python3
"""Read-only extension of analyze_stage4: exact milestones and cost/health attribution."""
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,now
from review_stage4_pilots import predictions,paired
from analyze_frontier import paired_bootstrap
from frontier_diagnostic import gates
INDEX=ROOT/'experiments/stage4'


def stats(values):
    a=np.asarray(values,dtype=float)
    assert np.isfinite(a).all()
    return dict(n=len(a),min=float(a.min()),max=float(a.max()),mean=float(a.mean()),
                p05=float(np.percentile(a,5)),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)))


def streak(values):
    best=current=0
    for yes in values:
        current=current+1 if yes else 0
        best=max(best,current)
    return best


def first_crossing(rows,threshold):
    return next((r for r in rows if r['rollout_tokens']>=threshold),None)


def main():
    pair=gates()
    base=read(INDEX/'formal_analysis_v1.json')
    out=INDEX/'postformal_data_v1'
    assert not (INDEX/'postformal_analysis_v1.json').exists()
    out.mkdir(exist_ok=True)
    protocol=read(INDEX/'formal_validation_protocol.json')
    runs={};windows={};points={};all_sources=[]
    for name,info in pair['runs'].items():
        path=Path(info['path'])
        # Existing analyzer output was created by the completed formal queue.
        curves=read(INDEX/path.name/'curves.json')
        assert curves['formal_completed'] and len(curves['windows'])==625
        expected={s['path']:s for s in base['runs'][name]['sources']}
        rows=[];counts=Counter();dispositions=Counter();subtypes=Counter();tokens=Counter()
        for i in range(625):
            file=path/'windows'/f'{i:04d}'/'commit.json'
            assert record(file)==expected[str(file)]
            c=read(file);s=c['state_after'];m=c['metrics'];old=c['state_before']
            optfile=file.parent/'update/update.json'
            assert record(optfile)==expected[str(optfile)]
            opt=read(optfile)
            assert s['training_groups']==(i+1)*8 and s['optimizer_steps']==(i+1)*2
            assert s['generated_groups']-old['generated_groups']==m['generated_groups']
            assert sum(m['output_tokens_by_disposition'].values())==m['output_tokens']
            # Class counts determine overflow for this accuracy-only selection rule.
            gc=m['group_counts'];selected=m['selected_groups'];n=m['generated_groups']
            overflow=gc.get('mixed',0)-selected if name=='dynamic' else 0
            dc={'selected':selected,'overflow_eligible':overflow,
                'rejected_all_correct':gc.get('all_correct',0) if name=='dynamic' else 0,
                'rejected_all_wrong':gc.get('all_wrong',0) if name=='dynamic' else 0}
            assert sum(dc.values())==n and overflow>=0
            counts.update(gc);dispositions.update(dc);subtypes.update(m['mixed_subtypes']);tokens.update(m['output_tokens_by_disposition'])
            minis=opt['minibatches'];assert len(minis)==2
            row=dict(window=i+1,training_groups=s['training_groups'],optimizer_steps=s['optimizer_steps'],
                policy_version=s['policy_version'],generated_groups=s['generated_groups'],window_generated_groups=n,
                rollout_tokens=s['prompt_tokens']+s['output_tokens'],output_tokens=s['output_tokens'],prompt_tokens=s['prompt_tokens'],
                window_amplification=n/selected,cumulative_amplification=s['generated_groups']/s['training_groups'],
                selected_groups=selected,expired_eligible_overflow=overflow,selected_group_yield=selected/n,
                **{k:gc.get(k,0)/n for k in ['all_wrong','mixed','all_correct']},
                **{k:m['mixed_subtypes'].get(k,0) for k in ['mixed_parsed_wrong','mixed_unparseable_only','mixed_both']},
                **{'output_tokens_'+k:m['output_tokens_by_disposition'].get(k,0) for k in dc},
                first_mini_clip=minis[0]['clip_fraction'],second_mini_clip=minis[1]['clip_fraction'],
                ratio_min=min(z['ratio']['min'] for z in minis),ratio_max=max(z['ratio']['max'] for z in minis),
                grad_norm_max=max(z['grad_norm'] for z in minis),entropy_mean=float(np.mean([z['entropy'] for z in minis])),
                response_length_mean=m['response_length']['mean'],response_length_p95=m['response_length']['p95'],
                truncation=m['truncation'],parameter_delta_l2=opt['parameter_delta_l2'],
                **{k:m.get(k) for k in ['sleep_seconds','wake_seconds','sync_seconds','actor_process_seconds','nvml_peak_bytes']})
            rows.append(row)
        state=read(path/'checkpoint.json')['state']
        assert sum(counts.values())==state['generated_groups'] and sum(tokens.values())==state['output_tokens']
        assert len(rows)==625
        windows[name]=rows
        points[name]={}
        for directory in sorted((path/'validation').iterdir()):
            if not (directory/'summary.json').exists():continue
            s=read(directory/'summary.json'); w=int(directory.name)
            pred,refs=predictions(path,w);all_sources.extend(refs)
            assert [r['prompt_id'] for r in pred]==protocol['monitor_ids']
            accuracy=sum(r['acc'] for r in pred)/512
            assert accuracy==s['accuracy']
            pt=dict(s,correct_count=sum(r['acc'] for r in pred),
                    unparseable=sum(r['parsed_answer'] is None for r in pred)/512,
                    generated_groups=0 if w==0 else rows[w-1]['generated_groups'],
                    source=record(directory/'summary.json'))
            if w: assert pt['cumulative_generated_total_tokens']==rows[w-1]['rollout_tokens']
            points[name][w]=(pt,pred)
        exposure={}
        for kind in ['generated','training']:
            e=state[kind+'_exposure'];assert sum(e.values())==state['generated_groups' if kind=='generated' else 'training_groups']
            exposure[kind]=dict(unique_prompts=len(e),histogram=dict(Counter(e.values())),max_exposure=max(e.values()),
                                repeated_prompt_count=sum(v>1 for v in e.values()))
        health={k:stats([r[k] for r in rows]) for k in ['ratio_min','ratio_max','first_mini_clip','second_mini_clip',
                    'grad_norm_max','entropy_mean','response_length_mean','response_length_p95','truncation','parameter_delta_l2',
                    'sleep_seconds','wake_seconds','sync_seconds','actor_process_seconds','nvml_peak_bytes']}
        health.update(first_mini_clips_all_zero=all(r['first_mini_clip']==0 for r in rows),
                      max_consecutive_clip_gt_09=streak([r['second_mini_clip']>.9 for r in rows]),
                      optimizer_inactive_windows=sum(r['parameter_delta_l2']<=0 for r in rows),nonfinite_values=0,
                      interpretation='Tight clipping is configured, not an error. Entropy and length distributions/early-late blocks retained; no invented collapse threshold.')
        def block(rr):
            n=sum(r['window_generated_groups'] for r in rr)
            return dict(windows=[rr[0]['window'],rr[-1]['window']],groups_generated=n,
                amplification=n/(len(rr)*8),**{k:sum(r[k]*r['window_generated_groups'] for r in rr)/n for k in ['all_wrong','mixed','all_correct','response_length_mean','truncation']},
                entropy_mean=float(np.mean([r['entropy_mean'] for r in rr])))
        runs[name]=dict(state={k:v for k,v in state.items() if not k.endswith('exposure')},group_counts=dict(counts),
            dispositions=dict(dispositions),mixed_subtypes=dict(subtypes),output_tokens_by_disposition=dict(tokens),
            output_token_fractions={k:v/state['output_tokens'] for k,v in tokens.items()},
            output_tokens_per_generated_group=state['output_tokens']/state['generated_groups'],
            output_tokens_per_trajectory=state['output_tokens']/(4*state['generated_groups']),
            physical_prompt_tokens_per_group=state['prompt_tokens']/state['generated_groups'],
            exposure=exposure,candidate_pool_size=15000,candidate_pool_pass_equivalents=state['generated_groups']/15000,
            candidate_epoch_indices=base['runs'][name]['candidate_epochs_encountered'],
            optimization_health=health,early_first64=block(rows[:64]),late_last64=block(rows[-64:]),
            committed_phase_seconds=base['runs'][name]['committed_phase_seconds'],
            phase_accounting=base['runs'][name]['phase_accounting'],raw_verifier=record(INDEX/f'{name}_formal_verification.json'),
            native_reload=record(path/'final_reload/result.json'))
        with (out/f'{name}_windows.csv').open('x',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    def comparison(va,db):
        a,ap=va;b,bp=db
        return dict(vanilla=a,dynamic=b,paired=paired(ap,bp),
                    bootstrap=paired_bootstrap([r['acc'] for r in ap],[r['acc'] for r in bp]))
    updates=[dict(training_groups=w*8,**comparison(points['vanilla'][w],points['dynamic'][w])) for w in protocol['checkpoint_windows']]
    tokens=[]
    common_max=min(rows[-1]['rollout_tokens'] for rows in windows.values())
    stride=protocol['generated_total_token_stride']
    for threshold in range(stride,common_max+1,stride):
        cross={name:first_crossing(rows,threshold) for name,rows in windows.items()}
        x=comparison(points['vanilla'][cross['vanilla']['window']],points['dynamic'][cross['dynamic']['window']])
        for name,r in cross.items():
            assert r['window']==1 or windows[name][r['window']-2]['rollout_tokens']<threshold
            trigger=read(Path(pair['runs'][name]['path'])/'validation'/f'{r["window"]:04d}'/'trigger.json')
            assert threshold in trigger['generated_total_token_milestones']
            x[name]['overshoot_tokens']=r['rollout_tokens']-threshold
        tokens.append(dict(threshold=threshold,**x))
    for label,rows in [('update_budget',updates),('shared_token_budget',tokens)]:
        flat=[]
        for row in rows:
            for name in ['vanilla','dynamic']:
                p=row[name]
                flat.append(dict(variant=name,milestone=row.get('threshold',row.get('training_groups')),training_groups=p['training_groups'],
                    optimizer_steps=p['optimizer_steps'],accuracy=p['accuracy'],correct_count=p['correct_count'],
                    generated_groups=p['generated_groups'],rollout_tokens=p['cumulative_generated_total_tokens'],
                    overshoot_tokens=p.get('overshoot_tokens',0),strict_format=p['strict_format'],unparseable=p['unparseable'],truncation=p['truncation']))
        with (out/f'{label}.csv').open('x',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=list(flat[0]));wr.writeheader();wr.writerows(flat)
    # Cross-stage conditions differ and are descriptive, not a controlled causal series.
    s3=read(ROOT/'experiments/stage3/s3_formal_20260909T053101_d99393/summary.json')
    s2path=ROOT/'experiments/stage2/s2_formal_20260909T025736_f607d9/summary.json'
    s2=read(s2path)
    pilot=read(INDEX/'pilot_analysis_v1.json')
    history=dict(stage2=dict(expected_sampling_amplification=s2['expected_sampling_amplification'],
                    group_fractions=s2['group_fractions'],scope=s2['amplification_scope'],source=record(s2path)),
                 stage3=dict(amplification=s3['sampling_amplification'],source=record(ROOT/'experiments/stage3/s3_formal_20260909T053101_d99393/summary.json')),
                 pilot_source=record(INDEX/'pilot_analysis_v1.json'),pilot=pilot['runs']['dynamic'] if 'runs' in pilot else pilot.get('dynamic'),
                 formal=runs['dynamic']['state']['generated_groups']/5000,
                 limitation='Different stage protocols/policies/budgets; not a paired-prompt longitudinal claim')
    result=dict(timestamp=now(),scope='FORMAL_MONITOR512_DESCRIPTIVE',pair=record(INDEX/'formal_pair.json'),
        source_analysis=record(INDEX/'formal_analysis_v1.json'),analyzer=record(Path(__file__)),runs=runs,
        accuracy_vs_update_budget=updates,accuracy_vs_shared_rollout_tokens=tokens,common_token_range_max=common_max,
        token_axis=protocol['token_axis'],token_exclusions=protocol['token_exclusions'],amplification_history=history,
        cost_decomposition=dict(group_ratio=runs['dynamic']['state']['generated_groups']/runs['vanilla']['state']['generated_groups'],
            output_tokens_per_group_ratio=runs['dynamic']['output_tokens_per_generated_group']/runs['vanilla']['output_tokens_per_generated_group'],
            total_rollout_token_ratio=windows['dynamic'][-1]['rollout_tokens']/windows['vanilla'][-1]['rollout_tokens']),
        observations=dict(final_delta_pp=100*updates[-1]['paired']['accuracy_delta'],
            largest_observed_group_milestone_delta_pp=max(100*r['paired']['accuracy_delta'] for r in updates),
            largest_delta_group_milestones=[r['training_groups'] for r in updates if r['paired']['accuracy_delta']==max(z['paired']['accuracy_delta'] for z in updates)],
            dynamic_wins_shared_token_thresholds=[r['threshold'] for r in tokens if r['paired']['accuracy_delta']>0]),
        limits=['Single training seed, monitor validation not test. All intervals exploratory and unadjusted for multiple looks.',
                'No interpolation, no checkpoint selection, no test/selection1024 consumed.',
                'Population frontier thinning is descriptive; prompt reuse assessed from real exposure maps.',
                'Rollout token ratios exclude optimizer, validation and sync/control; not GPU-hour ratios.'],
        raw_validation_sources=all_sources,window_source_manifest=base['runs']['vanilla']['sources']+base['runs']['dynamic']['sources'])
    immutable(INDEX/'postformal_analysis_v1.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for label,rows,xfield,xlabel in [('update_budget',updates,'training_groups','Training groups'),
            ('shared_token_budget',tokens,'cumulative_generated_total_tokens','Actual rollout tokens at shared first crossings')]:
        fig,ax=plt.subplots(figsize=(7,4))
        for name in ['vanilla','dynamic']:
            ax.plot([r[name][xfield] for r in rows],[100*r[name]['accuracy'] for r in rows],marker='o',label=name)
        ax.set(xlabel=xlabel,ylabel='Monitor512 accuracy (%)',title='Single-seed validation; no interpolation')
        ax.legend();ax.grid(alpha=.2);fig.tight_layout()
        for ext in ['svg','pdf']:fig.savefig(out/f'{label}.{ext}')
        plt.close(fig)
    immutable(out/'manifest.json',dict(analysis=record(INDEX/'postformal_analysis_v1.json'),files=[record(p) for p in out.iterdir() if p.is_file()]))
    print(json.dumps(result['observations']))


if __name__=='__main__':main()
