#!/usr/bin/env python3
"""Export paired scientific curves from committed raw-evidence reconstructions."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import INDEX,read,record,durable
from analyze_stage4 import analyze


def plot(pair_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pair=read(pair_path)
    data={v:analyze(Path(ref['path'])) for v,ref in pair['runs'].items()}
    directory=INDEX/(pair['mode'].lower()+'_plots')
    directory.mkdir(exist_ok=True)
    colors={'vanilla':'#2563eb','dynamic':'#d97706'}
    refs=[]
    def figure(name,x,y,xlabel,ylabel,validation=False):
        if validation and not any(r['validation'] for r in data.values()):
            return
        fig,ax=plt.subplots(figsize=(6.6,4.2),layout='constrained')
        for v,r in data.items():
            rows=r['validation'] if validation else r['windows']
            if rows:
                ax.plot([x(z) for z in rows],[y(z) for z in rows],label=v,color=colors[v],marker='o' if validation else None,
                    linestyle='none' if validation and pair['mode']=='PILOT' else '-',markersize=3)
        ax.set(xlabel=xlabel,ylabel=ylabel,title=pair['mode']+' · '+name.replace('_',' '))
        ax.grid(alpha=.2)
        if ax.lines:
            ax.legend(frameon=False)
        for suffix in ('svg','pdf'):
            path=directory/(name+'.'+suffix)
            fig.savefig(path)
            refs.append(record(path))
        plt.close(fig)
    figure('validation_accuracy_vs_training_groups',lambda z:z['training_groups'],lambda z:z['accuracy'],'Training groups','Exact answer-set accuracy',True)
    figure('validation_accuracy_vs_generated_output_tokens',lambda z:z['cumulative_generated_output_tokens'],lambda z:z['accuracy'],'Generated output tokens (training rollout)','Exact answer-set accuracy',True)
    figure('validation_accuracy_vs_generated_total_tokens',lambda z:z['cumulative_generated_total_tokens'],lambda z:z['accuracy'],'Generated prompt + output tokens (training rollout)','Exact answer-set accuracy',True)
    for name,fn,label in (
        ('pre_filter_mixed',lambda z:z['p_mixed'],'Pre-filter mixed fraction'),
        ('pre_filter_all_correct',lambda z:z['p_all_correct'],'Pre-filter all-correct fraction'),
        ('pre_filter_all_wrong',lambda z:z['p_all_wrong'],'Pre-filter all-wrong fraction'),
        ('sampling_amplification',lambda z:z['cumulative_amplification'],'Generated groups / training groups'),
        ('mixed_unparseable_only',lambda z:z['mixed_unparseable_only_share'],'Unparseable-only / mixed groups'),
        ('strict_format',lambda z:z['strict_format'],'Strict format fraction (all rollout)'),
        ('response_length_mean',lambda z:z['response_length']['mean'],'Mean response tokens (all rollout)'),
        ('response_length_p95',lambda z:z['response_length']['p95'],'P95 response tokens (all rollout)'),
        ('entropy',lambda z:sum(m['entropy'] for m in z['optimization'])/2,'Mean policy entropy (training trajectories)'),
        ('objective_clip_second_mini',lambda z:z['optimization'][1]['clip_fraction'],'Objective clip fraction (second mini)'),
        ('sequence_ratio_min',lambda z:min(m['ratio']['min'] for m in z['optimization']),'Minimum sequence importance ratio'),
        ('sequence_ratio_max',lambda z:max(m['ratio']['max'] for m in z['optimization']),'Maximum sequence importance ratio'),
        ('gradient_norm',lambda z:max(m['grad_norm'] for m in z['optimization']),'Maximum pre-clip gradient norm'),
        ('reward',lambda z:z['reward']['mean'],'Hybrid reward (all rollout)')):
        figure(name,lambda z:z['training_groups'],fn,'Training groups',label)
    durable(directory/'manifest.json',dict(pair=record(pair_path),figures=refs,
        scope='Committed windows; no interpolation or missing-result backfill',
        costs='Validation/control generations excluded from training rollout x axes and retained separately'))
    return refs


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--pair',required=True,type=Path)
    print(len(plot(p.parse_args().pair)),'figure artifacts')
