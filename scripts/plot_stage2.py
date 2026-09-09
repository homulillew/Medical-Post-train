"""Export plots directly from complete Stage 2 trajectory and group evidence."""
from pathlib import Path
import platform
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from medical_posttrain.evidence import write_json
from medical_posttrain.evidence.stage2 import read,jsonlines,selected_path,record,INDEX

root=selected_path('formal');s=read(root/'summary.json');assert s['completed_responses']==4000
rows=jsonlines(root/'trajectories.jsonl');groups=jsonlines(root/'groups.jsonl');out=INDEX/'figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white'})
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
counts=[s['correct_count_distribution'][str(k)] for k in range(5)]
axes[0,0].bar(range(5),counts,color=['#b65c4a','#d3a64b','#d3a64b','#d3a64b','#3b8d88'])
for i,c in enumerate(counts):axes[0,0].text(i,c+5,str(c),ha='center')
axes[0,0].set(xticks=range(5),xlabel='Correct responses in G=4',ylabel='Prompt groups',title='A  Full 1,000-group correctness distribution',ylim=(0,max(counts)*1.18))
classes={g['prompt_id']:g['classification'] for g in groups}
for name,color in [('all_wrong','#b65c4a'),('mixed','#d3a64b'),('all_correct','#3b8d88')]:
 values=np.sort([r['output_tokens'] for r in rows if classes[r['prompt_id']]==name])
 if len(values):axes[0,1].plot(values,np.arange(1,len(values)+1)/len(values),label=name.replace('_',' '),color=color)
axes[0,1].axvline(1024,color='#555555',linestyle='--',linewidth=1,label='response cap')
axes[0,1].set(xlabel='Output tokens',ylabel='Empirical cumulative fraction',title='B  Response length by group correctness');axes[0,1].legend(fontsize=8,loc='lower right')
for a,color in [(0,'#b65c4a'),(1,'#3b8d88')]:
 values=[r['semantic'] for r in rows if r['acc']==a]
 axes[1,0].hist(values,bins=np.linspace(0,1,31),histtype='step',linewidth=1.5,label=f'acc={a}, n={len(values)}',color=color)
axes[1,0].set(xlabel='Raw semantic score (including explicit missing/empty zeros)',ylabel='Responses',title='C  Semantic overlap does not certify correctness');axes[1,0].legend(fontsize=8)
for a,color in [(0,'#b65c4a'),(1,'#3b8d88')]:
 values=[r['total_reward'] for r in rows if r['acc']==a]
 axes[1,1].hist(values,bins=np.linspace(0,1,41),histtype='step',linewidth=1.5,label=f'acc={a}',color=color)
axes[1,1].axvspan(.05,.8,color='#eeeeee',zorder=0)
axes[1,1].set(xlabel='Total correctness-gated reward',ylabel='Responses',title='D  Wrong <= 0.05; correct >= 0.8');axes[1,1].legend(fontsize=8)
fig.suptitle('CMExam train profiling | fixed medical SFT | 1,000 prompts x 4 responses',fontsize=14)
for suffix in ('png','svg'):
 path=out/f'rollout_profile.{suffix}';assert not path.exists();fig.savefig(path,dpi=170)
 if suffix=='svg':path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
write_json(out/'provenance.json',dict(run_id=root.name,sources=[record(root/'trajectories.jsonl'),record(root/'groups.jsonl'),record(root/'summary.json')],
            script=record(__file__),python=sys.version,matplotlib=matplotlib.__version__,numpy=np.__version__,platform=platform.platform(),
            artifacts=[record(out/'rollout_profile.png'),record(out/'rollout_profile.svg')],
            note='Empirical counts/ECDF/histograms; no interpolated accuracy or invented advantage metrics. SVG line-end whitespace normalized after export.'))
