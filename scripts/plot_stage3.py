#!/usr/bin/env python3
"""Standalone descriptive refill/cost figure from completed real batch commits."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(Path(p).read_text())
selected=read(root/'experiments/stage3/selected_runs.json');out=Path(read(root/'experiments/stage3'/selected['formal']/'manifest.json')['artifact_root'])
s=read(out/'summary.json');assert s['status']=='FULL_PASS'
commits=[read(p/'commit.json') for p in sorted((out/'batches').glob('*'))]
fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
x=[0]+[c['state_after']['batches'] for c in commits]
for field,label,color in [('generated_groups','Generated groups','#536878'),('accepted_mixed_groups','Accepted mixed','#168a77')]:
    axes[0].step(x,[0]+[c['state_after'][field] for c in commits],where='post',label=label,color=color,linewidth=2)
axes[0].axhline(256,linestyle=':',color='#222222',label='Target256');axes[0].set(xlabel='Committed refill batch',ylabel='Cumulative groups',title='Real fixed-policy refill');axes[0].legend(frameon=False)
costs=s['costs']['output_tokens_by_disposition'];labels={'accepted':'Accepted','rejected_all_wrong':'Rejected0/4','rejected_all_correct':'Rejected4/4','overflow_eligible':'Overflow','invalid':'Invalid'}
colors=['#168a77','#bf6954','#d8a45a','#7f73a1','#bbbbbb'];left=0
for (key,value),color in zip(costs.items(),colors):
    if not value:continue
    axes[1].barh(['Generated output'],[value/1000],left=left/1000,label=f'{labels[key]}: {value:,}',color=color,height=.4);left+=value
axes[1].set(xlabel='Output tokens (thousands)',title=f'Accepted token fraction: {s["costs"]["accepted_token_fraction"]:.1%}')
axes[1].legend(loc='upper center',bbox_to_anchor=(.5,-.16),frameon=False,ncol=2)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.suptitle('Stage3 integration — no policy updates; rejected rollout cost is retained',fontsize=12)
path=root/'experiments/stage3/figures';path.mkdir(exist_ok=True)
fig.savefig(path/'refill_costs.png',dpi=180);fig.savefig(path/'refill_costs.svg')
print(path/'refill_costs.png')
