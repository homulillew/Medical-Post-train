"""Plot retained SFT evidence; run with the existing base analysis Python."""
import argparse
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args();root=Path(args.run)
    rows=[json.loads(l) for l in (root/'canonical_metrics.jsonl').open()];summary=json.loads((root/'summary.json').read_text())
    assert summary['run_class']=='FORMAL' and summary['examples']==20000
    out=Path('experiments/stage1/figures');out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    steps=[r['global_step'] for r in rows];loss=[r['loss'] for r in rows]
    axes[0,0].plot(steps,loss,color='#5577aa',alpha=.25,lw=.5,label='Train: raw update loss')
    axes[0,0].plot(steps,[statistics.mean(loss[max(0,i-49):i+1]) for i in range(len(loss))],color='#24508b',lw=1.5,label='Train: trailing 50-update mean')
    for n,marker,color in [(128,'o','#dd8b26'),(1000,'s','#b02c35')]:
        values=[r for r in summary['validation'] if r['examples']==n]
        axes[0,0].plot([r['global_step'] for r in values],[r['loss'] for r in values],marker=marker,color=color,lw=0,label=f'Validation: {n} fixed examples')
    axes[0,0].set(ylabel='Assistant-token mean NLL',xlabel='Optimizer update');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(steps,[r['learning_rate'] for r in rows],color='#386b54');axes[0,1].set(xlabel='Optimizer update',ylabel='Learning rate')
    axes[1,0].plot(steps,[r['sample_cursor'] for r in rows],color='#24508b');axes[1,0].axhline(20000,color='gray',linestyle='--',lw=.8);axes[1,0].set(xlabel='Optimizer update',ylabel='Unique examples consumed')
    axes[1,1].plot(steps,[r['nvml_peak_bytes']/2**30 for r in rows],label='NVML cumulative peak');axes[1,1].plot(steps,[r['allocated_peak_bytes']/2**30 for r in rows],label='Tensor allocation peak');axes[1,1].set(xlabel='Optimizer update',ylabel='GiB');axes[1,1].legend(fontsize=8)
    for ax in axes.flat:ax.grid(alpha=.2)
    fig.suptitle('Medical SFT: complete 20,000-example epoch')
    fig.savefig(out/'sft_training.png',dpi=160);fig.savefig(out/'sft_training.svg');plt.close(fig)
    (out/'provenance.json').write_text(json.dumps(dict(run_id=summary['run_id'],matplotlib_version=matplotlib.__version__,source_metrics=str(root/'canonical_metrics.jsonl'),smoothing='trailing mean of up to 50 observed updates; raw losses retained',validation_note='128-example monitor and 1000-example full validation are shown separately'),indent=2)+'\n')

if __name__=='__main__':main()
