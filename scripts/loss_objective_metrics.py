"""Offline retained-array diagnostics; these equations never drive optimization."""
from pathlib import Path
import numpy as np
from medical_posttrain.rl.common import read

BINS=['<192','[192,256)','[256,384)','>=384']
def length_bin(n):return BINS[0 if n<192 else 1 if n<256 else 2 if n<384 else 3]
def stats(a):
    a=np.asarray(a,dtype=np.float64)
    if not a.size:return None
    assert np.isfinite(a).all()
    return dict(min=float(a.min()),max=float(a.max()),mean=float(a.mean()),std=float(a.std()),
        **{f'p{p:02d}':float(np.percentile(a,p)) for p in [1,5,50,95,99]})

def metrics(path,cfg,objective='grpo'):
    path=Path(path);old=np.load(path/'old.npz');summary=read(path/'update.json');minis=[]
    for j,mini in enumerate(summary['minibatches']):
        e=np.load(path/f'mini_{j:02d}.npz');rows=[]
        for k,i in enumerate(e['indices']):
            mask=old['mask'][i].astype(bool);n=int(mask.sum())
            d=(e['current_logprobs'][k]-old['old_logprobs'][i])[mask].astype(np.float64)
            a=old['advantages'][i][mask].astype(np.float64)
            ratio=np.exp(np.clip(d,-20,20)) if objective=='grpo' else np.full(n,np.exp(min(d.mean(),10)))
            low=ratio<1-cfg['clip_ratio_low'];high=ratio>1+cfg['clip_ratio_high']
            clipped=np.clip(ratio,1-cfg['clip_ratio_low'],1+cfg['clip_ratio_high'])
            plain=-a*ratio;bound=-a*clipped;sur=np.maximum(plain,bound)
            dual=(a<0)&(sur>-a*3) if objective=='grpo' else np.zeros(n,dtype=bool)
            if objective=='grpo':sur=np.where(a<0,np.minimum(-a*3,sur),sur)
            rows.append(dict(index=int(i),length=n,length_bin=length_bin(n),ratio=ratio,delta=d,
                lower=low,upper=high,objective_clip=bound>plain,dual_clip=dual,surrogate=sur,
                entropy=float(e['entropy'][k]),derived_sequence_ratio=float(np.exp(d.mean()))))
        def aggregate(rs):
            if not rs:return dict(trajectories=0,status='NO_SAMPLES')
            cat=lambda key:np.concatenate([r[key] for r in rs])
            return dict(trajectories=len(rs),tokens=sum(r['length'] for r in rs),ratio=stats(cat('ratio')),
                ratio_absolute_deviation=stats(abs(cat('ratio')-1)),lower_clip_fraction=float(cat('lower').mean()),
                upper_clip_fraction=float(cat('upper').mean()),total_bound_clip_fraction=float((cat('lower')|cat('upper')).mean()),
                objective_clip_fraction=float(cat('objective_clip').mean()),dual_clip_fraction=float(cat('dual_clip').mean()),
                approximate_kl_proxy=float((-np.clip(cat('delta'),-20,20) if objective=='grpo' else -cat('delta')).mean()),
                absolute_surrogate=float(abs(cat('surrogate')).mean()),entropy=float(np.mean([r['entropy'] for r in rs])),
                gradient_related_proxy='NOT_IDENTIFIABLE: no per-trajectory parameter gradients retained',
                derived_sequence_ratio=stats([r['derived_sequence_ratio'] for r in rs]))
        minis.append(dict(mini_index=j,**aggregate(rows),length_bins={b:aggregate([r for r in rows if r['length_bin']==b]) for b in BINS},
            grad_norm=mini['grad_norm'],loss=mini['policy_loss'],nonfinite_count=0))
    norm=summary.get('initial_parameter_l2')
    return dict(objective=objective,minibatches=minis,parameter_delta_l2=summary['parameter_delta_l2'],
        normalized_parameter_drift=summary['parameter_delta_l2']/norm if norm else None,
        inactive_update=summary['parameter_delta_l2']==0,nonfinite_count=0,
        ratio_weighting='pooled response tokens; bin assignment per trajectory',
        objective_clip_definition='standard clipped branch exceeds unclipped branch; dual clipping separately reported')

def health(m,gates):
    s=m['minibatches'][1];r=s['ratio'];d=m['normalized_parameter_drift']
    failed=[]
    checks={'nonfinite':m['nonfinite_count']==0,'active_parameters':not m['inactive_update'],
        'parameter_explosion':d is not None and 0<d<=gates['max_normalized_drift'],
        'clipping_saturation':s['objective_clip_fraction']<gates['max_objective_clip'],
        'ratio_tail':r['p01']>=gates['min_p01'] and r['p99']<=gates['max_p99'] and r['min']>=gates['min_ratio'] and r['max']<=gates['max_ratio'],
        'finite_gradient':all(np.isfinite(x['grad_norm']) and x['grad_norm']>0 for x in m['minibatches'])}
    failed=[k for k,v in checks.items() if not v]
    return dict(healthy=not failed,failed=failed,active_clipping=s['objective_clip_fraction']>=gates['min_active_clip'])

def select(conditions,gates):
    good=[c for c in conditions if c['health']['healthy']]
    assert good,'BLOCKED: all preregistered diagnostic conditions failed health gates'
    active=[c for c in good if c['health']['active_clipping']]
    pool=active or good
    # The bounded tail/drift gates define "controlled/close" before measurement.
    lr=min(c['learning_rate'] for c in pool);pool=[c for c in pool if c['learning_rate']==lr]
    return min(pool,key=lambda c:abs(c['clip']-.2))
