#!/usr/bin/env python3
"""Attribute retained native outcome advantages and surrogate proxies, never gradients."""
from collections import Counter
import csv
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,now
from medical_posttrain.sampling.dynamic import classify
from medical_posttrain.reward.hybrid import reward
from frontier_diagnostic import gates

INDEX=ROOT/'experiments/stage4'
EPS=1e-6


def summarize(rows):
    result={}
    for kind in ['all','mixed','all_correct','all_wrong','mixed_parsed_wrong','mixed_unparseable_only','mixed_both']:
        rr=[r for r in rows if kind=='all' or r['classification']==kind or r['mixed_subtype']==kind]
        if not rr:
            result[kind]=dict(groups=0)
            continue
        def mean(key):return float(np.mean([r[key] for r in rr]))
        result[kind]=dict(groups=len(rr),trajectories=4*len(rr),reward_mean=mean('reward_mean'),
            within_group_reward_std_mean=mean('reward_std'),reward_min=min(r['reward_min'] for r in rr),
            reward_max=max(r['reward_max'] for r in rr),reward_std_zero_fraction=mean('reward_std_zero'),
            reward_std_positive_fraction=1-mean('reward_std_zero'),mean_abs_advantage=mean('mean_abs_advantage'),
            rms_advantage=float(np.sqrt(mean('mean_square_advantage'))),max_abs_advantage=max(r['max_abs_advantage'] for r in rr),
            nonzero_trajectory_fraction=mean('nonzero_trajectory_fraction'),nonzero_group_fraction=mean('nonzero_group'),
            all_zero_group_fraction=1-mean('nonzero_group'),abs_surrogate_mean=mean('abs_surrogate'),
            abs_backprop_scaled_surrogate_mean=mean('abs_surrogate')/16,clip_fraction=mean('clip_fraction'),
            ratio_abs_deviation_mean=mean('ratio_abs_deviation'),
            ratio_min=min(r['ratio_min'] for r in rr),ratio_max=max(r['ratio_max'] for r in rr))
        mixed=[r for r in rr if r['classification']=='mixed']
        if mixed:
            nc=sum(r['correct_count'] for r in mixed);nw=4*len(mixed)-nc
            result[kind]['correctness_discrimination']=dict(
                mean_advantage_correct=sum(r['sum_A_correct'] for r in mixed)/nc,
                mean_advantage_wrong=sum(r['sum_A_wrong'] for r in mixed)/nw,
                correct_A_positive_fraction=sum(r['correct_A_positive'] for r in mixed)/nc,
                wrong_A_negative_fraction=sum(r['wrong_A_negative'] for r in mixed)/nw,
                groups_mean_A_correct_gt_wrong_fraction=float(np.mean([r['discriminative'] for r in mixed])),
                mean_separation=float(np.mean([r['separation'] for r in mixed])))
    n=len(rows)
    return dict(by_type=result,densities=dict(
        correctness_contrast=sum(r['classification']=='mixed' for r in rows)/n,
        nonzero_advantage=sum(r['nonzero_group'] for r in rows)/n,
        correctness_discriminative=sum(r['discriminative'] for r in rows)/n))


def inspect_run(path,windows,artifact_index=None):
    from verify_stage4 import update
    cfg=read(path/'config.json');rows=[];sources=[];seen=set();seen_trajectories=set()
    for i in range(windows):
        w=path/'windows'/f'{i:04d}';commit=read(w/'commit.json')
        for ref in commit['artifacts']:
            if Path(ref['path']).name in ['selection.json','old_frozen.json','update.json']:
                assert record(ref['path'])==ref
        selection=read(w/'selection.json');groups=selection['groups']
        assert len(groups)==8
        # Reuses the unmodified native loss replay, mask/reward and FP64 error-bound checks.
        update(w/'update',groups,cfg)
        fs=[w/'selection.json',w/'commit.json',w/'update/old.npz',w/'update/old_frozen.json',
            w/'update/mini_00.npz',w/'update/mini_01.npz',w/'update/update.json']
        for f in fs:
            ref=record(f)
            if artifact_index is not None:assert ref==artifact_index[str(f)],str(f)
            sources.append(ref)
        with np.load(w/'update/old.npz') as old:
            av=old['advantages'][:,0].astype(float)
        losses=[];clips=[];ratios=[]
        for j in range(2):
            with np.load(w/'update'/f'mini_{j:02d}.npz') as mini:
                losses.extend(mini['losses']);clips.extend(mini['clips']);ratios.extend(mini['ratios'])
        decisions={d['group_id']:d for d in selection['decisions']}
        for j,g in enumerate(groups):
            assert g['group_id'] not in seen;seen.add(g['group_id'])
            scored=[]
            for r in g['responses']:
                assert r['trajectory_id'] not in seen_trajectories;seen_trajectories.add(r['trajectory_id'])
                rr=reward(r['raw_output'],r['ground_truth'],r['semantic'],''.join(r['options']),r['finish_reason'])
                assert rr['acc']==r['acc'] and rr['parser']['answer_set']==r['parsed_answer']
                assert abs(rr['score']-r['total_reward'])<1e-6
                scored.append(r)
            cls=classify(scored,selection['policy_version']);assert cls['valid']
            assert cls['classification']==decisions[g['group_id']]['classification']
            assert cls['mixed_subtype']==decisions[g['group_id']]['mixed_subtype']
            a=av[j*4:j*4+4];rw=np.asarray([r['total_reward'] for r in scored]);acc=np.asarray([r['acc'] for r in scored],dtype=bool)
            sep=float(a[acc].mean()-a[~acc].mean()) if cls['eligible'] else None
            ratio=np.asarray(ratios[j*4:j*4+4])
            rows.append(dict(window=i+1,group_id=g['group_id'],prompt_id=g['prompt_id'],classification=cls['classification'],
                mixed_subtype=cls['mixed_subtype'],correct_count=int(acc.sum()),
                reward_mean=float(rw.mean()),reward_std=float(rw.std(ddof=0)),reward_min=float(rw.min()),reward_max=float(rw.max()),
                reward_std_zero=int(rw.std(ddof=0)==0),mean_abs_advantage=float(np.abs(a).mean()),mean_square_advantage=float(np.square(a).mean()),
                max_abs_advantage=float(np.abs(a).max()),nonzero_trajectory_fraction=float((np.abs(a)>EPS).mean()),
                nonzero_group=int(np.any(np.abs(a)>EPS)),sum_A_correct=float(a[acc].sum()),sum_A_wrong=float(a[~acc].sum()),
                correct_A_positive=int((a[acc]>0).sum()),wrong_A_negative=int((a[~acc]<0).sum()),
                discriminative=int(cls['eligible'] and sep>0),separation=sep,
                abs_surrogate=float(np.abs(losses[j*4:j*4+4]).mean()),clip_fraction=float(np.mean(clips[j*4:j*4+4])),
                ratio_abs_deviation=float(np.abs(ratio-1).mean()),ratio_min=float(ratio.min()),ratio_max=float(ratio.max()),
                selection_source=str(w/'selection.json'),advantage_source=str(w/'update/old.npz')))
        if (i+1)%64==0:print(path.name,'windows',i+1,flush=True)
    assert len(rows)==len(seen)==windows*8 and len(seen_trajectories)==windows*32
    return dict(summary=summarize(rows),first512=summarize(rows[:512]),rows=rows,sources=sources,
        checks=dict(groups=len(seen),trajectories=len(seen_trajectories),native_update_replays=windows,
                    duplicate_groups=0,duplicate_trajectories=0,hash_mismatches=0))


def main():
    pair=gates()
    assert read(INDEX/'signal_density_protocol_v1.json')['epsilon']==EPS
    allruns={};checks={};refs={}
    for name,r in pair['runs'].items():
        path=Path(r['path']);manifest=read(INDEX/path.name/'artifacts_manifest.json')
        artifact_index={r['path']:r for r in manifest}
        x=inspect_run(path,625,artifact_index)
        file=INDEX/f'signal_density_{name}_groups_v1.csv'
        with file.open('x',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(x['rows'][0]),lineterminator='\n');writer.writeheader();writer.writerows(x['rows'])
        allruns[name]=dict(run_id=path.name,**x['summary'],first512=x['first512'],group_data=record(file))
        checks[name]=x['checks'];refs[name]=x['sources']
    den=allruns['vanilla']['densities']['correctness_discriminative']
    result=dict(timestamp=now(),scope='FORMAL_RETAINED_NATIVE_SIGNAL_COMPOSITION',protocol=record(INDEX/'signal_density_protocol_v1.json'),
        runs=allruns,effective_signal_amplification=allruns['dynamic']['densities']['correctness_discriminative']/den if den else None,
        per_group_gradient_norm='NOT_IDENTIFIABLE_FROM_RETAINED_ARTIFACTS',
        limits=['Advantage density and surrogate magnitude are not gradient quality or generalization.',
                'Raw native trajectory surrogate magnitude; backward scales each trajectory by1/16 within a mini.',
                'Group reward std uses ddof0; native advantage normalization uses ddof1 and epsilon1e-6.',
                'Aggregate group advantages may have bounded FP32 normalization error; replay uses original verifier tolerance.'])
    immutable(INDEX/'signal_density_analysis_v1.json',result)
    immutable(INDEX/'signal_density_verification_v1.json',dict(result='PASS',timestamp=now(),checks=checks,sources=refs,
        analysis=record(INDEX/'signal_density_analysis_v1.json'),verifier=record(Path(__file__)),
        reused_native_verifier=record(ROOT/'scripts/verify_stage4.py'),no_rollout_generated=True,no_optimizer_updates=True))
    print({n:r['densities'] for n,r in allruns.items()},flush=True)


if __name__=='__main__':main()
