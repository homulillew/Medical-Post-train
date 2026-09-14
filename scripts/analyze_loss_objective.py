#!/usr/bin/env python3
"""Retained evidence analysis for the independent single-seed objective ablation."""
from pathlib import Path
from collections import Counter
import sys,subprocess
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,now
from loss_objective_runtime import check
from loss_objective_metrics import metrics,stats,length_bin,BINS


def inspect(path,objective,gates,sft_norm):
    commits=[];windows=[];bins={b:[] for b in BINS};classes=Counter();subtypes=Counter();training_rows=[];sources=[];heavy=[]
    for i in range(64):
        w=path/'windows'/f'{i:04d}';c=read(w/'commit.json');commits.append(c)
        cfg=read(path/'config.json');m=metrics(w/'update',cfg,objective);summary=read(w/'update/update.json')
        old=np.load(w/'update/old.npz');selected=read(w/'selection.json')['groups'];rows=[r for g in selected for r in g['responses']]
        for j,mini in enumerate(m['minibatches']):
            e=np.load(w/'update'/f'mini_{j:02d}.npz')
            for k,index in enumerate(e['indices']):
                r=rows[index];mask=old['mask'][index].astype(bool);d=(e['current_logprobs'][k]-old['old_logprobs'][index])[mask].astype(np.float64);a=old['advantages'][index][mask].astype(np.float64)
                ratios=np.exp(np.clip(d,-20,20)) if objective=='grpo' else np.full(len(d),np.exp(min(d.mean(),10)))
                unclipped=-a*ratios;clipped=-a*np.clip(ratios,1-cfg['clip_ratio_low'],1+cfg['clip_ratio_high']);sur=np.maximum(unclipped,clipped)
                if objective=='grpo':sur=np.where(a<0,np.minimum(sur,-a*3),sur)
                row=dict(trajectory_id=r['trajectory_id'],prompt_id=r['prompt_id'],window=i,mini=j,length=len(d),acc=r['acc'],
                    ratio_abs_deviation=float(abs(ratios-1).mean()),lower_clip=float((ratios<1-cfg['clip_ratio_low']).mean()),upper_clip=float((ratios>1+cfg['clip_ratio_high']).mean()),
                    objective_clip=float((clipped>unclipped).mean()),absolute_surrogate=float(abs(sur).mean()),source=str(w/'selection.json'))
                bins[length_bin(len(d))].append(row);training_rows.append(row)
                if objective=='grpo' and len(d)>=384 and row['objective_clip']>=.1:heavy.append(row)
        roll=read(w/'rollout_metrics.json');classes.update(roll['group_counts']);subtypes.update(roll['mixed_subtypes'])
        normalized=summary['parameter_delta_l2']/sft_norm
        ratio=m['minibatches'][1]['ratio']
        violations=[]
        if m['nonfinite_count']:violations.append('nonfinite')
        if m['inactive_update']:violations.append('inactive')
        if normalized>gates['max_normalized_drift']:violations.append('parameter_drift')
        if not (ratio['p01']>=gates['min_p01'] and ratio['p99']<=gates['max_p99'] and ratio['min']>=gates['min_ratio'] and ratio['max']<=gates['max_ratio']):violations.append('ratio_tail')
        windows.append(dict(window=i,metrics=m,common_normalized_drift=normalized,common_stability_violations=violations,
            strict_format=roll['strict_format'],unparseable=roll['unparseable'],truncation=roll['truncation']))
        sources.extend(record(w/n) for n in ['commit.json','update/old.npz','update/mini_00.npz','update/mini_01.npz','update/update.json','selection.json','rollout_metrics.json'])
    state=commits[-1]['state_after'];assert state['training_groups']==512 and state['optimizer_steps']==128
    durations={key:sum(c['metrics'][key] for c in commits) for key in ['actor_process_seconds','sleep_seconds','wake_seconds','sync_seconds']}
    durations['generation_seconds']=sum(read(f)['generation_seconds'] for i in range(64) for f in (path/'windows'/f'{i:04d}'/'batches').glob('*/raw.json'))
    durations['reward_seconds']=sum(read(f)['seconds'] for i in range(64) for f in (path/'windows'/f'{i:04d}'/'batches').glob('*/reward_runtime.json'))
    length={}
    for b,rr in bins.items():
        length[b]=dict(n=len(rr),fraction=len(rr)/2048,accuracy=float(np.mean([r['acc'] for r in rr])) if rr else None,
            metrics={key:stats([r[key] for r in rr]) for key in ['ratio_abs_deviation','lower_clip','upper_clip','objective_clip','absolute_surrogate','length']},
            gradient_related_proxy='NOT_IDENTIFIABLE: no per-trajectory parameter gradients retained',weighting='equal trajectory; token means within each trajectory')
    minis=[x for w in windows for x in w['metrics']['minibatches']]
    return dict(path=str(path),objective=objective,training_groups=512,optimizer_steps=128,training_trajectories=2048,
        generated_groups=state['generated_groups'],amplification=state['generated_groups']/512,output_tokens=state['output_tokens'],prompt_tokens=state['prompt_tokens'],
        rollout_tokens=state['output_tokens']+state['prompt_tokens'],generated_mean_response_length=state['output_tokens']/(state['generated_groups']*4),
        group_counts=dict(classes),group_fractions={k:v/state['generated_groups'] for k,v in classes.items()},mixed_subtypes=dict(subtypes),
        overflow_groups=sum(sum(d['disposition']=='overflow_eligible' for d in read(path/'windows'/f'{i:04d}'/'selection.json')['decisions']) for i in range(64)),
        active_phase_seconds=durations,active_phase_hours=sum(durations.values())/3600,
        accounting='Phase sum excludes outer engine loading, idle pause, verification; actor_process_seconds already contains optimizer duration.',
        common_stability_violating_windows=sum(bool(w['common_stability_violations']) for w in windows),
        nonfinite_events=sum(w['metrics']['nonfinite_count'] for w in windows),inactive_windows=sum(w['metrics']['inactive_update'] for w in windows),
        grad_norm=stats([m['grad_norm'] for m in minis]),entropy=stats([m['entropy'] for m in minis]),parameter_delta=stats([w['metrics']['parameter_delta_l2'] for w in windows]),
        common_drift_normalizer='original SFT trainable parameter L2',length_sensitivity=length,windows=windows,training_rows=training_rows,heavy_clipping_candidates=sorted(heavy,key=lambda r:-r['objective_clip'])[:3],sources=sources)

def interpret(e,runs):
    a=e['paired']['vanilla_gspo_minus_grpo']['bootstrap']['ci95'];b=e['paired']['dynamic_gspo_minus_grpo']['bootstrap']['ci95']
    if a[0]>0 and b[0]>0:return 'GSPO_OBJECTIVE_ADVANTAGE_SUPPORTED'
    if a[1]<0 and b[1]<0:return 'GRPO_ADVANTAGE'
    if a[0]<=0<=a[1] and b[0]<=0<=b[1]:
        if all(runs[s+'_gspo']['common_stability_violating_windows']<runs[s+'_grpo']['common_stability_violating_windows'] for s in ['vanilla','dynamic']):return 'GSPO_STABILITY_ONLY'
        return 'NO_CLEAR_OBJECTIVE_DIFFERENCE'
    return 'INCONCLUSIVE'

def main():
    p=check();idx=ROOT/'experiments/stage4';e=read(idx/'loss_objective_eval_analysis_v1.json')
    for arm in ['vanilla','dynamic']:assert read(idx/f'grpo_{arm}512_verification_v1.json')['result']=='PASS'
    assert read(idx/'loss_objective_eval_verification_v1.json')['result']=='PASS'
    pair=read(p['formal_pair']['path']);paths={n+'_gspo':Path(r['path']) for n,r in pair['runs'].items()};paths.update({n+'_grpo':Path(r['path']) for n,r in p['runs'].items()})
    pre=read(p['preregistration']['path']);diag=read(p['diagnostic']['path']);first=next(c for c in diag['conditions'] if c.get('native_replay')=='PASS')
    sft_norm=read(Path(first['directory'])/'update/update.json')['initial_parameter_l2']
    runs={name:inspect(path,name.rsplit('_',1)[1],pre['health_gates'],sft_norm) for name,path in paths.items()}
    result=interpret(e,runs)
    grpo=e['paired']['dynamic_minus_vanilla_grpo']['bootstrap'];gspo=e['paired']['dynamic_minus_vanilla_gspo']['bootstrap']
    cross=('POSITIVE_POINT_ESTIMATE_BOTH_OBJECTIVES' if grpo['delta']>0 and gspo['delta']>0 else 'POSITIVE_ONLY_GSPO' if gspo['delta']>0 else 'POSITIVE_ONLY_GRPO' if grpo['delta']>0 else 'NO_POSITIVE_DYNAMIC_POINT_ESTIMATE')
    analysis=dict(timestamp=now(),scope='AUXILIARY_LOSS_OBJECTIVE_ABLATION_SINGLE_SEED_VALIDATION',evaluation=record(idx/'loss_objective_eval_analysis_v1.json'),
        runs=runs,scientific_interpretation=result,dynamic_cross_objective_interpretation=cross,
        interpretation_rules=p['interpretation_rules'],optimization_package_caveat=read(p['native_audit']['path'])['objective_package_differences'],
        length_sensitivity_caveat='On-policy trajectory lengths/content differ across arms. Bin association is descriptive, not a causal length effect. Gradient quality is NOT_IDENTIFIABLE.',
        negative_results='All paired directions and intervals retained, including regressions and nulls; no equivalence inference from intervals crossing0.',
        selection1024_used=False,final_tests_used=False,primary_conclusions_redefined=False)
    immutable(idx/'loss_objective_ablation_analysis_v1.json',analysis)
    manifest=read(p['evaluation_manifest']['path']);dataset={r['prompt_id']:r for r in read(manifest['dataset']['path'])};pred={};sources={}
    directory=Path(p['evaluation_artifact_path'])/'loss_objective_eval'
    for name in p['evaluation_order']:
        pred[name]={}
        for f in sorted((directory/name).glob('batch_*.json')):
            if 'reservation' in f.name:continue
            for r in read(f)['predictions']:pred[name][r['prompt_id']]=r;sources[name,r['prompt_id']]=record(f)
    cases=[];availability={}
    predicates={
        'grpo_wrong_gspo_correct':lambda vr,vg,dr,dg: not vr['acc'] and vg['acc'],
        'grpo_correct_gspo_wrong':lambda vr,vg,dr,dg: vr['acc'] and not vg['acc'],
        'both_correct_different_reasoning_candidate':lambda vr,vg,dr,dg:vr['acc'] and vg['acc'] and vr['output']!=vg['output'],
        'both_wrong_same_misconception_candidate':lambda vr,vg,dr,dg:not vr['acc'] and not vg['acc'] and vr['parsed_answer'] is not None and vr['parsed_answer']==vg['parsed_answer'],
        'dynamic_works_both_objectives':lambda vr,vg,dr,dg:not vr['acc'] and not vg['acc'] and dr['acc'] and dg['acc'],
        'dynamic_works_only_one_objective':lambda vr,vg,dr,dg:(not vr['acc'] and dr['acc']) != (not vg['acc'] and dg['acc'])}
    for category,predicate in predicates.items():
        ids=[pid for pid in manifest['prompt_ids'] if predicate(*[pred[n][pid] for n in ['vanilla_grpo','vanilla_gspo','dynamic_grpo','dynamic_gspo']])]
        availability[category]=len(ids)
        for pid in ids[:2]:
            cases.append(dict(category=category,prompt_id=pid,prompt=dataset[pid],responses={n:pred[n][pid] for n in pred},sources=[sources[n,pid] for n in pred],
                human_reviewed=False,read_in_full=False,clinical_reasoning_confirmed=False,
                machine_rule='Answer/output string predicates only. Different text does not establish materially different reasoning; same wrong option does not establish shared misconception. Requires human review.'))
    for name,r in runs.items():
        if not name.endswith('_grpo'):continue
        availability[name+'_long_heavy_clipping']=len(r['heavy_clipping_candidates'])
        for row in r['heavy_clipping_candidates']:
            selected=read(row['source']);response=next(t for g in selected['groups'] for t in g['responses'] if t['trajectory_id']==row['trajectory_id'])
            cases.append(dict(category='long_response_heavy_clipping_grpo',arm=name,metrics=row,response=response,source=record(row['source']),human_reviewed=False,clinical_reasoning_confirmed=False))
    immutable(idx/'loss_objective_case_candidates_v1.json',dict(timestamp=now(),status='MACHINE_CANDIDATES_ONLY',availability=availability,cases=cases,manual_reviews=[],missing_category_policy='Zero matches retained as missing, never fabricated'))
    assert record(ROOT/'project_state.json')==p['project_state'];assert record(p['selection_protocol']['path'])==p['selection_protocol']
    checks=dict(result='PASS',scope='AUXILIARY_LOSS_OBJECTIVE_ONLY',timestamp=now(),
        checks=['six fixed-batch native optimization diagnostics','diagnostic-only hyperparameter selection','protocol committed before auxiliary launch',
        'isolated512 validation','both512groups64windows128steps','two physical resumes','two full raw native verifiers','five greedy endpoints','paired10000seed20260914',
        'four fixed length bins and gradient proxy limitation','real Dynamic refill counts','machine-only cases','unchanged Stage4 FULL_PASS and Stage5 NOT_STARTED'],
        analysis=record(idx/'loss_objective_ablation_analysis_v1.json'),cases=record(idx/'loss_objective_case_candidates_v1.json'),
        training_verifiers={a:record(idx/f'grpo_{a}512_verification_v1.json') for a in ['vanilla','dynamic']},evaluation_verifier=record(idx/'loss_objective_eval_verification_v1.json'),
        selection1024_untouched=True,final_tests_untouched=True,READY_FOR_STAGE5='NO',full_stage4_verifier_run=False)
    # Validate actual commit ancestry and launch timestamps, not only protocol prose.
    protocol_commit=subprocess.check_output(['git','log','-1','--format=%H','--','experiments/stage4/grpo_objective_protocol_v1.json'],cwd=ROOT,text=True).strip()
    for r in p['runs'].values():
        launch=read(Path(r['path'])/'attempt_001/launch.json')
        subprocess.run(['git','merge-base','--is-ancestor',protocol_commit,launch['git_commit']],cwd=ROOT,check=True)
        assert launch['protocol']==record(idx/'grpo_objective_protocol_v1.json') and launch['timestamp']>p['timestamp']
    immutable(idx/'loss_objective_ablation_verification_v1.json',checks)
    handoff=dict(timestamp=now(),scope=analysis['scope'],protocol=record(idx/'grpo_objective_protocol_v1.json'),diagnostic=p['diagnostic'],
        analysis=checks['analysis'],verification=record(idx/'loss_objective_ablation_verification_v1.json'),cases=checks['cases'],
        evaluation=e,scientific_interpretation=result,dynamic_cross_objective_interpretation=cross,
        Stage4='FULL_PASS',Stage5='NOT_STARTED',READY_FOR_STAGE5='NO',human_reviews_confirmed=False,
        selection1024_untouched=True,final_tests_untouched=True,
        next_recommended_action='User performs genuine manual case review, then authorize final Stage4 acceptance; keep Stage5 gate closed until then.')
    immutable(ROOT/'experiments/handoffs/loss_objective_ablation_to_chatgpt_v1.json',handoff)
    # Append independent factual section; preserve formal conclusions and human-review status.
    table='\n'.join(f"| {n} | {v['correct']}/{v['n']} | {100*v['accuracy']:.4f}% |" for n,v in e['summary'].items())
    pairs='\n'.join(f"- `{n}`: {100*v['bootstrap']['delta']:+.4f} pp，95% CI [{100*v['bootstrap']['ci95'][0]:+.4f}, {100*v['bootstrap']['ci95'][1]:+.4f}] pp；gain/regress={v['wrong_to_correct']}/{v['correct_to_wrong']}，exact McNemar p={v['exact_mcnemar']:.6g}。" for n,v in e['paired'].items())
    text=f'''\n\n## GRPO vs GSPO Auxiliary Objective Ablation\n\n独立单 seed、512-group auxiliary validation 对照已完成。原有 5000-group GSPO formal pair 及其结论保持冻结。本节补充此前缺少的 token-level objective 对照；不是 primary formal GRPO 实验。\n\nGRPO 使用 verl `compute_policy_loss_vanilla`，原生 token ratio 与 dual-clip=3，和 GSPO 同用 native GRPO group-relative advantage、`seq-mean-token-mean` 聚合。六组原始固定批次诊断后按预注册规则选择 LR={p['selected']['learning_rate']:g}、symmetric clip={p['selected']['clip']:g}，没有使用新 validation 调参。原生 clamp、dual-clip 和 clip 尺度均属于 objective-package 差异，不能把全部差异归因于 ratio geometry。\n\n| Model | Correct/N | Accuracy |\n|---|---:|---:|\n{table}\n\n{pairs}\n\n描述性 interaction={e['interaction']['delta_pp']:+.4f} pp，95% paired bootstrap CI={e['interaction']['ci95_pp']} pp。预注册判定为 `{result}`；Dynamic 跨 objective 的结果为 `{cross}`。这是单 seed exploratory validation，非 final test，CI 跨零不代表等效。\n\nDynamic-GRPO 实际生成 {runs['dynamic_grpo']['generated_groups']} 组，amplification={runs['dynamic_grpo']['amplification']:.4f}，rollout tokens={runs['dynamic_grpo']['rollout_tokens']}；Dynamic-GSPO 对应 {runs['dynamic_gspo']['generated_groups']} 组、{runs['dynamic_gspo']['amplification']:.4f}、{runs['dynamic_gspo']['rollout_tokens']} tokens。生成量未人为匹配。\n\n四个固定长度区间的 ratio、clipping、absolute surrogate、accuracy 和样本分布见 [raw analysis](../../experiments/stage4/loss_objective_ablation_analysis_v1.json)。Per-length parameter-gradient proxy 为 `NOT_IDENTIFIABLE`；on-policy 长度关联不能解释为长度的因果效应。所有反向结果与无显著差异的比较均保留。\n\n两条 GRPO 均完成真实 32→40 groups / 8→10 steps fresh-process resume 与完整原生 raw verifier。五模型统一评估通过。案例仅为机器候选，未伪造人工审阅。Stage4 仍 `FULL_PASS`，Stage5 仍 `NOT_STARTED`，`READY_FOR_STAGE5=NO`；selection1024 与 final tests 未使用，未执行 full Stage4 verifier。证据入口：[handoff](../../experiments/handoffs/loss_objective_ablation_to_chatgpt_v1.json)。\n'''
    for name in ['04_gspo.md','04_gspo_interview_story.md']:
        file=ROOT/'docs/stage_reports'/name;assert '## GRPO vs GSPO Auxiliary Objective Ablation' not in file.read_text()
        file.write_text(file.read_text()+text)
    # A new revision receipt preserves the old immutable documentation draft hashes.
    immutable(idx/'loss_objective_documentation_addendum_v1.json',dict(timestamp=now(),reason='Authorized independent objective ablation completed; previous no-ablation scope now supplemented, formal conclusions preserved.',
        documents=[record(ROOT/'docs/stage_reports'/n) for n in ['04_gspo.md','04_gspo_interview_story.md']],old_deliverables_draft=record(idx/'deliverables_draft_v1.json'),
        old_draft_references_now_historical=True,manual_reviews_confirmed=False,READY_FOR_STAGE5='NO'))
    durable(idx/'loss_objective_status_v1.json',dict(status='AUXILIARY_VERIFIED_AND_ANALYZED',timestamp=now(),scientific_interpretation=result,READY_FOR_STAGE5='NO'))
    assert subprocess.run(['git','diff','--cached','--quiet'],cwd=ROOT).returncode==0
    files=list(idx.glob('loss_objective*'))+list(idx.glob('grpo_*'))+[ROOT/'experiments/handoffs/loss_objective_ablation_to_chatgpt_v1.json']
    for r in p['runs'].values():files+=list((idx/r['run_id']).glob('*.json'))
    files += [ROOT/'docs/stage_reports'/n for n in ['04_gspo.md','04_gspo_interview_story.md']]
    subprocess.run(['git','add','--',*[str(f.relative_to(ROOT)) for f in files if f.is_file()]],cwd=ROOT,check=True)
    subprocess.run(['git','diff','--cached','--check'],cwd=ROOT,check=True)
    subprocess.run(['git','commit','-m','handoff: retain grpo versus gspo auxiliary objective evidence'],cwd=ROOT,check=True)
    subprocess.run(['git','push','origin','main'],cwd=ROOT,check=True)
if __name__=='__main__':main()
