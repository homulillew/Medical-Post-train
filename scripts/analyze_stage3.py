#!/usr/bin/env python3
"""Freeze descriptive summaries and full representative groups after real refill."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.evidence import write_json
from medical_posttrain.evidence.stage2 import read, record
from medical_posttrain.evidence.stage3 import INDEX
from medical_posttrain.sampling.analysis import summarize


def main():
    selected=read(INDEX/'selected_runs.json')
    paths={k:Path(read(INDEX/v/'manifest.json')['artifact_root']) for k,v in selected.items()}
    formal=paths['formal'];summary=summarize(formal)
    assert read(formal/'status.json')['status']=='FULL_PASS' and summary['accepted_mixed_groups']==256
    groups=[g for b in sorted((formal/'batches').glob('*')) for g in read(b/'scored.json')['groups']]
    stats={g['group_id']:g for g in summary['groups']}
    byid={g['group_id']:g for g in groups}
    tests={
        'all_wrong':lambda s,g:s['classification']=='all_wrong',
        'all_correct':lambda s,g:s['classification']=='all_correct',
        'one_of_four':lambda s,g:s['correct_count']==1,
        'two_of_four':lambda s,g:s['correct_count']==2,
        'three_of_four':lambda s,g:s['correct_count']==3,
        'mixed_unparseable_only':lambda s,g:s['mixed_subtype']=='mixed_unparseable_only',
        'mixed_parsed_wrong':lambda s,g:s['mixed_subtype'] in ('mixed_parsed_wrong','mixed_both'),
        'high_semantic_all_wrong':lambda s,g:s['classification']=='all_wrong' and any(r['semantic']>.9 for r in g['responses']),
        'hybrid_variance_all_correct':lambda s,g:s['classification']=='all_correct' and s['hybrid_reward_std']>0,
        'long_accepted':lambda s,g:s['disposition']=='accepted',
        'long_rejected':lambda s,g:s['disposition'].startswith('rejected'),
        'short_accepted':lambda s,g:s['disposition']=='accepted',
        'multi_all_wrong':lambda s,g:s['answer_cardinality']>1 and s['classification']=='all_wrong',
        'multi_mixed':lambda s,g:s['answer_cardinality']>1 and s['classification']=='mixed',
        'mixed_both':lambda s,g:s['mixed_subtype']=='mixed_both',
        'overflow_eligible':lambda s,g:s['disposition']=='overflow_eligible',
        'truncation':lambda s,g:any(r['finish_reason']=='length' for r in g['responses']),
    }
    cases=[];categories={}
    for category,predicate in tests.items():
        candidates=[g for g in groups if predicate(stats[g['group_id']],g)]
        if not candidates:
            categories[category]=dict(status='NOT_OBSERVED',case_id=None);continue
        if category.startswith('long_'):chosen=max(candidates,key=lambda g:sum(r['output_tokens'] for r in g['responses']))
        elif category=='short_accepted':chosen=min(candidates,key=lambda g:sum(r['output_tokens'] for r in g['responses']))
        elif category=='high_semantic_all_wrong':chosen=max(candidates,key=lambda g:max(r['semantic'] for r in g['responses']))
        else:chosen=candidates[0]
        case_id=f's3_{category}_{chosen["encounter_index"]:04d}'
        categories[category]=dict(status='OBSERVED',case_id=case_id)
        cases.append(dict(case_id=case_id,run_id=formal.name,stage=3,category='SAMPLING_CASE',subtype=category,
            prompt_id=chosen['prompt_id'],group=chosen,statistics=stats[chosen['group_id']],
            observation=f"Accuracy vector {stats[chosen['group_id']]['acc_vector']}; disposition {stats[chosen['group_id']]['disposition']}; full raw responses retained.",
            hypothesis=None,alternative_explanations=['Parser failure may contribute accuracy=0; embedding similarity is not clinical correctness.'],
            followup='Monitor this slice per future policy window; no prompt blacklist or reward changes.',clinical_validation=False))
    categories['repeated_prompt_category_change']=dict(status='NOT_OBSERVED',case_id=None,reason='Formal stream did not revisit a prompt; cyclic support tested deterministically')
    categories['policy_updated_category_change']=dict(status='NOT_OBSERVED',case_id=None,reason='Fixed SFT policy and zero optimizer updates')
    categories['formal_starvation']=dict(status='NOT_OBSERVED',case_id=None,reason='Target256 was reached; bounded starvation branch covered by unit test')
    write_json(INDEX/'case_coverage.json',dict(run_id=formal.name,categories=categories,cases=cases,selection='Descriptive category mining, not representative prevalence or clinical validation'))
    review_ids=list(dict.fromkeys(c['group']['group_id'] for c in cases))
    for g in groups:
        if len(review_ids)>=14:break
        if g['group_id'] not in review_ids:review_ids.append(g['group_id'])
    if (INDEX/'manual_review_worklog.json').exists():
        review_ids=list(dict.fromkeys(e['group_id'] for e in read(INDEX/'manual_review_worklog.json')['entries']))
    write_json(INDEX/'manual_review_selection.json',dict(group_ids=review_ids,trajectory_count=len(review_ids)*4,selection='Manually read committed prefix0..15 plus groups16,20,45 while formal continued; covers all correctness buckets, mixed subtypes and multi-select; no model/experiment selection'))
    packet=[]
    for gid in review_ids:
        g=byid[gid];s=stats[gid]
        packet.append(f"GROUP {g['encounter_index']} {gid}\n{s}\nQUESTION: {g['responses'][0]['question']}\nOPTIONS: {g['responses'][0]['options']}\nGROUND TRUTH: {g['responses'][0]['ground_truth']}\n")
        for r in g['responses']:
            packet.append(f"TRAJECTORY {r['member_index']}: parsed={r['parsed_answer']} acc={r['acc']} sem={r['semantic']:.6f} format={r['format']} R={r['total_reward']:.6f} error={r['parse_error']}\n{r['raw_output']}\nEND TRAJECTORY\n")
    (INDEX/'manual_review_packet.txt').write_text('\n'.join(packet))
    write_json(INDEX/'groups.json',summary['groups'])
    write_json(INDEX/'cost_summary.json',summary['costs'])
    smoke=read(paths['smoke']/'summary.json')
    write_json(INDEX/'stream_overlap.json',dict(stage2_formal_prompt_overlap=summary['exposure']['stage2_formal_overlap'],
        smoke_formal_prompt_overlap=len(set(smoke['exposure']['prompt_exposure_count'])&set(summary['exposure']['prompt_exposure_count'])),
        scope='Prompt identity overlap only; separate run IDs, domain, encounter seeds and fresh generation; no smoke groups counted toward256'))
    failure_costs={}
    for name,path in paths.items():
        if not name.startswith('failed_'):continue
        recovered=INDEX/path.name/'recovered_cost_summary.json'
        failure_costs[name]=read(recovered)['costs'] if recovered.exists() else read(path/'observations.json')
    tokens=5000*summary['sampling_amplification']*4*summary['lengths']['mean']
    write_json(INDEX/'compute_calibration.json',dict(source=record(formal/'summary.json'),stage3_formal_costs=summary['costs'],stage3_smoke_costs=smoke['costs'],stage3_failed_run_costs=failure_costs,
        total_recorded_rollout_output_tokens=summary['costs']['generated_output_tokens']+smoke['costs']['generated_output_tokens']+sum(v['generated_output_tokens'] for v in failure_costs.values()),
        failed_lifecycle_time_scope='Failed-summary worker retained GPU during later failed-startup attempt; do not sum overlapping wall intervals as GPU hours. Active runtime checkpoints omit human pause/cleanup gaps; startup failure launch-to-error wall separately retained.',
        measured_amplification=summary['sampling_amplification'],stage4_dynamic_output_tokens=tokens,
        stage4_dynamic_generation_hours=tokens/summary['costs']['output_tokens_per_second']/3600,
        stage4_vanilla_contract_groups=5000,stage4_dynamic_contract_accepted_groups=5000,
        formula='5000 * measured_amplification * 4 * mean_output_tokens / measured_generation_tokens_per_second',
        limitations='Fixed SFT initial projection only; changing policy acceptance, actor updates, old-logprob, switching, validation and checkpoint costs remain unmeasured.'))
    print({k:summary[k] for k in ('generated_groups','accepted_mixed_groups','group_counts','sampling_amplification')})

if __name__=='__main__':main()
