#!/usr/bin/env python3
"""CPU preflight verifier; full acceptance requires actual future output/audit evidence."""
import sys,argparse,ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze,score,digest
IDX=ROOT/'experiments/stage5'

def preflight():
    from run_stage5_eval import validate_manifests
    from prepare_stage5_preflight import usage_inventory
    from medical_posttrain.evaluation.gates import stage4_gate,verify_seal
    validate_manifests();verify_seal()
    start=read(IDX/'preflight_start_snapshot_v1.json');check_ref(start['project_state'])
    try:stage4_gate()
    except PermissionError:blocked=True
    else:blocked=False
    assert blocked,'Preflight must not open the Stage4 gate'
    usage=usage_inventory();assert usage['selection1024_generations']==usage['cmexam_cmb_project_final_generations']==0
    tests=read(IDX/'preflight_test_results_v1.json');assert tests['result']=='PASS' and tests['failed']==0
    counts=read(IDX/'preflight_input_audit_v1.json');assert counts['cmexam_counts']==[6811,6809,6732] and counts['candidate_count']==20
    med=read(IDX/'cmb_medical_only_manifest_v1.json');m=read(med['primary']['path']);data=rows(m['data']['path'])
    assert med['ids']==[r['id'] for r in data if r['strata']['exam_class']!='考研政治']
    protocol=read(IDX/'open_qa_eval_protocol_v1.json');schedule=read(protocol['schedule']['path'])
    assert len(schedule['human_audit_ids'])>=82 and not protocol['human_reviews']
    assert not any(n in sys.modules for n in ['vllm','torch','transformers','peft']), 'CPU preflight unexpectedly imported model runtime'
    result=dict(result='STAGE5_PREFLIGHT_PASS',scope='PREFLIGHT_ONLY_NOT_STAGE5_EXECUTION',tests=ref(IDX/'preflight_test_results_v1.json'),
        candidate_count=20,selection_rule_hash=read(IDX/'selection_execution_plan_v1.json')['selection_rule_hash'],
        cmexam=[6811,6809,6732],cmb_primary=2000,cmb_medical_only=med['count'],open_qa_counts=protocol['counts'],
        stage4_gate_closed=True,selection1024_generations=0,project_final_generations=0,gpu_model_loads_by_stage5=0,paid_api_calls=0,
        project_state_unchanged=True,stage5_status='NOT_STARTED',stage6_status='NOT_STARTED',usage=usage,
        limitations=['GPU generation paths intentionally NOT_RUN; synthetic backend exercises durable execution','Human/judge identities unassigned; real audit mandatory later'])
    return result

def full():
    from medical_posttrain.evaluation.gates import final_gate
    from medical_posttrain.evaluation.blind import validate_judgment,human_audit,safety_aggregate,position_consistency,visible_answer,build_bundle,aggregate_blind,safety_divergences
    from analyze_stage5_eval import analyze,load_complete
    from select_stage5_checkpoint import rank_candidates
    selected=final_gate('sft');p=read(IDX/'checkpoint_selection_protocol_v1.json');root=Path(read(IDX/'selection_execution_plan_v1.json')['artifact_root'])
    for arm in ['vanilla','dynamic']:
        summaries=[]
        for c in p['candidates'][arm]:
            d=root/'selection'/c['checkpoint_id'].replace(':','_')/'selection';data=load_complete(d,p['dataset']['prompt_ids']);s=read(d/'complete.json')['summary']
            summaries.append(dict(checkpoint_id=c['checkpoint_id'],training_groups=c['training_groups'],n=s['n'],correct=s['correct_count'],unparseable=s['unparseable_count'],truncated=s['truncation_count']))
        assert rank_candidates(summaries,p['candidates'][arm])==selected['rankings'][arm]
    computed=analyze(root,write=False);assert computed==read(IDX/'project_exam_analysis_v1.json')
    # Replay raw parser and prove every retained attempt was explicitly accounted for.
    for complete in list((root/'selection').glob('*/selection/complete.json'))+list((root/'final').glob('*/*/complete.json')):
        meta=read(complete);dataset=complete.parent.name
        if dataset=='selection':gold=rows(root/'selection_inputs/items.jsonl')
        else:gold=rows(read(IDX/'manifests'/f'{dataset}.json')['data']['path'])
        gold={r['id']:r for r in gold}
        for response_ref in meta['response_refs']:
            check_ref(response_ref);response=read(response_ref['path']);d=Path(response_ref['path']).parent;receipt=read(d/'receipt.json');check_ref(receipt['raw'])
            raw=read(receipt['raw']['path']);attempts=sorted(d.glob('attempt_*.json'));assert len(attempts)==receipt['attempts']<=2
            assert response['checkpoint_id']==meta['checkpoint_id']
            expected={c['checkpoint_id']:c['adapter']['sha256'] for cs in p['candidates'].values() for c in cs}
            expected['sft']=p['fixed_sft']['adapter']['sha256']
            assert raw['adapter_sha256']==expected[response['checkpoint_id']]
            if len(attempts)>1:assert read(attempts[-1])['reason']=='explicit technical retry' and (d/'error_001.json').exists()
            if dataset in ['selection','cmexam_test_scorable','cmb_exam_clean_2000']:assert score(raw,gold[response['prompt_id']],response['checkpoint_id'],response['run_id'])==response
    schedule=read(IDX/'open_qa_blind_schedule_v1.json');public=read(root/'blind_bundle/judge_visible.json');private=read(root/'blind_bundle/private_identity_key.json')
    items=[];outputs={role:{} for role in ['sft','selected_vanilla','selected_dynamic']}
    role_ids={'sft':'sft',**{'selected_'+a:c['checkpoint_id'] for a,c in selected['selected_checkpoints'].items()}}
    for dataset in ['cmb_clin','open_qa_retention_200']:
        m=read(IDX/'manifests'/f'{dataset}.json');items.extend(rows(m['data']['path']))
        for role,cid in role_ids.items():outputs[role].update({r['prompt_id']:r for r in load_complete(root/'final'/cid.replace(':','_')/dataset,m['ids'],False)})
    expected_public,expected_private=build_bundle(items,outputs,schedule,None)
    assert public==expected_public and private==expected_private
    assert private==schedule['entries'];public_ids={r['pair_id'] for r in public}
    assert len(public_ids)==len(public) and all(set(r)=={'pair_id','question','context','A','B','dimensions','score_range'} for r in public)
    assert all('<think' not in r['A']+r['B'] for r in public)
    judgments=read(root/'blind_bundle/judgments.json');assert {j['pair_id'] for j in judgments}==public_ids and len(judgments)==len(public_ids)
    for j in judgments:
        validate_judgment(j);assert j.get('judge_is_candidate') is False
        check_ref(j['raw_judgment']);assert j['rubric_protocol_sha256']==ref(IDX/'open_qa_eval_protocol_v1.json')['sha256']
    bias=position_consistency(judgments,schedule)
    safety={};critical=set();protocol=read(IDX/'open_qa_eval_protocol_v1.json')
    for role in protocol['models']:
        evidence=read(root/'blind_bundle'/f'safety_{role}.json');safety[role]=safety_aggregate(evidence,protocol['safety']['ids']);critical.update(safety[role]['critical_ids'])
    base_ids={e['pair_id']:e['prompt_id'] for e in schedule['entries']}
    for j in judgments:
        if j.get('critical_safety_flag'):critical.add(base_ids[j['pair_id'].removesuffix('-flip')])
    audit=human_audit(read(root/'blind_bundle/human_reviews.json'),schedule['human_audit_ids'],critical);assert audit['result']=='PASS'
    assert read(ROOT/'project_state.json')['stages']['6']['status']=='NOT_STARTED'
    open_stats=aggregate_blind(judgments,schedule,items)
    safety_rows={role:read(root/'blind_bundle'/f'safety_{role}.json') for role in protocol['models']}
    regressions={a+'_to_'+b:safety_divergences(safety_rows[a],safety_rows[b]) for a,b in [('sft','selected_dynamic'),('selected_vanilla','selected_dynamic')]}
    final=dict(computed,status='VERIFIED',open_qa={k:v for k,v in open_stats.items() if k.endswith(':clinical')},retention={k:v for k,v in open_stats.items() if k.endswith(':retention')},safety=dict(summary=safety,pairwise=regressions),human_audit=audit,position_consistency=bias)
    freeze(IDX/'final_results_v1.json',final)
    return dict(result='PASS',scope='FULL',stage=5,position_consistency=bias,human_audit=audit,safety=safety,analysis=ref(IDX/'final_results_v1.json'))

if __name__=='__main__':
    a=argparse.ArgumentParser();g=a.add_mutually_exclusive_group(required=True);g.add_argument('--preflight',action='store_true');g.add_argument('--full',action='store_true');args=a.parse_args()
    result=preflight() if args.preflight else full()
    freeze(IDX/('stage5_preflight_verification_v1.json' if args.preflight else 'verification-final.json'),result)
    print(result['result'])
