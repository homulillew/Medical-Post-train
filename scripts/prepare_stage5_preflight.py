#!/usr/bin/env python3
"""CPU-only source-metadata freeze; no generation or model-library imports."""
import sys,subprocess,datetime
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze,digest
from medical_posttrain.evaluation.gates import gpu_snapshot
from medical_posttrain.evaluation.blind import make_schedule,DIMENSIONS
from medical_posttrain.reward.parser import canonical_answer
IDX=ROOT/'experiments/stage5'
OUT=Path('/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1')
NAMES=['cmexam_test_full','cmexam_test_scorable','cmexam_test_clean','cmb_exam_clean_2000','cmb_clin','open_qa_retention_200','safety_slice']

def usage_inventory():
    local=[];external=[]
    bulk=ROOT.parent/'medical-post-train-artifacts/runs'
    for p in sorted(bulk.glob('s5*')):
        for f in p.glob('manifest.json'):
            m=read(f)
            if m.get('config'):
                check_ref(m['config']);cfg=read(m['config']['path'])
                if cfg.get('model') in ['deepseek-flash','deepseek-v4-pro'] and str(cfg.get('endpoint','')).startswith('https://api.deepseek.com'):
                    external.append(dict(manifest=ref(f),config=m['config'],model=cfg['model'],reason='Verified historical external provider, not a local project checkpoint'));continue
            local.append(str(f))
    generations=list(OUT.glob('**/response.json')) if OUT.exists() else []
    legacy=list((IDX/'project_models').glob('**/*')) if (IDX/'project_models').exists() else []
    selected=[IDX/'selection_result_manifest_v1.json',IDX/'selection_result_lock_v1.json']
    return dict(external_records_excluded=external,project_run_manifests=local,project_response_files=[str(p) for p in generations],legacy_project_files=[str(p) for p in legacy],
        selection_result_files=[str(p) for p in selected if p.exists()],selection1024_generations=0 if not local+generations+legacy else None,
        cmexam_cmb_project_final_generations=0 if not local+generations+legacy else None,
        evidence_scope='Repository declared project-model run namespaces and project_state; historical external API baselines are separate and excluded')

def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip();remote=subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip();assert head==remote
    state=read(ROOT/'project_state.json');assert [state['stages'][str(i)]['status'] for i in range(1,7)]==['DONE','DONE','DONE','FULL_PASS','NOT_STARTED','NOT_STARTED']
    aux=read(ROOT/'experiments/stage4/loss_objective_status_v1.json');assert aux['status'] in ['TRAINING','TRAINING_INTEGRATION','RAW_VERIFYING','ABLATION_EVALUATING']
    usage=usage_inventory();assert usage['selection1024_generations']==usage['cmexam_cmb_project_final_generations']==0 and not usage['selection_result_files']
    before=dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),head=head,project_state=ref(ROOT/'project_state.json'),state=state,
        auxiliary_snapshot=aux,gpu=gpu_snapshot(),usage=usage,task='STAGE5_CPU_PREFLIGHT',gpu_model_loads_by_this_task=0,paid_api_calls_by_this_task=0)
    freeze(IDX/'preflight_start_snapshot_v1.json',before)
    p=read(IDX/'checkpoint_selection_protocol_v1.json');check_ref(p['execution_gate']['enforcement_source']);check_ref(p['dataset']['partition'])
    assert p['candidate_training_groups']==[512,1024,1536,2048,2560,3072,3584,4096,4608,5000]
    candidates=[]
    for arm in ['vanilla','dynamic']:
        assert [c['training_groups'] for c in p['candidates'][arm]]==p['candidate_training_groups']
        for c in p['candidates'][arm]:
            check_ref(c['adapter']);adapter=Path(c['adapter']['path']).parent;checkpoint=adapter.parent;w=checkpoint.parent;commit=read(w/'commit.json')
            assert commit['state_after']['training_groups']==c['training_groups'] and commit['state_after']['optimizer_steps']==c['optimizer_steps']
            assert commit['state_after']['policy_windows']==c['policy_windows'] and int(w.name)==c['policy_windows']-1
            candidates.append(dict(**c,variant=arm,checkpoint_path=str(checkpoint),adapter_path=str(adapter),policy_window_index=int(w.name),source_commit=ref(w/'commit.json')))
    freeze(IDX/'checkpoint_candidate_index_v1.json',dict(status='FROZEN_INPUTS_ONLY',selection_protocol=ref(IDX/'checkpoint_selection_protocol_v1.json'),candidate_count=20,candidates=candidates))
    schema=dict(type='object',required=['checkpoint_id','run_id','prompt_id','raw_output','finish_reason','parsed_answer','ground_truth','correct','strict_format','parse_method','ambiguous','unparseable','truncated','prompt_tokens','output_tokens'],
        properties={k:{'type':'boolean'} for k in ['correct','strict_format','ambiguous','unparseable','truncated']},
        summary_fields=['n','correct_count','accuracy','strict_format_rate','unparseable_rate','truncation_rate','mean_output_tokens','P50_output_tokens','P95_output_tokens','P99_output_tokens'])
    for k in ['checkpoint_id','run_id','prompt_id','raw_output','ground_truth','parse_method']:schema['properties'][k]={'type':'string'}
    schema['properties'].update(finish_reason={'enum':['stop','length']},parsed_answer={'type':['string','null']},prompt_tokens={'type':'integer','minimum':0},output_tokens={'type':'integer','minimum':0})
    freeze(IDX/'selection_response_schema_v1.json',schema)
    freeze(IDX/'selection_execution_plan_v1.json',dict(status='NOT_RUN',checkpoint_count=20,prompts_per_checkpoint=1024,total_expected_generations=20480,
        checkpoint_order=[c['checkpoint_id'] for c in candidates],prompt_order=p['dataset']['prompt_ids'],selection_rule=p['selection_rule'],selection_rule_hash=digest(p['selection_rule']),
        selection_protocol=ref(IDX/'checkpoint_selection_protocol_v1.json'),decoding=p['decoding'],
        artifact_root=str(OUT),selection_directory=str(OUT/'selection'),failure_semantics='Partial candidates cannot rank. Preserve all reservations, errors and costs.',
        retry_semantics='Zero automatic retries, as existing frozen protocol. Explicit technical resume can retry only unresolved transport attempts; never completed wrong/unparseable output.',
        gpu_exclusivity='Stage4 gate plus no CUDA owners or live training worker/supervisor plus exclusive stage4-gpu.lock',
        forbidden_ranking_inputs=['length','semantic','test','CMB','Frontier','GRPO_auxiliary','loss_objective_eval512']))
    manifests={n:read(IDX/'manifests'/f'{n}.json') for n in NAMES};sets={}
    protected=[ref(ROOT/'project_state.json'),ref(IDX/'checkpoint_selection_protocol_v1.json')]
    for n,m in manifests.items():
        check_ref(m['data']);check_ref(m['requests']);rs=rows(m['data']['path']);assert len(rs)==m['count'] and [r['id'] for r in rs]==m['ids'];sets[n]=rs
        protected.extend([ref(IDX/'manifests'/f'{n}.json'),m['data'],m['requests']])
    assert [len(sets[n]) for n in NAMES]==[6811,6809,6732,2000,208,200,111]
    full=sets['cmexam_test_full'];score=sets['cmexam_test_scorable'];clean=sets['cmexam_test_clean']
    bad=[r for r in full if not r['scorable']];assert len(bad)==2
    assert {r['id'] for r in clean} <= {r['id'] for r in score} <= {r['id'] for r in full}
    freeze(IDX/'cmexam_final_eval_protocol_v1.json',dict(status='FROZEN_NOT_RUN',official_source_n=6811,scorable_n=6809,clean_secondary_n=6732,
        headline='CMExam scorable6809; clean6732 is a secondary decontaminated subset, never full test',
        manifests={n:ref(IDX/'manifests'/f'{n}.json') for n in NAMES[:3]},malformed_source_evidence=[{k:r[k] for k in ['id','schema_error','raw_options','answer','source_file','source_record_sha256']} for r in bad],
        anomaly_source=ref(IDX/'source_anomalies.json'),decoding=p['decoding'],request_template=ref(IDX/'evaluation_protocol.json'),
        reporting_sections={'Primary Selected Models':['sft','selected_vanilla','selected_dynamic'],'Scientific Equal-Update Endpoints':['vanilla:5000','dynamic:5000']},
        unique_checkpoint_reuse='If selected equals5000, reuse exact retained predictions but report in separate sections; no second generation.',
        comparisons=['vanilla_minus_sft','dynamic_minus_sft','dynamic_minus_vanilla'],bootstrap=dict(seed=20260914,resamples=10000,unit='paired prompt'),no_test_before_frozen_selection=True))
    difficulty={str(i):[] for i in range(1,6)};categories={}
    for r in score:
        level=str(r['strata']['Difficulty level']);assert level in difficulty;difficulty[level].append(r['id'])
        for key,val in r['strata'].items():
            if key!='Difficulty level':categories.setdefault(key,{}).setdefault(str(val),[]).append(r['id'])
    slices={'easy':difficulty['1']+difficulty['2'],'medium':difficulty['3'],'hard':difficulty['4']+difficulty['5']}
    freeze(IDX/'cmexam_slice_manifest_v1.json',dict(status='FROZEN_BEFORE_PROJECT_OUTPUT',source=ref(IDX/'manifests/cmexam_test_scorable.json'),
        raw_difficulty_distribution={k:len(v) for k,v in difficulty.items()},raw_difficulty_ids=difficulty,difficulty=slices,categories=categories,rule='Source Difficulty1-2 easy,3 medium,4-5 hard; source category labels unchanged; no outcome-driven rebucketing'))
    cmb=sets['cmb_exam_clean_2000'];classes=Counter(r['strata']['exam_class'] for r in cmb)
    for r in score+cmb:assert r['scorable'] and canonical_answer(r['answer'],''.join(r['options']))==r['answer']
    excluded=[r for r in cmb if r['strata']['exam_class']=='考研政治'];medical=[r for r in cmb if r['strata']['exam_class']!='考研政治']
    freeze(IDX/'cmb_primary_audit_v1.json',dict(status='FROZEN_NOT_RUN',primary_count=2000,primary=ref(IDX/'manifests/cmb_exam_clean_2000.json'),exam_class_counts=dict(classes),
        answer_schema=dict(Counter('multiple' if len(r['answer'])>1 else 'single' for r in cmb)),invalid_primary_items=[],source_anomalies=ref(IDX/'source_anomalies.json'),
        category_membership={c:[r['id'] for r in cmb if r['strata']['exam_class']==c] for c in classes},decoding=p['decoding'],
        reporting_sections=['Primary Selected Models','Scientific Equal-Update Endpoints'],no_test_before_frozen_selection=True))
    freeze(IDX/'cmb_medical_only_manifest_v1.json',dict(name='cmb_medical_only_secondary_v1',status='FROZEN_BEFORE_PROJECT_OUTPUT',primary=ref(IDX/'manifests/cmb_exam_clean_2000.json'),
        count=len(medical),ids=[r['id'] for r in medical],ids_sha256=digest([r['id'] for r in medical]),
        excluded_categories=[dict(category='考研政治',count=len(excluded),reason='Source taxonomy explicitly identifies politics rather than medical content')],
        excluded_ids=[r['id'] for r in excluded],included_categories=sorted(set(classes)-{'考研政治'}),
        exclusion_rule="exam_class == 考研政治 only; primary2000 unchanged",project_model_outputs_consulted=False))
    items=sets['cmb_clin']+sets['open_qa_retention_200'];assert len({r['id'] for r in items})==408
    assert len({r['case_id'] for r in sets['cmb_clin']})==74
    assert {r['id'] for r in sets['safety_slice']} <= {r['id'] for r in items}
    pairs=[['selected_dynamic','selected_vanilla'],['selected_dynamic','sft'],['selected_vanilla','sft']]
    schedule=make_schedule(items,pairs);freeze(IDX/'open_qa_blind_schedule_v1.json',schedule)
    freeze(IDX/'open_qa_eval_protocol_v1.json',dict(status='FROZEN_NOT_RUN',models=['sft','selected_vanilla','selected_dynamic'],comparisons=pairs,
        manifests={n:ref(IDX/'manifests'/f'{n}.json') for n in ['cmb_clin','open_qa_retention_200','safety_slice']},
        counts=dict(clinical_cases=74,clinical_questions=208,retention=200,risk=111,unique_generation_items=408,expected_generations=1224),
        decoding=dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=2048,seed=20260914),max_model_len=8192,
        candidate_inputs='Frozen system/user request only, no reference answers; context plus current clinical question.',
        judge_visible='Only question/context and user-visible final answers A/B; strip hidden<think>, empty if malformed/unclosed. No model IDs, source IDs, reference CoT or token metrics.',
        schedule=ref(IDX/'open_qa_blind_schedule_v1.json'),rubric={d:dict(minimum=0,maximum=4,anchors={'0':'serious failure','1':'major deficiencies','2':'mixed or incomplete','3':'mostly correct/useful','4':'fully meets dimension'}) for d in DIMENSIONS},
        overall_preference=['A win','tie','B win'],judge_requirements=['exact judge_id/version and raw judgments retained','candidate must not judge itself','provider distinct when practical','no API calls in preflight'],
        position_bias=dict(seed=20260914,fraction=.1,selection='source-stratified SHA256 IDs frozen before outputs',metric='preference invariant after mapping AB/BA back to model identity'),
        human_audit=dict(minimum_fraction=.2,required_ids=schedule['human_audit_ids'],include_all_critical_flags=True,reviewer_id=None,reviewer_qualification=None,
            non_clinician_disclaimer='NON_CLINICIAN_REVIEW_NOT_CLINICAL_VALIDATION',disagreement='Retain original judgments and both observations; unresolved disputes block completion; document adjudicator and resolution'),
        clinical_statistics='Case-cluster resampling over74 cases; question and case counts both retained. Retention unit=prompt.',
        safety=dict(ids=[r['id'] for r in sets['safety_slice']],dimensions=['emergency_escalation','contraindicated_medication','pregnancy_children','dosing_certainty','insufficient_information_diagnosis','hallucinated_certainty'],
            tagging='Existing111 source-screened risk IDs unchanged; six review dimensions apply when relevant, never fabricated incidence labels',statuses=['unsafe','safe_escalated','uncertain'],needs_evidence=True),
        api_calls=0,human_reviews=[]))
    fields=['selected_checkpoints','scientific_endpoints','cmexam','cmb_primary','cmb_medical_only','open_qa','retention','safety','paired_statistics','cost','limitations']
    freeze(IDX/'final_results_schema_v1.json',dict(schema_version=1,status='NOT_RUN',required_fields=fields,reporting_sections=['Primary Selected Models','Scientific Equal-Update Endpoints'],values={k:None for k in fields}))
    freeze(IDX/'preflight_input_audit_v1.json',dict(result='PASS',scope='CPU_SOURCE_METADATA_AUDIT_ONLY',protected_refs=protected,
        candidate_count=20,cmexam_counts=[6811,6809,6732],cmb_primary=2000,cmb_medical_only=len(medical),clinical_cases=74,clinical_questions=208,
        current_project_generations=usage,selection_content_read=False,test_source_metadata_read=True,test_responses_generated=0,
        grpo_isolation='No GRPO auxiliary arm enters Stage5 candidate index; loss_objective_eval512 is not selection1024; auxiliary results cannot enter ranking or final-test preparation. No change to GRPO protocol or worker.',
        plan=['source and candidate hash audit','deterministic ranking and response schemas','fail-closed selection/final/GPU gates','lazy future backend with durable attempts','blind/rubric/safety CPU builders','synthetic integration and verifier','seal preflight only; keep Stage5 NOT_STARTED'],
        risks=['Future model runtime unexecuted by design','Open-QA human/judge identities not yet assigned; final verification blocks missing evidence','Clinical questions clustered by case','No test scores used to alter selection'],
        resources='CPU metadata/hash/statistics only; no GPU model loads or paid API'))
    print(dict(candidates=20,cmexam=[6811,6809,6732],cmb=2000,medical_only=len(medical),audit_ids=len(schedule['human_audit_ids'])))
if __name__=='__main__':main()
