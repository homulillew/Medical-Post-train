#!/usr/bin/env python3
"""Seal CPU preflight evidence without opening Stage5 or touching GPU workers."""
import sys,xml.etree.ElementTree as ET,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.evaluation.core import freeze,ref,read
IDX=ROOT/'experiments/stage5'

def main():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    source=Path(sys.argv[1]);tree=ET.parse(source);suites=list(tree.getroot().iter('testsuite'));tests=sum(int(s.get('tests','0')) for s in suites);failed=sum(int(s.get('failures','0'))+int(s.get('errors','0')) for s in suites)
    assert tests>=27 and failed==0
    destination=IDX/'preflight_tests_final.xml';destination.write_bytes(source.read_bytes())
    freeze(IDX/'preflight_test_results_v1.json',dict(result='PASS',tests=tests,failed=failed,synthetic_only=True,gpu_model_loads=0,paid_api_calls=0,
        junit=ref(destination),source=ref(ROOT/'tests/test_stage5_preflight.py'),failed_attempts=[ref(IDX/'preflight_tests_attempt_001.xml')],
        initial_failure_resolution='Source-hash tamper already refused execution but raised ValueError; standardized rejection to PermissionError and reran complete CPU suite.',
        runtime_import_guard='Subprocess MetaPathFinder rejects torch/vllm/transformers/peft/requests/httpx imports during dry-run'))
    freeze(IDX/'open_qa_judgment_schema_v1.json',dict(status='NOT_RUN',required=['pair_id','preference','scores','judge_id','judge_version','judge_is_candidate','raw_judgment','rubric_protocol_sha256'],
        rubric_protocol=ref(IDX/'open_qa_eval_protocol_v1.json'),preference=['A win','tie','B win'],score_range=[0,4],judge_is_candidate=False,
        raw_judgment='Immutable path/SHA256/bytes reference to original judge output; no API invocation in this task',human_reviews=[]))
    freeze(IDX/'final_execution_plan_v1.json',dict(status='NOT_RUN',selection_prerequisite='FROZEN_SELECTED with hashed result lock; Stage4 DONE/full PASS/READY YES',
        sections=['Primary Selected Models','Scientific Equal-Update Endpoints'],models=['sft','selected_vanilla','selected_dynamic','vanilla:5000','dynamic:5000'],
        source_datasets=['cmexam_test_scorable6809','cmb_exam_clean_2000','cmb_clin208','retention200'],
        secondary_reuse=['cmexam_clean6732','cmb_medical_only1929','safety111'],secondary_extra_generations=0,
        exam_unique_checkpoint_range=[3,5],exam_expected_generation_range=[3*8809,5*8809],open_expected_generations=1224,
        duplicate_selected_5000_policy='Reuse identical raw output when checkpoint matches; always separate scientific vs selected reporting sections',
        commands=dict(selection='.venv-train/bin/python scripts/run_stage5_eval.py --selection-only',final='.venv-train/bin/python scripts/run_stage5_eval.py --final-only',
            explicit_resume='add --resume; add --technical-retry only for an unresolved technical failure; zero automatic retries'),
        generation_status='NOT_RUN',runtime_validation='CPU synthetic integration only; GPU runtime must meet frozen versions and context checks before future inference'))
    names=['checkpoint_candidate_index_v1.json','selection_execution_plan_v1.json','selection_response_schema_v1.json','cmexam_final_eval_protocol_v1.json','cmexam_slice_manifest_v1.json',
        'cmexam_source_metadata_audit_v1.json','cmb_primary_audit_v1.json','cmb_medical_only_manifest_v1.json','cmb_source_metadata_audit_v1.json','open_qa_eval_protocol_v1.json',
        'open_qa_blind_schedule_v1.json','open_qa_judgment_schema_v1.json','final_results_schema_v1.json','final_execution_plan_v1.json','preflight_start_snapshot_v1.json','preflight_input_audit_v1.json',
        'preflight_test_results_v1.json','checkpoint_selection_protocol_v1.json','evaluation_protocol.json']
    scripts=['prepare_stage5_preflight.py','select_stage5_checkpoint.py','run_stage5_eval.py','stage5_model_worker.py','analyze_stage5_eval.py','verify_stage5.py','seal_stage5_preflight.py','checkpoint_selection.py']
    sources=[ROOT/'scripts'/s for s in scripts]+list((ROOT/'src/medical_posttrain/evaluation').glob('*.py'))+[ROOT/'tests/test_stage5_preflight.py',ROOT/'src/medical_posttrain/reward/parser.py',ROOT/'src/medical_posttrain/data/exam.py']
    freeze(IDX/'preflight_seal_v1.json',dict(status='FROZEN_CPU_PREFLIGHT_INPUTS_AND_HARNESS',timestamp=now,artifacts=[ref(IDX/n) for n in names],sources=[ref(p) for p in sources],
        project_state_reference_is_start_snapshot_only='Stage4 must later advance to DONE; its mutable project_state hash is protected during preflight only, not a permanent future launch requirement.'))
    freeze(IDX/'preflight_v1.json',dict(status='PREPARED_AWAITING_PREFLIGHT_VERIFIER',scope='CPU_ONLY',timestamp=now,seal=ref(IDX/'preflight_seal_v1.json'),tests=ref(IDX/'preflight_test_results_v1.json'),
        result_artifacts='All NOT_RUN/empty until real future execution and audits',Stage5='NOT_STARTED',Stage6='NOT_STARTED',gpu_model_loads=0,paid_api_calls=0,
        grpo_auxiliary='Read-only observation; existing worker and frozen protocol untouched',
        next_action='Allow running GRPO auxiliary to finish. After Stage4 human review/full verifier reaches DONE and READY_FOR_STAGE5=YES, run selection1024 with exclusive GPU ownership. No final CMExam/CMB tests before selection is frozen.'))
    print('SEALED_PREFLIGHT_ONLY')
if __name__=='__main__':main()
