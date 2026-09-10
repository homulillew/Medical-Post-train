#!/usr/bin/env python3
"""Publish verified preparation manifests; never start Stage5 evaluation."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, durable, now


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def freeze(out):
    index = ROOT / 'experiments/stage5'
    result = read(out / 'verification.json')
    assert result['result'] == 'PASS' and result['api_calls'] == result['model_generations'] == 0
    assert read(ROOT / 'project_state.json')['stages']['5']['status'] == 'NOT_STARTED'
    for ref in read(out / 'run.json')['code']:
        assert record(ref['path']) == ref, 'Preparation source changed after run'
    assert record(result['verifier']['path']) == result['verifier']
    inputs = read(out / 'inputs.json')
    summary = read(out / 'summary.json')
    source_inventory = []
    for ref in inputs['stage1_sources']['sources']:
        path = Path(ref['path'])
        original = list(csv.DictReader(path.open())) if path.suffix == '.csv' else read(path)
        source_inventory.append(dict(**ref, raw_count=len(original), fields=list(original[0]),
            purpose='source/exclusion', source_split=ref['source']))
    for ref in inputs['official_cmb_additions']:
        if Path(ref['path']).suffix != '.json':
            continue
        original = read(ref['path'])
        source_inventory.append(dict(**ref, raw_count=len(original), fields=list(original[0]),
            purpose='gold_key' if 'choice-answer' in ref['path'] else 'clinical_source'))
    inventory = dict(sources=source_inventory, input_hashes=record(out / 'inputs.json'),
        used_sft_train=20000, used_sft_validation=1000, rl_pool=15000, monitor=512, selection=1024,
        actual_cmb_schema=dict(exam_type=6, exam_class=28, exam_subject=173,
            source_question_types={'单项选择题':9999, '多项选择题':1190, 'C型选择题':11},
            six_option_questions=42, answer_id_join='11200/11200', metadata_type_discrepancies=11),
        clinical_cases=74, clinical_questions=208, final_counts=summary['counts'])
    immutable(index / 'source_inventory.json', inventory)
    for filename in ('verification.json', 'decontamination_report.json', 'source_anomalies.json',
                     'stratification.json', 'evaluation_protocol.json', 'request_inventory.json'):
        immutable(index / filename, read(out / filename))
    for path in sorted((out / 'manifests').glob('*.json')):
        meta = read(path)
        data = rows(meta['data']['path'])
        assert record(meta['data']['path']) == meta['data']
        # Stable IDs stay in Git; full questions/references live in the bulk JSONL.
        immutable(index / 'manifests' / path.name, dict(**meta,
            ids=[r['id'] for r in data], requests=record(out / 'requests' / (meta['name'] + '.jsonl'))))
    old = out.parent / 's5_data_20260910T063500_v3'
    previous = read(old / 'summary.json')
    equality = {k: v['data']['sha256'] == previous['manifests'][k]['data']['sha256']
                for k, v in summary['manifests'].items()}
    assert all(equality[k] for k in ('cmexam_test_full', 'cmexam_test_scorable', 'cmexam_test_clean',
        'cmexam_sota_probe_512', 'cmb_exam_clean_2000', 'cmb_sota_probe_280', 'cmb_clin'))
    immutable(index / 'rebuild_consistency.json', dict(result='PASS', previous=record(old / 'summary.json'),
        current=record(out / 'summary.json'), byte_identical=equality,
        intentional_changes=['retention source eligibility: no MC option blocks/unavailable attachments',
                             'source-risk tags: exclude pregnancy similes; follow updated retention membership'],
        model_outputs_observed=False))
    rubric = ROOT / 'configs/stages/s5_open_qa_rubric_v1.json'
    supplement = dict(version='stage5-evaluation-supplement-v1', timestamp=now(),
        parent_contract=record(ROOT / 'contracts/stage_budgets.json'),
        scope='Stage5 only. Does not change the parent Stage0-4 contract or frozen training runs.',
        authority=[record(ROOT / 'docs/stages/05_evaluation.md'),
                   record(ROOT / 'docs/implementation/STAGE5_EVALUATION_DESIGN.md'),
                   record(ROOT / 'docs/implementation/STAGE5_PREP_AGENT_TASK.md')],
        mandatory=dict(primary_checkpoints=3, cmexam_official_source=6811,
            cmexam_scorable=6809, cmexam_clean=6732, cmb_clean=2000,
            external_clinical_cases=result['clinical_cases'], external_clinical_questions=result['clinical_questions'],
            retention_questions=200, source_risk_slice=summary['counts']['safety_slice'],
            minimum_human_review_fraction=.2, all_critical_safety_failures_require_human_review=True),
        source_anomalies=record(index / 'source_anomalies.json'),
        cmexam_acceptance='Official source6811 retained; two malformed records excluded from executable requests. Final report must disclose the missing-A cases and all denominators; preparation does not assert full6811 clean evaluation.',
        clinical_eval='Text case description + current question. No images or prior reference answers. Cluster uncertainty by case_id.',
        rubric=record(rubric), human_review_status='NOT_PERFORMED; required after model generation',
        checkpoint_selection_status='NOT_PERFORMED; must freeze validation-only selection before local test generation')
    immutable(ROOT / 'contracts/stage5_evaluation_supplement_v1.json', supplement)
    failures = []
    for launch in sorted(index.glob('s5_data_*_launch.json')):
        info = read(launch)
        old_out = Path(info['artifact_root'])
        failures.append(dict(run_id=old_out.name, launch=record(launch),
            status_snapshot=read(old_out / 'status.json'), run=record(old_out / 'run.json'),
            disposition='FROZEN_CURRENT' if old_out == out else 'RETAINED_FAILED_OR_SUPERSEDED_PREPARATION'))
    immutable(index / 'preparation_history.json', failures)
    artifact = out / 'evaluation_dataset_bundle.tar.gz'
    with tarfile.open(artifact, 'x:gz') as archive:
        for name in ('sets', 'requests', 'manifests', 'inputs.json', 'summary.json',
                     'verification.json', 'decontamination_report.json', 'source_anomalies.json',
                     'evaluation_protocol.json', 'request_inventory.json', 'stratification.json',
                     'exclusions.jsonl'):
            archive.add(out / name, arcname=name)
    immutable(index / 'dataset_freeze_v1.json', dict(status='DATASETS_FROZEN', timestamp=now(), run_id=out.name,
        artifact_root=str(out), bundle=record(artifact), verification=record(out / 'verification.json'),
        dataset_manifests=[record(p) for p in sorted((index / 'manifests').glob('*.json'))],
        external_probe=record(out / 'requests/external_probe_all.jsonl'),
        external_probe_requests=summary['external_probe_requests'],
        protocol=record(index / 'evaluation_protocol.json'),
        contract_supplement=record(ROOT / 'contracts/stage5_evaluation_supplement_v1.json'),
        stage5_status='NOT_STARTED', api_calls=0, project_model_test_generations=0,
        source_files=[record(ROOT / p) for p in ('scripts/prepare_stage5_datasets.py',
            'scripts/stage5_eval_data.py', 'scripts/verify_stage5_datasets.py',
            'scripts/fetch_stage5_sources.py', 'scripts/freeze_stage5_datasets.py')]))
    durable(out / 'status.json', dict(status='DATASETS_FROZEN', timestamp=now(),
        verification=record(out / 'verification.json'), stage5_status='NOT_STARTED'))
    print(json.dumps(dict(status='DATASETS_FROZEN', counts=summary['counts'],
                         api_requests=summary['external_probe_requests'], bundle=str(artifact))))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    freeze(Path(a.run))
