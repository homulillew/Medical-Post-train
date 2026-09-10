#!/usr/bin/env python3
"""Verify source-to-set joins, exclusions, quotas and reference-free requests."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, now
from medical_posttrain.data.exam import options_schema
from medical_posttrain.reward.parser import canonical_answer
from stage5_eval_data import balanced, digest, request, safety_tags, retention_quality_reason


def rows(path):
    return [json.loads(line) for line in Path(path).open()]


def verify(out):
    started = time.monotonic()
    inputs = read(out / 'inputs.json')
    summary = read(out / 'summary.json')
    source = {s['source']: s for s in inputs['stage1_sources']['sources']}
    additions = {Path(s['path']).name: s for s in inputs['official_cmb_additions']}
    sets = {}
    for path in sorted((out / 'manifests').glob('*.json')):
        manifest = read(path)
        assert record(manifest['data']['path']) == manifest['data']
        data = rows(manifest['data']['path'])
        assert len(data) == manifest['count'] == len({r['id'] for r in data})
        assert digest(data) == manifest['content_sha256']
        sets[manifest['name']] = data
        reqs = rows(out / 'requests' / (manifest['name'] + '.jsonl'))
        assert len(reqs) == manifest['request_count'] == sum(r['scorable'] for r in data)
        assert reqs == [request(r) for r in data if r['scorable']]
        assert all(set(r) == {'id', 'task', 'messages'} for r in reqs)
        assert all([m['role'] for m in r['messages']] == ['system', 'user'] for r in reqs)
    assert len(sets['cmexam_test_full']) == 6811
    assert len(sets['cmexam_test_scorable']) == 6809
    for name, count in [('cmexam_sota_probe_512', 512), ('cmb_exam_clean_2000', 2000),
                        ('cmb_sota_probe_280', 280), ('open_qa_retention_200', 200),
                        ('open_qa_sota_probe_100', 100)]:
        assert len(sets[name]) == count
    for probe, parent in [('cmexam_sota_probe_512', 'cmexam_test_clean'),
                          ('cmexam_test_clean', 'cmexam_test_scorable'),
                          ('cmexam_test_scorable', 'cmexam_test_full'),
                          ('cmb_sota_probe_280', 'cmb_exam_clean_2000'),
                          ('open_qa_sota_probe_100', 'open_qa_retention_200')]:
        mapping = {r['id']: r for r in sets[parent]}
        assert all(r == mapping[r['id']] for r in sets[probe])
    for name, count in [('cmb_exam_clean_2000', None), ('cmb_sota_probe_280', 10)]:
        counts = Counter(r['strata']['exam_class'] for r in sets[name])
        assert len(counts) == 28
        assert sorted(counts.values()) == ([71] * 16 + [72] * 12 if count is None else [10] * 28)
    assert Counter(r['source'] for r in sets['open_qa_retention_200']) == {'medical_o1': 100, 'huatuo': 100}
    assert Counter(r['source'] for r in sets['open_qa_sota_probe_100']) == {'medical_o1': 50, 'huatuo': 50}
    assert all(retention_quality_reason(r['question']) is None for r in sets['open_qa_retention_200'])

    # Check joins directly against the official source records, including gold keys.
    cm_source = list(csv.DictReader(open(source['cmexam_test']['path'])))
    for row in sets['cmexam_test_full']:
        original = cm_source[row['source_row']]
        assert digest(original) == row['source_record_sha256']
        assert row['question'] == original['Question']
        assert row['raw_options'] == original['Options']
        assert row['reference_explanation'] == original['Explanation']
        if row['scorable']:
            assert row['options'] == options_schema(original['Options'])
            assert row['answer'] == canonical_answer(original['Answer'], ''.join(row['options']))
        else:
            assert row['source_row'] in (2866, 5331) and row['schema_error'] == 'noncontiguous_options'
    cmb_source = read(source['cmb_test']['path'])
    gold = {str(r['id']): r for r in read(additions['CMB-test-choice-answer.json']['path'])}
    assert len(gold) == len(cmb_source) == 11200
    assert {str(r['id']) for r in cmb_source} == set(gold)
    for row in sets['cmb_exam_clean_2000']:
        original = cmb_source[row['source_row']]
        answer = gold[str(original['id'])]
        assert digest(original) == row['source_record_sha256']
        assert digest(answer) == row['gold_record_sha256']
        assert row['question'] == original['question']
        assert row['options'] == {k: v.strip() for k, v in sorted(original['option'].items()) if isinstance(v, str) and v.strip()}
        assert row['answer'] == canonical_answer(answer['answer'], ''.join(row['options']))
        assert all(original[k] == answer[k] for k in ('exam_type', 'exam_class', 'exam_subject'))
    clin_source = {str(r['id']): r for r in read(additions['CMB-Clin-qa.json']['path'])}
    clinical_counts = Counter()
    for row in sets['cmb_clin']:
        original = clin_source[str(row['source_id'])]
        qa = original['QA_pairs'][row['question_index']]
        assert row['question'] == qa['question'] and row['reference_answer'] == qa['answer']
        assert row['context'] == original['description']
        assert row['source_record_sha256'] == digest(original)
        clinical_counts[str(row['source_id'])] += 1
    assert all(n == len(clin_source[k]['QA_pairs']) for k, n in clinical_counts.items())
    for name in ('medical_o1', 'huatuo'):
        raw = read(source[name]['path'])
        for row in sets['open_qa_retention_200']:
            if row['source'] != name:
                continue
            original = raw[row['source_row']]
            q = original['Question'] if name == 'medical_o1' else original['conversations'][0]['value']
            a = original['Response'] if name == 'medical_o1' else original['conversations'][-1]['value']
            assert row['question'] == q
            assert row['reference_answer'].strip() == a.strip()

    membership = {sid: c['cluster_id'] for c in rows(out / 'extended_clusters.jsonl') for sid in c['members']}
    governed = {Path(r['path']).name: r for r in inputs['governed_sources']}
    used = {r['sample_id'] for r in read(governed['train_ids.json']['path']) + read(governed['validation_ids.json']['path'])}
    used |= {r['prompt_id'] for r in rows(inputs['rl_pool']['path'])}
    partition = read(inputs['partitions']['path'])
    used |= set(partition['monitor'] + partition['selection'])
    forbidden = {membership[sid] for sid in used}
    for name in ('cmexam_test_clean', 'cmb_exam_clean_2000', 'cmb_clin', 'open_qa_retention_200'):
        assert all(r['id'] not in used and r['cluster_id'] not in forbidden for r in sets[name])
        if name != 'cmb_clin':
            assert len({r['cluster_id'] for r in sets[name]}) == len(sets[name])
    exam_clusters = {r['cluster_id'] for r in sets['cmexam_test_full']}
    cmb_clusters = {r['cluster_id'] for r in sets['cmb_exam_clean_2000']}
    assert not exam_clusters & cmb_clusters
    clinical_clusters = {r[k] for r in sets['cmb_clin'] for k in ('cluster_id', 'context_cluster_id')}
    assert not clinical_clusters & (forbidden | exam_clusters | cmb_clusters)
    retention_clusters = {r['cluster_id'] for r in sets['open_qa_retention_200']}
    assert not retention_clusters & (clinical_clusters | exam_clusters | cmb_clusters)
    benchmark_clusters = {cid for sid, cid in membership.items() if sid.startswith(('cmexam_', 'cmb_test:'))}
    assert not retention_clusters & benchmark_clusters

    checks = [('cmexam_sota_probe_512', sets['cmexam_test_clean'], 512, 'cmexam_probe',
               lambda r: (r['strata']['Difficulty level'], r['strata']['Clinical Department'])),
              ('cmb_exam_clean_2000', rows(out / 'selection_populations/cmb_eligible.jsonl'), 2000, 'cmb_final', lambda r: r['strata']['exam_class']),
              ('cmb_sota_probe_280', sets['cmb_exam_clean_2000'], 280, 'cmb_probe', lambda r: r['strata']['exam_class']),
              ('open_qa_retention_200', rows(out / 'selection_populations/retention_eligible.jsonl'), 200, 'retention_final', lambda r: r['source']),
              ('open_qa_sota_probe_100', sets['open_qa_retention_200'], 100, 'retention_probe', lambda r: r['source'])]
    for name, population, n, domain, key in checks:
        chosen, _ = balanced(list(reversed(population)), n, domain, key)
        assert [r['id'] for r in chosen] == [r['id'] for r in sets[name]]
    open_ids = {r['id'] for name in ('cmb_clin', 'open_qa_retention_200') for r in sets[name]}
    assert all(r['id'] in open_ids and r['safety_tags'] == safety_tags(r) for r in sets['safety_slice'])
    inventory = read(out / 'request_inventory.json')
    bundle = rows(out / 'requests/external_probe_all.jsonl')
    assert len(bundle) == len({r['id'] for r in bundle}) == inventory['external_probe_request_count']
    expected = [request(r) for name in ('cmexam_sota_probe_512', 'cmb_sota_probe_280', 'cmb_clin', 'open_qa_sota_probe_100') for r in sets[name]]
    assert bundle == expected
    assert read(ROOT / 'project_state.json')['stages']['5']['status'] == 'NOT_STARTED'
    pair = read(ROOT / 'experiments/stage4/formal_pair.json')
    from medical_posttrain.evidence import sha256
    assert all(sha256(ROOT / p) == h for p, h in pair['execution_hashes'].items())
    receipt = dict(result='PASS', timestamp=now(), scope='Evaluation dataset preparation only',
        counts=summary['counts'], clinical_cases=len(clinical_counts), clinical_questions=sum(clinical_counts.values()),
        external_probe_requests=len(bundle), source_join_checks='PASS', decontamination_checks='PASS',
        subset_and_quota_checks='PASS', reverse_order_selection_replay='PASS',
        request_reference_isolation='PASS', frozen_stage4_files_unchanged=len(pair['execution_hashes']),
        stage5_status='NOT_STARTED', api_calls=0, model_generations=0,
        historical_source_anomalies_disclosed=record(out / 'source_anomalies.json'),
        seconds=time.monotonic() - started, verifier=record(__file__), summary=record(out / 'summary.json'))
    immutable(out / 'verification.json', receipt)
    return receipt


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    print(json.dumps(verify(Path(a.run)), indent=2))
