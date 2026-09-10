#!/usr/bin/env python3
"""Prepare complete governed evaluation sets; never invoke a model."""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import uuid
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.data.stage1 import DuplicateGraph, normalize_question, quality_reason, text_hash
from medical_posttrain.data.medical import medical_o1, huatuo
from medical_posttrain.data.exam import options_schema
from medical_posttrain.reward.parser import canonical_answer
from medical_posttrain.rl.common import read, record, immutable, durable, now
from stage5_eval_data import SEED, EXAM_PROMPT, OPEN_PROMPT, SAFETY_RULES, balanced, digest, rank, request, safety_tags, retention_quality_reason

DATA = Path('/data/WSH/medical-post-train-artifacts/data')
S1 = DATA / 'stage1/s1_data_20260908T144854_bac3a7'
S2 = Path('/data/WSH/medical-post-train-artifacts/runs/s2_data_20260909T024052_870aba')
NEW = DATA / 'stage5/raw/6c8ece46097dae736c6805dd3b831e1a38c08971'
REV = NEW.name
INDEX = ROOT / 'experiments/stage5'


def lines(path):
    with Path(path).open() as f:
        for line in f:
            yield json.loads(line)


def dump(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n')


def event(name, **fields):
    print(json.dumps(dict(timestamp=now(), event=name, **fields)), flush=True)


def prepare(out):
    started = time.monotonic()
    raw = read(DATA / 'stage1/raw_manifest.json')
    sources = {r['source']: r for r in raw['sources']}
    required = [S1 / n for n in ('clusters.jsonl', 'clean_candidates.jsonl', 'train_ids.json', 'validation_ids.json')]
    old_refs = {r['path']: r for r in read(ROOT / 'experiments/stage1/s1_data_20260908T144854_bac3a7/artifacts_manifest.json')}
    for path in required:
        assert record(path) == old_refs[str(path)], str(path)
    for source in sources.values():
        assert record(source['path']) == {k: source[k] for k in ('path', 'sha256', 'bytes')}
    downloads = read(NEW / 'download_manifest.json')
    for source in downloads:
        assert record(source['path']) == {k: source[k] for k in ('path', 'sha256', 'bytes')}
    expected_s2 = read(ROOT / 'experiments/stage2/s2_data_20260909T024052_870aba/summary.json')
    assert record(S2 / 'candidate_pool.jsonl') == expected_s2['pool']
    assert record(S2 / 'validation_partitions.json') == expected_s2['validation_partition']
    immutable(out / 'inputs.json', dict(stage1_sources=raw, official_cmb_additions=downloads,
        governed_sources=[record(p) for p in required],
        rl_pool=record(S2 / 'candidate_pool.jsonl'), partitions=record(S2 / 'validation_partitions.json')))
    train_ids = {r['sample_id'] for r in read(S1 / 'train_ids.json')}
    val_ids = {r['sample_id'] for r in read(S1 / 'validation_ids.json')}
    rl_ids = {r['prompt_id'] for r in lines(S2 / 'candidate_pool.jsonl')}
    partition = read(S2 / 'validation_partitions.json')
    used_validation = set(partition['monitor'] + partition['selection'])
    forbidden_ids = train_ids | val_ids | rl_ids | used_validation
    used_sft_ids = train_ids | val_ids
    assert len(train_ids) == 20000 and len(val_ids) == 1000 and len(rl_ids) == 15000
    assert len(used_validation) == 1536

    # Reconstruct the exact historical question corpus; retain its sealed unions.
    base = list(lines(DATA / 'stage1/benchmark_questions.jsonl'))
    corpus = [dict(id=r['sample_id'], text=r['question']) for r in base]
    for source, adapter in (('medical_o1', medical_o1), ('huatuo', huatuo)):
        for i, row in enumerate(read(sources[source]['path'])):
            try:
                messages = adapter(row)
                if quality_reason(messages):
                    continue
            except ValueError:
                continue
            corpus.append(dict(id=f'{source}:{sources[source]["revision"]}:{i}',
                text='\n'.join(m['content'] for m in messages if m['role'] == 'user')))
    old_count = len(corpus)
    clinical_source = read(NEW / 'CMB-Clin-qa.json')
    clinical_ref = record(NEW / 'CMB-Clin-qa.json')
    clinical = []
    for case in clinical_source:
        cid = f'cmb_clin:{REV}:{case["id"]}'
        assert case['description'].strip() and case['QA_pairs']
        corpus.append(dict(id=cid + ':context', text=case['description']))
        for j, qa in enumerate(case['QA_pairs']):
            sid = cid + f':q{j}'
            assert qa['question'].strip() and qa['answer'].strip()
            corpus.append(dict(id=sid, text=case['description'] + '\n' + qa['question']))
            clinical.append(dict(id=sid, source='cmb_clin', source_revision=REV,
                source_id=case['id'], source_split='official_clin', case_id=cid, question_index=j,
                task='open_qa', question=qa['question'], context=case['description'],
                reference_answer=qa['answer'], source_record_sha256=digest(case),
                source_file=clinical_ref, scorable=True))
    index = {r['id']: i for i, r in enumerate(corpus)}
    assert len(index) == len(corpus)
    texts = [normalize_question(r['text']) or 'unusablebenchmark' + text_hash(r['id']) for r in corpus]
    graph = DuplicateGraph(texts)
    for c in lines(S1 / 'clusters.jsonl'):
        positions = [index[sid] for sid in c['members']]
        for i in positions[1:]:
            graph.parents[i] = positions[0]
    event('extending_sealed_duplicate_graph', original_records=old_count, total=len(corpus))
    edges = []
    graph.join(lambda i, j, reason, similarity: edges.append(dict(
        left=corpus[i]['id'], right=corpus[j]['id'], reason=reason, similarity=similarity)),
        lambda n: event('duplicate_graph_progress', processed=n, total=len(corpus)))
    groups = defaultdict(list)
    for i, row in enumerate(corpus):
        groups[graph.root(i)].append(row['id'])
    membership = {}
    cluster_members = {}
    for members in groups.values():
        cid = 'cluster:' + digest(sorted(members))
        cluster_members[cid] = members
        for sid in members:
            membership[sid] = cid
    dump(out / 'extended_clusters.jsonl', [dict(cluster_id=k, members=v) for k, v in sorted(cluster_members.items())])
    dump(out / 'new_duplicate_edges.jsonl', edges)
    del graph, texts, corpus, index, groups
    forbidden_clusters = {membership[sid] for sid in forbidden_ids}
    benchmark_clusters = {membership[r['sample_id']] for r in base}
    exclusions = []
    anomalies = []

    def reject(row, reason):
        exclusions.append(dict(id=row['id'], source=row['source'], reason=reason,
            cluster_id=row['cluster_id'], matched_ids=cluster_members[row['cluster_id']]))

    def compact_exam(row, sid, source, revision, source_row, question, options, answer, strata, error=None):
        return dict(id=sid, source=source, source_revision=revision, source_row=source_row,
            source_split='test', source_record_sha256=digest(row),
            task='exam', question=question, options=options, answer=answer,
            strata=strata, cluster_id=membership[sid], scorable=error is None,
            schema_error=error)

    cm = []
    with open(sources['cmexam_test']['path']) as f:
        cm_source = list(csv.DictReader(f))
    for i, row in enumerate(cm_source):
        sid = f'cmexam_test:{sources["cmexam_test"]["revision"]}:{i}'
        error = None
        try:
            options = options_schema(row['Options'])
            answer = canonical_answer(row['Answer'], ''.join(options))
        except ValueError as exc:
            error = str(exc)
            options = {}  # Preserve raw options verbatim; do not repair missing choices.
            answer = row['Answer']
            anomalies.append(dict(id=sid, reason=error, raw_options=row['Options'], raw_answer=row['Answer']))
        item = compact_exam(row, sid, 'cmexam', sources['cmexam_test']['revision'], i,
            row['Question'], options, answer, {k: row[k] for k in (
                'Disease Group', 'Area of Competency', 'Clinical Department', 'Medical Discipline', 'Difficulty level')}, error)
        item.update(raw_options=row['Options'], reference_explanation=row['Explanation'],
                    source_file=sources['cmexam_test'])
        cm.append(item)
    assert len(cm) == 6811
    cm_clusters = {r['cluster_id'] for r in cm}
    seen = set()
    cm_clean = []
    for row in sorted(cm, key=lambda r: rank('cmexam_clean', r['id'])):
        reason = ('invalid_exam_schema' if not row['scorable'] else
                  'training_or_validation_cluster_overlap' if row['cluster_id'] in forbidden_clusters else
                  'within_track_duplicate_cluster' if row['cluster_id'] in seen else None)
        if reason:
            reject(row, reason)
        else:
            seen.add(row['cluster_id'])
            cm_clean.append(row)
    cm_probe, cm_quotas = balanced(cm_clean, 512, 'cmexam_probe',
        lambda r: (r['strata']['Difficulty level'], r['strata']['Clinical Department']))

    cmb_source = read(sources['cmb_test']['path'])
    gold_source = read(NEW / 'CMB-test-choice-answer.json')
    gold_ref = record(NEW / 'CMB-test-choice-answer.json')
    gold = {str(r['id']): r for r in gold_source}
    assert len(gold) == len(gold_source) == len(cmb_source) == 11200
    assert set(gold) == {str(r['id']) for r in cmb_source}
    cmb_all, cmb_eligible, seen = [], [], set()
    for i, row in enumerate(cmb_source):
        g = gold[str(row['id'])]
        assert all(row[k] == g[k] for k in ('exam_type', 'exam_class', 'exam_subject'))
        sid = f'cmb_test:{sources["cmb_test"]["revision"]}:{i}'
        options = {k: v.strip() for k, v in sorted(row['option'].items()) if isinstance(v, str) and v.strip()}
        assert 2 <= len(options) <= 26 and list(options) == list('ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)])
        answer = canonical_answer(g['answer'], ''.join(options))
        item = compact_exam(row, sid, 'cmb_exam', sources['cmb_test']['revision'], i,
            row['question'], options, answer, {k: row[k] for k in ('exam_type', 'exam_class', 'exam_subject', 'question_type')})
        item.update(source_id=row['id'], source_file=sources['cmb_test'], gold_source_file=gold_ref,
                    gold_source_revision=REV, gold_record_sha256=digest(g))
        if row['question_type'] != g['question_type']:
            anomalies.append(dict(id=sid, reason='question_type_version_mismatch',
                                  question_type=row['question_type'], gold_metadata_type=g['question_type']))
        cmb_all.append(item)
    for row in sorted(cmb_all, key=lambda r: rank('cmb_clean', r['id'])):
        reason = ('training_or_validation_cluster_overlap' if row['cluster_id'] in forbidden_clusters else
                  'cross_track_cmexam_overlap' if row['cluster_id'] in cm_clusters else
                  'within_track_duplicate_cluster' if row['cluster_id'] in seen else None)
        if reason:
            reject(row, reason)
        else:
            seen.add(row['cluster_id'])
            cmb_eligible.append(row)
    assert len({r['strata']['exam_class'] for r in cmb_eligible}) == 28
    cmb_final, cmb_quotas = balanced(cmb_eligible, 2000, 'cmb_final', lambda r: r['strata']['exam_class'])
    assert sorted(cmb_quotas.values()) == [71] * 16 + [72] * 12
    cmb_probe, cmb_probe_quotas = balanced(cmb_final, 280, 'cmb_probe', lambda r: r['strata']['exam_class'])
    assert set(cmb_probe_quotas.values()) == {10}
    cmb_clusters = {r['cluster_id'] for r in cmb_final}

    clinical_clean = []
    blocked_cases = {}
    for row in clinical:
        row['cluster_id'] = membership[row['id']]
        context_cluster = membership[row['case_id'] + ':context']
        row['context_cluster_id'] = context_cluster
        touched = {row['cluster_id'], context_cluster}
        if touched & forbidden_clusters:
            blocked_cases[row['case_id']] = 'clinical_case_training_or_validation_overlap'
        elif touched & (cm_clusters | cmb_clusters):
            blocked_cases[row['case_id']] = 'clinical_case_cross_track_exam_overlap'
    clinical_contexts = set()
    for cid in sorted({r['case_id'] for r in clinical}, key=lambda x: rank('clinical_case', x)):
        rows = [r for r in clinical if r['case_id'] == cid]
        context_cluster = rows[0]['context_cluster_id']
        reason = blocked_cases.get(cid)
        if not reason and context_cluster in clinical_contexts:
            reason = 'duplicate_clinical_case_context'
        if reason:
            for row in rows:
                reject(row, reason)
        else:
            clinical_contexts.add(context_cluster)
            clinical_clean.extend(rows)
    clinical_clusters = {r[k] for r in clinical_clean for k in ('cluster_id', 'context_cluster_id')}
    retention_eligible = []
    for row in lines(S1 / 'clean_candidates.jsonl'):
        sid = row['sample_id']
        cid = membership[sid]
        item = dict(id=sid, source=row['source'], cluster_id=cid)
        reason = ('sft_train_or_validation_used' if sid in used_sft_ids else
                  'benchmark_cluster_overlap' if cid in benchmark_clusters else
                  'training_or_validation_cluster_overlap' if cid in forbidden_clusters else
                  'clinical_cluster_overlap' if cid in clinical_clusters else retention_quality_reason(row['question']))
        if reason:
            reject(item, reason)
            continue
        # Only final source answers are reference text; source CoT is never a candidate input.
        final = row['messages'][-1]['content']
        import re
        answer = re.fullmatch(r'(?:<think>[\s\S]*?</think>\s*)?<answer>([\s\S]*)</answer>', final)
        assert answer, sid
        item.update(task='open_qa', question=row['question'], context='', reference_answer=answer[1],
            source_revision=row['source_revision'], source_row=row['source_row'],
            source_split='unused_sft_source', source_file=sources[row['source']],
            source_record_sha256=digest(row), scorable=True)
        retention_eligible.append(item)
    # New clinical bridges can merge historical clusters; choose one deterministic representative.
    retained, seen = [], set()
    for row in sorted(retention_eligible, key=lambda r: rank('retention_unique', r['id'])):
        if row['cluster_id'] in seen:
            reject(row, 'retention_duplicate_cluster')
        else:
            seen.add(row['cluster_id'])
            retained.append(row)
    retention, retention_quotas = balanced(retained, 200, 'retention_final', lambda r: r['source'])
    assert set(retention_quotas.values()) == {100}
    retention_probe, _ = balanced(retention, 100, 'retention_probe', lambda r: r['source'])
    safety = []
    for row in clinical_clean + retention:
        tags = safety_tags(row)
        if any(t['tag'] != 'insufficient_information' for t in tags):
            safety.append(dict(**row, safety_tags=tags, safety_tag_status='SOURCE_RULE_SCREENED',
                safety_scope='Heuristic source risk slice; no candidate safety judgment'))

    sets = dict(cmexam_test_full=cm, cmexam_test_scorable=[r for r in cm if r['scorable']],
        cmexam_test_clean=cm_clean, cmexam_sota_probe_512=cm_probe,
        cmb_exam_clean_2000=cmb_final, cmb_sota_probe_280=cmb_probe,
        cmb_clin=clinical_clean, open_qa_retention_200=retention,
        open_qa_sota_probe_100=retention_probe, safety_slice=safety)
    dump(out / 'exclusions.jsonl', exclusions)
    immutable(out / 'source_anomalies.json', anomalies)
    immutable(out / 'stratification.json', dict(cmexam_probe=cm_quotas, cmb_final=cmb_quotas,
        cmb_probe=cmb_probe_quotas, retention=retention_quotas))
    # Retain eligible populations so an independent verifier can reconstruct the selected IDs.
    for name, rows in dict(cmb_eligible=cmb_eligible, retention_eligible=retained).items():
        dump(out / 'selection_populations' / f'{name}.jsonl', rows)
    manifests = {}
    for name, rows in sets.items():
        for row in rows:
            row['selection_seed'] = SEED
        path = out / 'sets' / f'{name}.jsonl'
        dump(path, rows)
        meta = dict(name=name, count=len(rows), unique_ids=len({r['id'] for r in rows}),
            unique_clusters=len({r['cluster_id'] for r in rows}), seed=SEED,
            data=record(path), content_sha256=digest(rows),
            scorable_count=sum(r['scorable'] for r in rows),
            request_count=sum(r['scorable'] for r in rows),
            clinical_case_count=len({r['case_id'] for r in rows if 'case_id' in r}))
        assert meta['unique_ids'] == len(rows)
        immutable(out / 'manifests' / f'{name}.json', meta)
        manifests[name] = meta
        dump(out / 'requests' / f'{name}.jsonl', [request(r) for r in rows if r['scorable']])
    probe_names = ['cmexam_sota_probe_512', 'cmb_sota_probe_280', 'cmb_clin', 'open_qa_sota_probe_100']
    probe_rows = [r for name in probe_names for r in sets[name]]
    dump(out / 'requests/external_probe_all.jsonl', [request(r) for r in probe_rows])
    all_rows = sets['cmexam_test_scorable'] + cmb_final + clinical_clean + retention
    dump(out / 'requests/project_final_all.jsonl', [request(r) for r in all_rows])
    immutable(out / 'decontamination_report.json', dict(
        method='Inherited sealed Stage1 clusters + same-rule graph extension for clinical cases',
        thresholds=dict(char3_candidate=.65, char5_confirm=.85, sequence_confirm=.90),
        semantic_decontamination=False, inputs=record(out / 'inputs.json'),
        extended_clusters=record(out / 'extended_clusters.jsonl'), new_edges=record(out / 'new_duplicate_edges.jsonl'),
        exclusions=record(out / 'exclusions.jsonl'),
        exclusion_counts=dict(Counter(r['source'] + ':' + r['reason'] for r in exclusions)),
        source_anomalies=record(out / 'source_anomalies.json'),
        clinical_source_cases=len(clinical_source), clinical_source_questions=len(clinical),
        clinical_retained_cases=len({r['case_id'] for r in clinical_clean}),
        clinical_retained_questions=len(clinical_clean),
        caveat='CMExam official full/scorable sets retain source duplicate membership; only clean sets and probes claim cluster exclusions. Clinical questions within a case are correlated.'))
    immutable(out / 'evaluation_protocol.json', dict(version='stage5-eval-v1', seed=SEED,
        exam_system_prompt=EXAM_PROMPT, open_system_prompt=OPEN_PROMPT,
        parser=record(ROOT / 'src/medical_posttrain/reward/parser.py'),
        scoring='exact canonical option set; no partial credit; valid unparseable response is incorrect',
        candidate_inputs='Only system/user messages from requests/*.jsonl; no gold/reference fields',
        clinical_context='Original case description plus current question; no previous reference answers',
        clinical_statistical_unit='case_id; report both case and question counts',
        safety_rules=SAFETY_RULES, safety_status='Source-rule-screened; candidate flags require separate rubric audit',
        safety_exclusions='Negated clause-local keyword mentions; insulin-like biomarker text; drug-allergy-history-only text; weak uncertainty markers alone do not enter the risk slice',
        api_calls_performed=0, project_test_generations_performed=0,
        api_identity_and_decoding='Freeze exact provider/model/accepted parameters and token caps before API execution; applies to all chosen probe items',
        retries='Transport/rate-limit failures only, never valid wrong/unparseable outputs; retain every attempt',
        failure_accounting='Report planned/completed/failed; unresolved transport failures keep run incomplete; never silently drop from denominator',
        primary_exam_sets=['cmexam_test_full', 'cmb_exam_clean_2000'],
        cmexam_source_anomaly_policy='Preserve all6811; 2 missing-A source items have no valid strict-schema request. Separately report6809 scorable and clean-set metrics; final full-source acceptance must disclose anomaly disposition.',
        stage5_status='NOT_STARTED', local_model_test_gate='Stage4 complete and checkpoint selection frozen'))
    immutable(out / 'request_inventory.json', dict(external_probe_request_count=len(probe_rows),
        external_probe=record(out / 'requests/external_probe_all.jsonl'),
        project_requests_per_model=len(all_rows), project_requests_three_models=3 * len(all_rows),
        external_probe_tracks={n: len(sets[n]) for n in probe_names},
        deduplicate_safety_requests=True, api_cost_incurred=0,
        note='Safety is a view of existing open QA requests, not extra generation. API token/cost estimate requires chosen provider/tokenizer and output cap.'))
    summary = dict(run_id=out.name, status='PREPARED_UNVERIFIED', stage5_status='NOT_STARTED',
        counts={k: len(v) for k, v in sets.items()}, source_anomalies=len(anomalies),
        external_probe_requests=len(probe_rows), project_requests_per_model=len(all_rows),
        elapsed_seconds=time.monotonic() - started, manifests=manifests,
        api_calls=0, local_model_generations=0)
    immutable(out / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run-id')
    args = p.parse_args()
    name = args.run_id or 's5_data_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    out = DATA / 'stage5' / name
    out.mkdir(parents=True, exist_ok=False)
    code = [Path(__file__), ROOT / 'scripts/stage5_eval_data.py']
    with zipfile.ZipFile(out / 'source.zip', 'x') as archive:
        for path in code:
            archive.write(path, path.name)
    immutable(out / 'run.json', dict(run_id=name, run_class='DIAGNOSTIC', purpose='EVALUATION_PREPARATION', stage=5,
        command=sys.argv, cwd=str(ROOT), timestamp=now(), code=[record(p) for p in code],
        source_archive=record(out / 'source.zip'),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_state=subprocess.check_output(['git', 'status', '--porcelain'], text=True),
        environment=dict(python=sys.version, platform=platform.platform(), gpu_used=False),
        planned=dict(cmexam_official=6811, cmb_clean=2000, retention=200,
                     clinical='all clean source cases', safety='source-supported slice')))
    durable(out / 'status.json', dict(status='RUNNING', pid=os.getpid(), timestamp=now()))
    event('run_created', path=str(out))
    try:
        summary = prepare(out)
        durable(out / 'status.json', dict(status=summary['status'], timestamp=now()))
        event('preparation_finished', **{k: v for k, v in summary.items() if k != 'manifests'})
    except BaseException as exc:
        durable(out / 'status.json', dict(status='FAILED', error=repr(exc), traceback=traceback.format_exc(), timestamp=now()))
        raise


if __name__ == '__main__':
    main()
