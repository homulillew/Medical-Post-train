#!/usr/bin/env python3
"""Verify revised exam scoring against immutable raw evidence, without inference.

Reuses the historical full token replay only after verifying its complete-file
references, every original response/raw receipt, and v1 score reconstruction.
Recomputes v3 predictions and all published statistics independently of summaries.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read, rows, ref, check_ref, digest, score, summarize
from medical_posttrain.evaluation.exam_parser_v3 import rescore
from medical_posttrain.evaluation.exam_parser_v2 import rescore as rescore_v2
from stage5_objective_evidence import put, roles, EXAMS
import rescore_stage5_exams_v3 as revision

IDX = ROOT/'experiments/stage5'
DEST = IDX/'closure_v3_20260918'


def verify_job(summary, previous_receipt):
    for key in ['predictions', 'provenance', 'generation_spec', 'original_complete']:
        check_ref(summary[key])
    assert previous_receipt['result'] == 'PASS'
    assert previous_receipt['complete'] == summary['original_complete']
    assert previous_receipt['raw_token_decode'] and previous_receipt['request_lineage']
    spec = read(summary['generation_spec']['path'])
    for key in ['items', 'requests', 'adapter']: check_ref(spec[key])
    assert spec['adapter'] == previous_receipt['adapter']
    items, requests = rows(spec['items']['path']), rows(spec['requests']['path'])
    predictions = rows(summary['predictions']['path'])
    provenance = read(summary['provenance']['path'])
    complete = read(summary['original_complete']['path'])
    cid = spec['checkpoint_id']
    assert cid == summary['checkpoint_id'] == previous_receipt['checkpoint_id']
    assert len(items) == len(requests) == len(predictions) == len(provenance) == complete['n']
    assert [x['id'] for x in items] == [x['id'] for x in requests] == complete['prompt_ids']
    assert [x['prompt_id'] for x in predictions] == complete['prompt_ids']
    assert [x['response'] for x in provenance] == complete['response_refs']
    original, revised, old_v2 = [], [], []
    for item, request, prediction, source in zip(items, requests, predictions, provenance):
        for key in ['receipt', 'response', 'raw']: check_ref(source[key])
        receipt = read(source['receipt']['path'])
        assert all(source[k] == v for k,v in receipt.items())
        d = Path(source['receipt']['path']).parent
        assert d.name == digest(item['id'])[:24]
        reservation = read(d/'reservation.json')
        assert reservation['request_sha256'] == digest(request)
        assert reservation['prompt_id'] == item['id']
        assert Path(source['raw']['path']) == d/f"raw_attempt_{receipt['attempts']:03d}.json"
        raw = read(source['raw']['path'])
        assert raw['adapter_sha256'] == spec['adapter']['sha256']
        old = read(source['response']['path'])
        assert old == score(raw, item, cid, spec['run_id'])
        fresh = rescore(old, ''.join(item['options']))
        assert prediction == fresh, f'V3 prediction mismatch: {cid} {item["id"]}'
        original.append(old); revised.append(fresh)
        old_v2.append(rescore_v2(old, ''.join(item['options'])))
    assert summarize(original) == complete['summary'] == summary['original_summary']
    assert summarize(revised) == summary['summary']
    assert summarize(old_v2) == summary['v2_summary']
    assert dict(Counter(x['parser_error'] or 'parsed' for x in revised)) == summary['new_error_counts']
    return dict(result='PASS', n=len(items), checkpoint_id=cid, dataset=spec['dataset'],
                summary=summary['summary'], predictions=summary['predictions'],
                generation_complete=summary['original_complete'], raw_receipts=len(provenance),
                token_replay='Historical full token replay reused through unchanged complete/response/raw hashes')


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for source in read(revision.INDEX/'source_manifest.json')['sources']: check_ref(source)
    historical = read(IDX/'stage5_objective_verification_v1.json')
    old_handoff = read(ROOT/'experiments/handoffs/stage5_results_to_chatgpt_v1.json')
    check_ref(old_handoff['evidence']['stage5_objective_verification_v1.json'])
    assert historical['result'] == 'PASS'
    prior = {r['complete']['path']:r for r in historical['receipts']}
    primary, endpoints = roles(); verified=[]; job_refs=[]
    for dataset in EXAMS:
        for cid in sorted(set(primary.values()) | set(endpoints.values())):
            path = revision.INDEX/f"{cid.replace(':','_')}_{dataset}.json"
            summary=read(path)
            verified.append(verify_job(summary, prior[summary['original_complete']['path']]))
            job_refs.append(ref(path))
            print(f'PASS {cid} {dataset}', flush=True)
    # Rebuild all summaries, secondary subsets, slices and paired statistics in
    # scratch space rather than accepting existing published tables as expected.
    published = revision.INDEX
    with tempfile.TemporaryDirectory(prefix='stage5-v3-audit-') as temp:
        revision.INDEX = Path(temp)
        try:
            revision.final_tables()
            for filename in ['cmexam_final_results_v3.json','cmb_final_results_v3.json']:
                assert read(Path(temp)/filename) == read(published/filename), filename
        finally:
            revision.INDEX = published
    receipt = dict(result='PASS', status='OBJECTIVE_EVAL_PASS', scope='V3_FINAL_EXAMS_ONLY',
        parser_version='stage5-exam-parser-v3', jobs=len(verified), responses=sum(x['n'] for x in verified),
        receipts=verified, job_summaries=job_refs, statistics_recomputed=True,
        source_manifest=ref(published/'source_manifest.json'), verifier=ref(__file__),
        historical_raw_verification=ref(IDX/'stage5_objective_verification_v1.json'),
        results={n:ref(published/n) for n in ['cmexam_final_results_v3.json','cmb_final_results_v3.json']},
        selection=ref(IDX/'selection_result_manifest_v1.json'),
        training_checkpoint_pruning='Historical Stage4 PASS predates owner-authorized cleanup; pruned intermediate optimizer/native files are not reverified here.',
        open_qa_judge='PENDING', human_reviews_completed=0, clinical_validation=False,
        full_stage5_complete=False, stage6='NOT_STARTED')
    put(DEST/'objective_verification_v3.json', receipt)
    print(json.dumps(dict(result='PASS',jobs=receipt['jobs'],responses=receipt['responses'])), flush=True)


if __name__ == '__main__': main()
