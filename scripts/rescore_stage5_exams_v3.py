#!/usr/bin/env python3
"""CPU-only, additive rescoring of immutable Stage5 exam generations.

--watch follows the existing generator and scores each completed job once.
Original v1 scores remain intact and all checkpoints use the same v3 parser.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from medical_posttrain.evaluation.core import read, rows, ref, check_ref, freeze, digest, score, summarize, sliced
from medical_posttrain.evaluation.exam_parser_v3 import VERSION, rescore
from medical_posttrain.rl.common import durable
from stage5_objective_evidence import roles, comparison_table, put, EXAMS

OUT = Path("/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1")
INDEX = ROOT / "experiments/stage5/parser_correction_v3"
DERIVED = OUT / "rescored_v3"


def now():
    return datetime.now(timezone.utc).isoformat()


def score_job(directory, destination, partial=False):
    specs = sorted(directory.glob("execution_spec_*.json"))
    assert specs, f"No generation spec: {directory}"
    spec = read(specs[-1])
    check_ref(spec["items"])
    items = rows(spec["items"]["path"])
    assert len({i['id'] for i in items}) == len(items)
    complete = directory / "complete.json"
    assert partial or complete.exists()
    new, old, provenance, v2 = [], [], [], []
    from medical_posttrain.evaluation.exam_parser_v2 import rescore as rescore_v2
    errors = Counter()
    from medical_posttrain.reward.parser import parse as parse_v1
    for item in items:
        d = directory / digest(item["id"])[:24]
        if not (d / "receipt.json").exists():
            assert partial, f"Missing receipt: {d}"
            continue
        receipt = read(d / "receipt.json")
        check_ref(receipt["response"])
        check_ref(receipt["raw"])
        assert Path(receipt["response"]["path"]) == d / "response.json"
        assert Path(receipt["raw"]["path"]) == d / f"raw_attempt_{receipt['attempts']:03d}.json"
        raw, response = read(receipt["raw"]["path"]), read(d / "response.json")
        assert raw['adapter_sha256'] == spec['adapter']['sha256']
        assert response == score(raw, item, spec["checkpoint_id"], spec["run_id"])
        old.append(response)
        new.append(rescore(response, "".join(item["options"])))
        v2.append(rescore_v2(response, "".join(item["options"])))
        errors[parse_v1(response['raw_output'], ''.join(item['options']), response['finish_reason']).error_type or 'parsed'] += 1
        provenance.append(dict(prompt_id=item['id'], receipt=ref(d/'receipt.json'), **receipt))
    assert new, "No committed responses yet"
    if not partial:
        meta = read(complete)
        assert meta['n'] == len(items) == len(new)
        assert meta['prompt_ids'] == [r['prompt_id'] for r in new]
        assert meta['response_refs'] == [r['response'] for r in provenance]
        assert meta['summary'] == summarize(old)
    destination.mkdir(parents=True, exist_ok=True)
    predictions = destination / "predictions.jsonl"
    with predictions.open("x") as f:
        for row in new:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    freeze(destination/'provenance.json', provenance)
    result = dict(parser_version=VERSION, status='PARTIAL' if partial else 'COMPLETE',
                  dataset=spec['dataset'], checkpoint_id=spec['checkpoint_id'], n=len(new),
                  expected_n=len(items), original_summary=summarize(old), summary=summarize(new),
                  v2_summary=summarize(v2),
                  new_error_counts=dict(Counter(r['parser_error'] or 'parsed' for r in new)),
                  previously_parsed_answers_changed=sum(a['parsed_answer'] is not None and a['parsed_answer'] != b['parsed_answer'] for a,b in zip(v2,new)),
                  changed_answers=sum(a['parsed_answer'] != b['parsed_answer'] for a,b in zip(old,new)),
                  original_error_counts=dict(errors), predictions=ref(predictions),
                  provenance=ref(destination/'provenance.json'), generation_spec=ref(specs[-1]),
                  original_complete=ref(complete) if not partial else None,
                  new_generations=0, selection_changed=False,
                  integrity_scope='receipt/raw hashes, adapter identity, v1 score replay; full token replay remains in original objective verifier')
    freeze(destination/'summary.json', result)
    return result


def final_tables():
    primary, endpoints = roles()
    for dataset, name in zip(EXAMS, ['cmexam_final_results_v3.json', 'cmb_final_results_v3.json']):
        if (INDEX/name).exists(): continue
        if not all((DERIVED/cid.replace(':','_')/dataset/'summary.json').exists() for cid in set(primary.values()) | set(endpoints.values())): continue
        data = {cid: rows(DERIVED/cid.replace(':','_')/dataset/'predictions.jsonl')
                for cid in set(primary.values()) | set(endpoints.values())}
        if dataset == EXAMS[0]:
            membership=read(ROOT/'experiments/stage5/cmexam_slice_manifest_v1.json')
            groups=membership['difficulty']
            secondary_ids=read(ROOT/'experiments/stage5/manifests/cmexam_test_clean.json')['ids']
            slices={cid:dict(difficulty=sliced(v,groups), categories={k:sliced(v,g) for k,g in membership['categories'].items()}) for cid,v in data.items()}
        else:
            groups=read(ROOT/'experiments/stage5/cmb_primary_audit_v1.json')['category_membership']
            secondary_ids=read(ROOT/'experiments/stage5/cmb_medical_only_manifest_v1.json')['ids']
            slices={cid:sliced(v,groups) for cid,v in data.items()}
        secondary_set=set(secondary_ids)
        filtered={cid:[r for r in v if r['prompt_id'] in secondary_set] for cid,v in data.items()}
        paired_slices={k:comparison_table({cid:[r for r in v if r['prompt_id'] in set(ids)] for cid,v in data.items()},primary)['paired'] for k,ids in groups.items() if ids}
        put(INDEX/name, dict(parser_version=VERSION,dataset=dataset,
            primary=comparison_table(data,primary),scientific_endpoints=comparison_table(data,endpoints),
            secondary=dict(n=len(secondary_ids),primary=comparison_table(filtered,primary),scientific_endpoints=comparison_table(filtered,endpoints)),
            slices=slices,paired_slices=paired_slices,seed=20260914,resamples=10000,
            interpretation='Post-observation parser bug correction authorized by owner; report alongside archived v1 results.',
            selection_manifest=ref(ROOT/'experiments/stage5/selection_result_manifest_v1.json')))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--watch', action='store_true')
    args=ap.parse_args()
    INDEX.mkdir(parents=True,exist_ok=True)
    with (INDEX/'rescore.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        # Pin correction code, decision and selection before reading new scores.
        sources=read(INDEX/'source_manifest.json')
        for source in sources['sources']: check_ref(source)
        primary,endpoints=roles()
        jobs=[(cid,dataset) for dataset in EXAMS for cid in sorted(set(primary.values())|set(endpoints.values()))]
        try:
            while True:
                completed=[]
                for cid,dataset in jobs:
                    directory=OUT/'final'/cid.replace(':','_')/dataset
                    dest=DERIVED/cid.replace(':','_')/dataset
                    if not (directory/'complete.json').exists(): continue
                    if not (dest/'summary.json').exists():
                        result=score_job(directory,dest)
                        put(INDEX/f"{cid.replace(':','_')}_{dataset}.json",result)
                        print(json.dumps(dict(event='rescored',checkpoint=cid,dataset=dataset,summary=result['summary']),ensure_ascii=False),flush=True)
                    saved=read(dest/'summary.json')
                    check_ref(saved['predictions']);check_ref(saved['original_complete'])
                    completed.append(ref(dest/'summary.json'))
                all_done=len(completed)==len(jobs)
                final_tables()
                durable(INDEX/'status.json',dict(status='EXAM_RESCORING_COMPLETE' if all_done else 'WAITING_FOR_GENERATIONS',
                    timestamp=now(),completed_jobs=len(completed),expected_jobs=len(jobs),results=completed,
                    parser_version=VERSION,new_generations=0,full_stage5_complete=False))
                if all_done or not args.watch: break
                time.sleep(30)
        except BaseException:
            failure=dict(status='FAILED',timestamp=now(),traceback=traceback.format_exc())
            freeze(INDEX/f'failure_{time.time_ns()}.json',failure)
            durable(INDEX/'status.json',failure)
            raise


if __name__ == '__main__':
    main()
