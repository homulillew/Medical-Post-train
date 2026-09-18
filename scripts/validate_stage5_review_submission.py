#!/usr/bin/env python3
"""Validate submitted blind reviews without making any medical judgments.

This checks schema/coverage only. Reviewer declarations do not prove expertise,
actual human participation or independent medical correctness.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze
from medical_posttrain.evaluation.blind import validate_judgment


def validate_submission(judgments, reviews, public, schedule, initially_required):
    expected={r['pair_id'] for r in public}
    mapping={e['pair_id']:e['prompt_id'] for e in schedule['entries']}
    errors=[]; valid={};seen=set();critical=set()
    for j in judgments:
        pid=j.get('pair_id')
        try:
            assert pid in expected, 'unknown pair_id'
            assert pid not in seen, 'duplicate pair_id'
            seen.add(pid)
            validate_judgment(j)
            assert isinstance(j.get('rationale'),str) and j['rationale'].strip(), 'rationale required'
            flags=j.get('critical_safety_flags')
            assert isinstance(flags,list) and all(isinstance(x,str) and x.strip() for x in flags), 'explicit safety flag list required (empty allowed)'
            assert j.get('judge_is_candidate') is False, 'judge independence declaration required'
            assert isinstance(j.get('raw_judgment'),dict), 'raw judgment artifact reference required'
            check_ref(j['raw_judgment'])
            if flags:critical.add(mapping[pid.removesuffix('-flip')])
            valid[pid]=j
        except (AssertionError,ValueError,KeyError,TypeError,OSError) as exc:
            errors.append(dict(pair_id=pid,reason=str(exc)))
    needed=set(initially_required)|critical
    required_pairs={e['pair_id'] for e in schedule['entries'] if e['prompt_id'] in needed}
    audited=set();seen_reviews=set()
    for r in reviews:
        pid=r.get('pair_id')
        try:
            assert pid in mapping and pid not in seen_reviews, 'unknown/duplicate human pair_id'
            seen_reviews.add(pid)
            assert r.get('reviewed_in_full') is True, 'full human review required'
            assert isinstance(r.get('reviewer_id'),str) and r['reviewer_id'].strip(), 'reviewer identity required'
            assert isinstance(r.get('observation'),str) and r['observation'].strip(), 'review observation required'
            assert type(r.get('is_clinician')) is bool, 'qualification disclosure required'
            if not r['is_clinician']:
                assert r.get('disclaimer')=='NON_CLINICIAN_REVIEW_NOT_CLINICAL_VALIDATION', 'non-clinician disclaimer required'
            assert type(r.get('disagreement')) is bool, 'explicit disagreement boolean required'
            if r['disagreement']:
                assert r.get('resolution') and r.get('adjudicator_id'), 'unresolved disagreement'
            audited.add(pid)
        except (AssertionError,ValueError,KeyError,TypeError) as exc:
            errors.append(dict(pair_id=pid,reason=str(exc),kind='human_audit'))
    missing=sorted(expected-set(valid));missing_reviews=sorted(required_pairs-audited)
    return dict(status='INVALID' if errors else 'INCOMPLETE' if missing or missing_reviews else 'SCHEMA_AND_COVERAGE_PASS',
        expected_judgments=len(expected),valid_judgments=len(valid),missing_judgments=missing,
        required_human_items=len(needed),required_human_pairs=len(required_pairs),
        valid_human_pairs=len(audited),missing_human_pairs=missing_reviews,errors=errors,
        new_critical_prompt_ids=sorted(critical),full_stage5_complete=False,clinical_validation=False,
        remaining_checks=['Judge identity/independence provenance','Attributable safety judgments for all 333 responses',
            'Position consistency','Clinical case-cluster statistics','Full Stage5 acceptance'])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--judgments',type=Path,required=True)
    ap.add_argument('--human-reviews',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    manifest=read(ROOT/'experiments/stage5/closure_v3_20260918/review_packet.json')
    for name in ['judge_packet.jsonl','human_audit_packet.jsonl']:check_ref(manifest['public_files'][name])
    result=validate_submission(rows(args.judgments),rows(args.human_reviews) if args.human_reviews else [],
        rows(manifest['public_files']['judge_packet.jsonl']['path']),
        read(ROOT/'experiments/stage5/open_qa_blind_schedule_v1.json'),manifest['private_required_prompt_ids'])
    result['input_judgments']=ref(args.judgments)
    result['input_reviews']=ref(args.human_reviews) if args.human_reviews else None
    result['validator']=ref(__file__)
    freeze(args.output,result)
    print(json.dumps({k:result[k] for k in ['status','expected_judgments','valid_judgments','required_human_items','valid_human_pairs']}))
    return 0 if result['status']=='SCHEMA_AND_COVERAGE_PASS' else 2


if __name__=='__main__':sys.exit(main())
