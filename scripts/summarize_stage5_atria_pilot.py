#!/usr/bin/env python3
"""Describe a completed bounded API pilot, without making full-set claims."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'), str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read, rows, ref, check_ref, freeze
from medical_posttrain.evaluation.blind import DIMENSIONS
from validate_stage5_review_submission import validate_submission


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--runs', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True)
    judgments=[];attempted=set();usage=Counter();runs=[];requests=0;origins={}
    artifacts=[];diagnostics=[]
    for run in args.runs:
        protocol=read(run/'protocol.json');check_ref(protocol['packet']);check_ref(protocol['rubric'])
        assert ref(run/'runner_source.py')['sha256']==protocol['source']['sha256']
        run_usage=Counter();n=0;valid=0
        for path in sorted(run.glob('requests/*/request.json')):
            request=read(path);attempted.add(request['pair_id']);n+=1;requests+=1
            raw_path=path.parent/'raw_response.json'
            if raw_path.exists():
                wrapper=read(raw_path);raw=json.loads(wrapper['body'])
                run_usage.update({k:v for k,v in raw.get('usage',{}).items() if type(v) is int})
                text=''.join(c.get('text','') for o in raw.get('output',[]) if o.get('type')=='message' for c in o.get('content',[]) if c.get('type')=='output_text')
                diagnostics.append(dict(run_id=protocol['run_id'],pair_id=request['pair_id'],
                    http_status=wrapper['http_status'],response_status=raw.get('status'),
                    valid=(path.parent/'judgment.json').exists(),seconds=wrapper['seconds'],
                    response_id=raw.get('id'),request_id=raw.get('request_id'),
                    usage=raw.get('usage'),visible_characters=len(text),
                    invalid_visible_excerpt=text[:240] if not (path.parent/'judgment.json').exists() else None,
                    raw=ref(raw_path)))
            judge_path=path.parent/'judgment.json'
            if judge_path.exists():
                j=read(judge_path);assert j['pair_id'] not in origins
                judgments.append(j);origins[j['pair_id']]=protocol['run_id'];valid+=1
        usage.update(run_usage)
        runs.append(dict(run_id=protocol['run_id'],requests=n,valid_judgments=valid,
                         usage=dict(run_usage),parameters=protocol['parameters'],protocol=ref(run/'protocol.json')))
        artifacts.extend(ref(p) for p in sorted(run.rglob('*')) if p.is_file() and p.name!='execution.lock')
    manifest=read(ROOT/'experiments/stage5/closure_v3_20260918/review_packet.json')
    check_ref(manifest['public_files']['judge_packet.jsonl'])
    public=rows(manifest['public_files']['judge_packet.jsonl']['path'])
    authorized={p['pair_id'] for p in public[:20]}
    assert attempted<=authorized and set(origins)<=attempted
    judgments.sort(key=lambda j:next(i for i,p in enumerate(public[:20]) if p['pair_id']==j['pair_id']))
    with (out/'judgments.jsonl').open('x') as f:
        for j in judgments:f.write(json.dumps(j,ensure_ascii=False)+'\n')
    schedule=read(ROOT/'experiments/stage5/open_qa_blind_schedule_v1.json')
    receipt=validate_submission(judgments,[],public,schedule,manifest['private_required_prompt_ids'])
    assert receipt['status']=='INCOMPLETE' and not receipt['errors'] and receipt['valid_judgments']==len(judgments)
    freeze(out/'pilot_validation.json',receipt)
    mapping={e['pair_id']:e for e in schedule['entries']}
    by_id={j['pair_id']:j for j in judgments}
    scores=defaultdict(lambda:defaultdict(list));comparisons=defaultdict(Counter)
    flips=[];unique=set();cases=set();base=0
    inverse={'A win':'B win','B win':'A win','tie':'tie'}
    for j in judgments:
        pid=j['pair_id'];entry=mapping[pid.removesuffix('-flip')]
        unique.add(entry['prompt_id']);cases.add(entry['prompt_id'].rsplit(':q',1)[0])
        if pid.endswith('-flip'):
            original=by_id.get(pid.removesuffix('-flip'))
            if original is None:continue
            flips.append(dict(pair_id=pid,consistent=j['preference']==inverse[original['preference']],
                              original=original['preference'],flipped=j['preference']))
            continue
        base+=1
        models=entry['models'];label=origins[pid]+': '+' vs '.join(sorted(models))
        winner='tie' if j['preference']=='tie' else models[0 if j['preference']=='A win' else 1]
        comparisons[label][winner]+=1
        for side,model in zip(['A','B'],models):
            for dim in DIMENSIONS:scores[origins[pid]+': '+model][dim].append(j['scores'][side][dim])
    # A model can appear twice per question; retain that dependence in the report.
    result=dict(scope='INTEGRATION_PILOT_ONLY',
        judgments=len(judgments),attempted_unique_entries=len(attempted),authorized_entries=20,
        api_judge_requests=requests,invalid_attempts=requests-len(judgments),
        untested_entries=sorted(authorized-attempted),entries_without_valid_judgments=sorted(attempted-set(origins)),
        runs=runs,judgment_origins=origins,
        unique_questions_with_valid_judgments=len(unique),clinical_cases_with_valid_judgments=len(cases),base_comparisons=base,
        position_flip_comparisons=len(flips),position_consistency=flips,
        pair_preferences={k:dict(v) for k,v in comparisons.items()},
        model_scores={m:{d:dict(n=len(v),sum=sum(v),mean=sum(v)/len(v)) for d,v in ds.items()} for m,ds in scores.items()},
        score_unit='0–4 rubric, repeated answer appearances; nonrepresentative frozen prefix; no inferential statistics',
        critical_flagged_pairs=[j['pair_id'] for j in judgments if j['critical_safety_flags']],
        usage=dict(usage),currency_cost=None,validation=ref(out/'pilot_validation.json'),
        source_judgments=ref(out/'judgments.jsonl'),human_reviews_completed=0,full_stage5_complete=False,
        safety_note='API flags are pending real human adjudication, not clinical findings.')
    freeze(out/'pilot_analysis.json',result)
    freeze(out/'request_diagnostics.json',diagnostics)
    freeze(out/'artifacts_manifest.json',dict(artifacts=artifacts,analysis_source=ref(__file__)))
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':main()
