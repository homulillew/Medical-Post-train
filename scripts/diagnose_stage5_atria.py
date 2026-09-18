#!/usr/bin/env python3
"""Replay frozen Atria responses; optionally run at most three authorized probes."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import urllib.request
import urllib.error

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze
from medical_posttrain.evaluation import atria
from medical_posttrain.evaluation.atria import decode_response,extract,normalize,JudgeResponseError
from medical_posttrain.rl.common import durable
import run_stage5_atria_judge as runner
from validate_stage5_review_submission import validate_submission

BASE=Path('/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1/atria_judge')
HISTORY=ROOT/'experiments/stage5/atria_judge/pilot_20260918_summary/request_diagnostics.json'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--live',action='store_true');args=ap.parse_args()
    run_id='s5_atria_parser_v2_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=BASE/run_id;out.mkdir()
    index=ROOT/'experiments/stage5/atria_judge'/run_id;index.mkdir()
    previous=read(BASE/'s5_atria_dawn_20260918T081417Z/protocol.json')
    check_ref(previous['packet']);check_ref(previous['rubric'])
    public=rows(previous['packet']['path']);items={r['pair_id']:r for r in public}
    protocol=dict(run_id=run_id,run_class='DIAGNOSTIC',stage=5,
        start_time=datetime.now(timezone.utc).isoformat(),command=[sys.executable,*sys.argv],
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()),
        environment=dict(python=platform.python_version(),host=platform.node(),gpu='not used',client='stdlib urllib'),
        source=ref(__file__),runner=ref(runner.__file__),parser=ref(atria.__file__),
        frozen_history=ref(HISTORY),packet=previous['packet'],rubric=previous['rubric'],
        request_protocol=dict(model=previous['model'],prompt=previous['prompt'],parameters=previous['parameters']),
        historical_input_selection=[1,4,5],max_api_requests=3 if args.live else 0,
        hypothesis='Explicit system/user message boundaries may avoid provider instructions-field conversion issues. Unconfirmed.',
        selection_reason='Previously repetitive, non-JSON and empty visible responses; exclude all new judgments from evaluation totals.',
        contract='No retries, no score imputation, no replacement of historical judgments. Parser alias version is disclosed.',
        live_validation_authorization='Owner explicitly allowed at most 3 additional calls' if args.live else None)
    freeze(out/'protocol.json',protocol)
    for name,path in [('source.py',__file__),('runner_source.py',runner.__file__),('parser_source.py',atria.__file__)]:
        (out/name).write_bytes(Path(path).read_bytes())
    freeze(index/'run.json',dict(run_id=run_id,protocol=ref(out/'protocol.json'),artifact_root=str(out)))
    diagnostics=[];judgments=[];unchanged=0
    for old in read(HISTORY):
        check_ref(old['raw']);wrapper=read(old['raw']['path']);raw_bytes=wrapper['body'].encode('utf-8')
        row=dict(run_id=old['run_id'],pair_id=old['pair_id'],original_valid=old['valid'],raw=old['raw'])
        try:
            response=decode_response(raw_bytes)
            j=normalize(extract(response),items[old['pair_id']],old['raw'])
            if old['valid']:
                assert j==read(Path(old['raw']['path']).parent/'judgment.json'), 'Previously valid judgment changed'
                unchanged+=1
            judgments.append(j)
            row.update(status='COMPAT_NORMALIZED' if 'parser_normalization' in j else 'VALID',normalization=j.get('parser_normalization'))
        except JudgeResponseError as exc:
            assert not old['valid'], 'Regression in a previously valid judgment'
            row.update(status='INVALID',failure_code=exc.code,reason=str(exc))
        diagnostics.append(row)
    assert len({j['pair_id'] for j in judgments})==len(judgments)
    with (out/'reparsed_judgments.jsonl').open('x') as f:
        for j in judgments:f.write(json.dumps(j,ensure_ascii=False)+'\n')
    manifest=read(ROOT/'experiments/stage5/closure_v3_20260918/review_packet.json')
    receipt=validate_submission(judgments,[],public,read(ROOT/'experiments/stage5/open_qa_blind_schedule_v1.json'),manifest['private_required_prompt_ids'])
    assert not receipt['errors'] and receipt['status']=='INCOMPLETE'
    freeze(out/'submission_validation.json',receipt)
    replay=dict(requests=len(diagnostics),unique_entries=20,previous_valid_unchanged=unchanged,
        compatible_valid=len(judgments),newly_normalized=len(judgments)-unchanged,
        unique_entries_without_valid_judgment=20-len(judgments),
        failure_counts=dict(Counter(d['failure_code'] for d in diagnostics if d['status']=='INVALID')),
        diagnostics=diagnostics,judgments=ref(out/'reparsed_judgments.jsonl'),validation=ref(out/'submission_validation.json'),
        retrospective_parser_change=True,paid_requests=0,original_artifacts_unchanged=True)
    freeze(out/'replay.json',replay);freeze(index/'replay.json',replay)
    print(json.dumps(dict(run_id=run_id,replay_valid=len(judgments),unchanged=unchanged,live_limit=protocol['max_api_requests'])),flush=True)
    live=[];usage=Counter()
    if args.live:
        key=runner.credential()
        for i in protocol['historical_input_selection']:
            item=public[i];dest=out/'live'/f'{i:04d}';dest.mkdir(parents=True)
            body=runner.build_request(protocol['request_protocol'],item)
            freeze(dest/'request.json',dict(pair_id=item['pair_id'],endpoint=runner.ENDPOINT,body=body,timestamp=datetime.now(timezone.utc).isoformat()))
            req=urllib.request.Request(runner.ENDPOINT,data=json.dumps(body,ensure_ascii=False).encode('utf-8'),
                headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'},method='POST')
            start=time.time();result=dict(pair_id=item['pair_id'],index=i,request=ref(dest/'request.json'),included_in_evaluation=False)
            try:
                try:
                    with urllib.request.urlopen(req,timeout=180) as response:
                        status=response.status;raw=response.read();headers=response.headers
                except urllib.error.HTTPError as exc:
                    status=exc.code;raw=exc.read();headers=exc.headers
                freeze(dest/'raw_response.json',runner.capture_response(raw,status,time.time()-start,headers,key))
                result.update(http_status=status,raw=ref(dest/'raw_response.json'))
                if status!=200:raise JudgeResponseError('http_error',f'HTTP {status}')
                response=decode_response(raw)
                result['usage']=response.get('usage',{})
                usage.update({k:v for k,v in result['usage'].items() if type(v) is int})
                judgment=normalize(extract(response),item,result['raw'])
                freeze(dest/'diagnostic_judgment.json',judgment)
                result.update(status='VALID',normalization=judgment.get('parser_normalization'))
            except JudgeResponseError as exc:
                result.update(status='INVALID',failure_code=exc.code,reason=str(exc).replace(key,'[REDACTED]'))
            except (OSError,ValueError) as exc:
                result.update(status='TRANSPORT_ERROR',reason=str(exc).replace(key,'[REDACTED]'))
            freeze(dest/'result.json',result);live.append(result)
            durable(index/'status.json',dict(status='RUNNING',attempted=len(live),valid=sum(r['status']=='VALID' for r in live),max_requests=3))
            print(json.dumps(result,ensure_ascii=False),flush=True)
            if result['status']=='TRANSPORT_ERROR' or result.get('failure_code')=='http_error':break
    summary=dict(run_id=run_id,status='DIAGNOSTIC_COMPLETE',end_time=datetime.now(timezone.utc).isoformat(),
        replay=ref(out/'replay.json'),live_requested=args.live,live_requests=len(live),live_valid=sum(r['status']=='VALID' for r in live),
        live_results=live,live_usage=dict(usage),currency_cost=None,
        scientific_conclusion='Client parser and diagnostic checks validated; upstream root cause remains unconfirmed.',
        no_automatic_retry=True,full_stage5_complete=False,human_reviews_completed=0)
    freeze(out/'summary.json',summary);freeze(index/'summary.json',summary);durable(index/'status.json',summary)
    artifacts=[ref(p) for p in sorted(out.rglob('*')) if p.is_file()]
    freeze(index/'artifacts_manifest.json',dict(artifacts=artifacts))
    print(json.dumps(dict(run_id=run_id,live_requests=len(live),live_valid=summary['live_valid'],live_usage=dict(usage))),flush=True)


if __name__=='__main__':main()
