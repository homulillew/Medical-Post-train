#!/usr/bin/env python3
"""Read-only API usage: summarize the existing outputs after a user budget stop."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np
from stage5_open_qa_judge import validate_judgment,DIMENSIONS
from medical_posttrain.rl.common import read,record,immutable,now


def summarize(out):
    stopped=read(out/'user_stop.json');assert stopped['status']=='PARTIAL_USER_STOPPED_BUDGET'
    cfg=read(out/'config.json');manifest=read(out/'manifest.json')
    for ref in manifest['code']+[manifest['config'],manifest['items'],cfg['request_file'],cfg['evaluation_protocol']]:
        assert record(ref['path'])==ref
    items={r['anonymous_id']:r for r in map(json.loads,(out/'items.jsonl').open())}
    requests=list(map(json.loads,(out/'judge_requests.jsonl').open()))
    scored=[];invalid=[];tokens=Counter();cost=0.0
    for p in sorted(out.glob('requests/*/result.json')):
        r=read(p);item=items[r['id']];req=requests[int(p.parent.name)]
        assert record(r['raw_response']['path'])==r['raw_response']
        raw_response=read(r['raw_response']['path']);raw=json.loads(raw_response['body'])
        assert raw_response['http_status']==200 and len(raw['choices'])==1
        assert raw['model']==r['model']==cfg['model'] and req['id']==r['id']
        assert read(p.parent/'request.json')==dict(id=req['id'],task=req['task'],body=dict(model=cfg['model'],messages=req['messages'],**cfg['parameters']))
        choice=raw['choices'][0];assert not choice['message'].get('tool_calls') and not choice['message'].get('function_call')
        assert choice['message'].get('content')==r['content'] and choice['finish_reason']==r['finish_reason']
        u=raw['usage'];assert u==r['usage'] and u['total_tokens']==u['prompt_tokens']+u['completion_tokens']
        assert u['prompt_cache_hit_tokens']+u['prompt_cache_miss_tokens']==u['prompt_tokens']
        tokens.update({k:u[k] for k in ('prompt_tokens','completion_tokens','prompt_cache_hit_tokens','prompt_cache_miss_tokens')})
        tokens['reasoning_tokens']+=u.get('completion_tokens_details',{}).get('reasoning_tokens',0)
        t=datetime.fromisoformat(read(p.parent/'attempts'/f'{r["attempt"]:03d}'/'reservation.json')['timestamp'])
        peak=t.weekday()<5 and (1<=t.hour<4 or 6<=t.hour<10)
        prices=cfg['pricing']['peak_usd_per_million' if peak else 'off_peak_usd_per_million']
        cost+=(u['prompt_cache_hit_tokens']*prices['input_cache_hit']+u['prompt_cache_miss_tokens']*prices['input_cache_miss']+u['completion_tokens']*prices['output'])/1e6
        try:
            assert r['finish_reason']=='stop'
            v=validate_judgment(json.loads(r['content']),item['payload']['candidate_answer'],item['context'])
            scored.append(dict(id=item['id'],anonymous_id=r['id'],track=item['track'],case_id=item['case_id'],question=item['payload']['question'],candidate_response=item['candidate_response'],raw_judge_response=record(p),scores={k:v['score'] for k,v in v['scores'].items()},judgment=v,human_audit='PENDING'))
        except (AssertionError,ValueError,KeyError,TypeError) as e:invalid.append(dict(id=item['id'],reason=str(e),response=record(p)))
    assert len(scored)+len(invalid)==stopped['completed_results']
    rng=np.random.default_rng(20260910);tracks={}
    for track in ('cmb_clin','open_qa_sota_probe_100'):
        subset=[r for r in scored if r['track']==track];clusters=sorted({r['case_id'] or r['id'] for r in subset});dimensions={}
        for dim in DIMENSIONS:
            values=[r['scores'][dim] for r in subset]
            if subset:
                sums=np.array([sum(r['scores'][dim] for r in subset if (r['case_id'] or r['id'])==c) for c in clusters])
                counts=np.array([sum((r['case_id'] or r['id'])==c for r in subset) for c in clusters])
                picks=rng.integers(0,len(clusters),(10000,len(clusters)));boot=sums[picks].sum(axis=1)/counts[picks].sum(axis=1)
                interval=list(map(float,np.quantile(boot,[.025,.975])))
            else:interval=None
            dimensions[dim]=dict(mean=float(np.mean(values)) if values else None,n=len(values),distribution={str(i):values.count(i) for i in range(5)},conditional_cluster_bootstrap_95=interval)
        tracks[track]=dict(scored=len(subset),observed_clusters=len(clusters),dimensions=dimensions,potential_critical_flags=sum(bool(r['judgment']['critical_flags']) for r in subset))
    unknown=sum(read(Path(p)/'reservation.json')['reserved_upper_usd'] for p in stopped['inflight_unknown_outcomes'])
    summary=dict(run_id=out.name,status='PARTIAL_USER_STOPPED_BUDGET',timestamp=now(),judge=cfg['model'],judge_documented_version=cfg['provider_documented_version'],candidate_items_planned=308,judge_requests_planned=306,
        completed_responses=len(scored)+len(invalid),valid_judgments=len(scored),invalid_judgments=invalid,tracks=tracks,tokens=dict(tokens),known_returned_usage_cost_usd=cost,
        unknown_inflight_requests=len(stopped['inflight_unknown_outcomes']),unknown_inflight_conservative_reservation_usd=unknown,
        cost_note='Returned-usage cost excludes unknown in-flight billing and separate connectivity control; reservation is a planning guard, not an invoice.',
        human_audit='PENDING_REAL_HUMAN',stage5_status='NOT_STARTED',no_web_search=True,
        interpretation='Non-random partial completion subset, all scores conditional on observed outputs. Does not estimate full308-item quality. Human judgment and same-provider bias remain unresolved.',
        empty_answer_scope='The original308 responses include two known empty answers. Neither has an API judgment; they are not inserted into this completed-judge subset mean.',
        bootstrap=dict(seed=20260910,replicates=10000,unit='clinical case',limitation='Conditional descriptive interval only; non-random truncation and judge bias are not covered.'),source=record(__file__))
    with (out/'partial_scored_judgments.jsonl').open('x') as f:
        for r in scored:f.write(json.dumps(r,ensure_ascii=False)+'\n')
    summary['scored_judgments']=record(out/'partial_scored_judgments.jsonl')
    index=Path(__file__).resolve().parents[1]/'experiments/stage5/open_qa_judgments'/out.name
    cases=[r for r in scored if any(v<3 for v in r['scores'].values()) or r['judgment']['critical_flags']]
    for base in (out,index):
        immutable(base/'partial_summary.json',summary);immutable(base/'partial_cases.json',dict(selection='All observed scores below3 in any dimension or potential critical flags',cases=cases))
    immutable(index/'partial_verification.json',dict(result='PASS_FOR_OBSERVED_SUBSET_ONLY',timestamp=now(),completed=len(scored)+len(invalid),valid=len(scored),full_run_complete=False,human_audit='PENDING',summary=record(out/'partial_summary.json')))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':summarize(Path(sys.argv[1]))
