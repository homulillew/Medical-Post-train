#!/usr/bin/env python3
"""Verify/scoring for the external probe; open QA requires independent judging."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, now
from medical_posttrain.reward.parser import parse
from stage5_eval_data import request


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p = k / n
    center = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return [center-half, center+half]


def analyze(out):
    cfg = read(out / 'config.json')
    run_manifest = read(out / 'manifest.json')
    for ref in [run_manifest['config'], run_manifest['source_archive'],
                cfg['evaluation_protocol'], *run_manifest['code']]:
        assert record(ref['path']) == ref
    assert read(out / 'status.json')['status'] == 'RESPONSES_COMPLETE'
    freeze = read(cfg['dataset_freeze']['path'])
    assert record(cfg['dataset_freeze']['path']) == cfg['dataset_freeze']
    expected = [json.loads(l) for l in Path(cfg['request_file']['path']).open()]
    assert record(cfg['request_file']['path']) == cfg['request_file']
    protocol = read(cfg['evaluation_protocol']['path'])
    assert record(protocol['parser']['path']) == protocol['parser']
    data_root = Path(freeze['artifact_root'])
    mapping = {}
    for name in ('cmexam_sota_probe_512', 'cmb_sota_probe_280', 'cmb_clin', 'open_qa_sota_probe_100'):
        manifest = read(data_root / 'manifests' / (name + '.json'))
        assert record(manifest['data']['path']) == manifest['data']
        for line in Path(manifest['data']['path']).open():
            row = json.loads(line)
            assert row['id'] not in mapping
            mapping[row['id']] = name, row
    assert len(mapping) == len(expected) == 1100
    assert len({r['id'] for r in expected}) == 1100
    assert len(list(out.glob('requests/*/result.json'))) == 1100
    totals = defaultdict(Counter)
    details = []
    slices = defaultdict(lambda: defaultdict(Counter))
    token_counts = Counter()
    prices = cfg['pricing']
    cost = 0.0
    missing_usage = 0
    latencies = []
    for i, item in enumerate(expected):
        folder = out / 'requests' / f'{i:04d}'
        saved = read(folder / 'request.json')
        result = read(folder / 'result.json')
        name, row = mapping[item['id']]
        assert request(row) == item
        assert saved == dict(id=item['id'], task=item['task'], body=dict(model=cfg['model'], messages=item['messages'], **cfg['parameters']))
        assert result['id'] == item['id'] and result['model'] == cfg['model']
        assert record(result['raw_response']['path']) == result['raw_response']
        response = read(result['raw_response']['path'])
        raw = json.loads(response['body'])
        assert response['http_status'] == 200 and len(raw['choices']) == 1
        assert result['model'] == raw['model'] and result['response_id'] == raw.get('id')
        choice = raw['choices'][0]
        assert result['finish_reason'] == choice['finish_reason']
        assert not choice['message'].get('tool_calls') and not choice['message'].get('function_call')
        assert result['content'] == (choice['message'].get('content') or '')
        assert result['reasoning_content'] == choice['message'].get('reasoning_content')
        assert result['usage'] == raw.get('usage')
        totals[name]['completed'] += 1
        totals[name]['length_limited'] += result['finish_reason'] == 'length'
        totals[name]['empty_visible_answers'] += not bool(result['content'].strip())
        latencies.append(result['seconds'])
        detail = dict(id=item['id'], track=name, response=record(folder / 'result.json'),
            content=result['content'], finish_reason=result['finish_reason'], usage=result['usage'])
        if row['task'] == 'exam':
            parsed = parse(result['content'], ''.join(row['options']), result['finish_reason'])
            correct = parsed.answer_set == row['answer']
            totals[name]['correct'] += correct
            totals[name]['parse_success'] += parsed.answer_set is not None
            detail.update(gold=row['answer'], parsed=parsed.to_dict(), correct=correct)
            keys = ['Difficulty level', 'Clinical Department'] if name.startswith('cmexam') else ['exam_class']
            for key in keys:
                bucket = slices[name][key + ':' + str(row['strata'][key])]
                bucket['n'] += 1
                bucket['correct'] += correct
        else:
            detail.update(quality_score=None, quality_status='PENDING_INDEPENDENT_JUDGE_AND_HUMAN_AUDIT',
                          case_id=row.get('case_id'))
        details.append(detail)
        usage = result['usage']
        if usage is None:
            missing_usage += 1
            continue
        assert usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']
        hit = usage.get('prompt_cache_hit_tokens', usage.get('prompt_tokens_details', {}).get('cached_tokens', 0))
        miss = usage.get('prompt_cache_miss_tokens', usage['prompt_tokens'] - hit)
        assert hit + miss == usage['prompt_tokens']
        reservation = read(folder / 'attempts' / f'{result["attempt"]:03d}' / 'reservation.json')
        t = datetime.fromisoformat(reservation['timestamp'])
        peak = t.weekday() < 5 and (1 <= t.hour < 4 or 6 <= t.hour < 10)
        rates = prices['peak_usd_per_million' if peak else 'off_peak_usd_per_million']
        cost += (hit*rates['input_cache_hit'] + miss*rates['input_cache_miss'] + usage['completion_tokens']*rates['output']) / 1e6
        token_counts.update(prompt_tokens=usage['prompt_tokens'], completion_tokens=usage['completion_tokens'],
            prompt_cache_hit_tokens=hit, prompt_cache_miss_tokens=miss,
            reasoning_tokens=usage.get('completion_tokens_details', {}).get('reasoning_tokens', 0))
    tracks = {}
    for name, counts in totals.items():
        tracks[name] = dict(counts)
        if 'correct' in counts:
            tracks[name].update(accuracy=counts['correct']/counts['completed'],
                parse_success_rate=counts['parse_success']/counts['completed'],
                accuracy_wilson_95=wilson(counts['correct'], counts['completed']))
        else:
            tracks[name]['quality_evaluation'] = 'PENDING_INDEPENDENT_JUDGE_AND_HUMAN_AUDIT'
    attempts = list(out.glob('requests/*/attempts/*/reservation.json'))
    unknown = list(out.glob('requests/*/attempts/*/orphan.json'))
    error_responses = []
    for path in out.glob('requests/*/attempts/*/response.json'):
        r = read(path)
        if r['http_status'] != 200:
            error_responses.append(dict(response=record(path), http_status=r['http_status']))
    assert all(t['completed'] == n for t, n in zip(
        [tracks[k] for k in ('cmexam_sota_probe_512', 'cmb_sota_probe_280', 'cmb_clin', 'open_qa_sota_probe_100')],
        [512, 280, 208, 100]))
    with (out / 'scored_predictions.jsonl').open('x') as f:
        for detail in details:
            f.write(json.dumps(detail, ensure_ascii=False) + '\n')
    summary = dict(result='PASS', scope='External probe generation and exact-answer exam scoring only',
        timestamp=now(), run_id=out.name, model=cfg['model'], provider_documented_version=cfg['provider_documented_version'],
        tracks=tracks, tokens=dict(token_counts), missing_usage_responses=missing_usage,
        successful_response_calculated_cost_usd=cost,
        cost_scope='Published prices applied to returned usage; excludes separately retained connectivity control and unknown billed transport outcomes. Not an account invoice.',
        physical_request_attempts=len(attempts), transport_error_responses=error_responses,
        unknown_outcomes=len(unknown), no_tools_verified=True, no_web_search_enabled=True,
        expected_responses=1100, completed_responses=len(details),
        mean_request_latency_seconds=sum(latencies)/len(latencies),
        category_and_difficulty_slices={k: {b: dict(v) for b, v in d.items()} for k, d in slices.items()},
        predictions=record(out / 'scored_predictions.jsonl'), config=record(out / 'config.json'),
        analyzer=record(__file__), parser=protocol['parser'], stage5_status='NOT_STARTED',
        caveats=['External API contextual reference; not a controlled model-scale/training comparison.',
            'Probe accuracy applies to frozen stratified subsets; compare local models on identical IDs later.',
            'Open-ended response generation is complete; medical quality and safety have not been judged.',
            'Provider documented version is pinned as a documentation snapshot, not a verifiable weight hash.'])
    immutable(out / 'summary.json', summary)
    index = ROOT / 'experiments/stage5/external_baselines' / out.name
    immutable(index / 'summary.json', summary)
    immutable(index / 'config.json', cfg)
    immutable(index / 'verification.json', dict(result='PASS', timestamp=now(), summary=record(out / 'summary.json'),
        responses=1100, source_request_identity='PASS', raw_response_identity='PASS',
        parser_reference=protocol['parser'], no_tool_calls=True, open_qa_judgment='PENDING'))
    print(json.dumps({k: v for k, v in summary.items() if k != 'category_and_difficulty_slices'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    analyze(Path(a.run))
