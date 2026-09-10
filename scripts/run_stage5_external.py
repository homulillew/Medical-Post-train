#!/usr/bin/env python3
"""Auditable closed-book external API evaluation, with bounded transport retries."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, durable, now


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def redact(value, key):
    return json.loads(json.dumps(value, ensure_ascii=False).replace(key, '[REDACTED]')) if key else value


def payload(row, config):
    assert set(row) == {'id', 'task', 'messages'}
    assert [m['role'] for m in row['messages']] == ['system', 'user']
    assert all(isinstance(m['content'], str) for m in row['messages'])
    body = dict(model=config['model'], messages=row['messages'], **config['parameters'])
    assert body['tool_choice'] == 'none' and 'tools' not in body
    assert not any(k in body for k in ('web_search', 'search', 'enable_search', 'retrieval'))
    assert body['stream'] is False
    return body


def valid_response(data, requested_model):
    assert data['model'] == requested_model, 'Returned model differs from frozen identifier'
    assert len(data['choices']) == 1
    choice = data['choices'][0]
    assert not choice['message'].get('tool_calls'), 'Unexpected tool call: closed-book protocol violation'
    assert not choice['message'].get('function_call'), 'Unexpected function call'
    assert choice['finish_reason'] != 'tool_calls', 'Unexpected tool finish reason'
    assert isinstance(choice['message'].get('content'), (str, type(None)))
    return choice


def post(body, key, timeout, endpoint='https://api.deepseek.com/chat/completions'):
    assert endpoint == 'https://api.deepseek.com/chat/completions'
    req = urllib.request.Request(endpoint, data=json.dumps(body, ensure_ascii=False).encode(),
        headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'}, method='POST')
    started = time.monotonic()
    try:
        with urllib.request.build_opener(NoRedirect()).open(req, timeout=timeout) as response:
            content = response.read().decode()
            return dict(http_status=response.status, body=content.replace(key, '[REDACTED]'),
                headers={k: v for k, v in response.headers.items()
                         if k.lower() in ('date', 'x-request-id', 'request-id', 'x-ds-request-id')},
                seconds=time.monotonic() - started)
    except urllib.error.HTTPError as exc:
        return dict(http_status=exc.code, body=exc.read().decode(errors='replace').replace(key, '[REDACTED]'),
                    seconds=time.monotonic() - started)
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return dict(http_status=None, transport_error=type(exc).__name__,
                    seconds=time.monotonic() - started, billing_unknown=True)


def conservative_cost(body, config):
    # UTF-8 byte count is a conservative planning proxy, not provider billing.
    prompt_proxy = len(json.dumps(body['messages'], ensure_ascii=False).encode()) + 256
    rates = config['pricing']['peak_usd_per_million']
    return (prompt_proxy * rates['input_cache_miss'] + body['max_tokens'] * rates['output']) / 1e6


def main(out, key_file):
    out = Path(out)
    key = Path(key_file).read_text().strip()
    assert key and (Path(key_file).stat().st_mode & 0o077) == 0
    config = read(out / 'config.json')
    manifest = read(out / 'manifest.json')
    assert record(out / 'config.json') == manifest['config']
    for ref in manifest['code'] + [config['request_file'], config['evaluation_protocol']]:
        assert record(ref['path']) == ref, ref['path']
    lock = (out / 'run.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    items = [json.loads(line) for line in Path(config['request_file']['path']).open()]
    assert len(items) == config['request_count'] == len({r['id'] for r in items})
    stop = threading.Event()
    mutex = threading.Lock()
    reserved = sum(read(p)['reserved_upper_usd'] for p in out.glob('requests/*/attempts/*/reservation.json'))
    counters = dict(completed=sum(1 for _ in out.glob('requests/*/result.json')), failed=0)
    error = []

    def status(state='RUNNING'):
        with mutex:
            durable(out / 'status.json', dict(status=state, timestamp=now(), pid=os.getpid(),
                planned=len(items), **counters, reserved_upper_usd=reserved,
                no_web_search=True, no_tools=True, stage5_status='NOT_STARTED'))

    def heartbeat():
        while not stop.wait(10):
            status()
    status()
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()

    def process(i, row):
        nonlocal reserved
        folder = out / 'requests' / f'{i:04d}'
        folder.mkdir(parents=True, exist_ok=True)
        body = payload(row, config)
        request_path = folder / 'request.json'
        expected = dict(id=row['id'], task=row['task'], body=body)
        if request_path.exists():
            assert read(request_path) == expected
        else:
            immutable(request_path, expected)
        if (folder / 'result.json').exists():
            return
        # A durable200 response is authoritative even if the prior process died
        # before publishing result.json. Never resample a returned wrong answer.
        for attempt_no in range(1, config['max_transport_attempts'] + 1):
            if stop.is_set():
                return
            attempt = folder / 'attempts' / f'{attempt_no:03d}'
            raw_path = attempt / 'response.json'
            if raw_path.exists():
                response = read(raw_path)
            elif (attempt / 'reservation.json').exists():
                # Ambiguous crash boundary: reserve its full possible cost and
                # retain it as paid/unknown before the next transport attempt.
                if not (attempt / 'orphan.json').exists():
                    immutable(attempt / 'orphan.json', dict(timestamp=now(), status='UNKNOWN_TRANSPORT_OUTCOME',
                        candidate_score_unknown=True, billing_unknown=True))
                continue
            else:
                with mutex:
                    cost = conservative_cost(body, config)
                    if reserved + cost > config['conservative_reservation_cap_usd']:
                        error.append('CONSERVATIVE_BUDGET_CAP')
                        stop.set()
                        return
                    reserved += cost
                    immutable(attempt / 'reservation.json', dict(timestamp=now(),
                        id=row['id'], attempt=attempt_no, reserved_upper_usd=cost,
                        request=record(request_path), billing='UNKNOWN until response usage arrives'))
                response = post(body, key, config['timeout_seconds'], config['endpoint'])
                response['timestamp'] = now()
                immutable(raw_path, redact(response, key))
            if response['http_status'] == 200:
                # Invalid successful responses are retained and stop the run for
                # diagnosis, rather than being repeatedly sampled for a valid answer.
                data = json.loads(response['body'])
                choice = valid_response(data, config['model'])
                immutable(folder / 'result.json', dict(id=row['id'], task=row['task'],
                    attempt=attempt_no, model=data['model'], response_id=data.get('id'),
                    finish_reason=choice['finish_reason'], content=choice['message'].get('content') or '',
                    reasoning_content=choice['message'].get('reasoning_content'),
                    usage=data.get('usage'), raw_response=record(raw_path),
                    timestamp=now(), seconds=response['seconds']))
                with mutex:
                    counters['completed'] += 1
                    print(json.dumps(dict(event='completed', index=i, completed=counters['completed'],
                                          planned=len(items), timestamp=now())), flush=True)
                return
            if response['http_status'] not in (None, 408, 429, 500, 502, 503, 504):
                with mutex:
                    error.append(f'HTTP_{response["http_status"]}')
                    counters['failed'] += 1
                stop.set()
                return
            if attempt_no < config['max_transport_attempts']:
                stop.wait(min(30, 2 ** attempt_no))
        with mutex:
            counters['failed'] += 1
        immutable(folder / 'failure.json', dict(timestamp=now(), reason='TRANSPORT_RETRIES_EXHAUSTED'))

    try:
        with ThreadPoolExecutor(max_workers=config['concurrency']) as pool:
            futures = [pool.submit(process, i, row) for i, row in enumerate(items)]
            for future in as_completed(futures):
                try:
                    future.result()
                except BaseException as exc:
                    with mutex:
                        error.append(type(exc).__name__ + ':' + str(exc).replace(key, '[REDACTED]'))
                    stop.set()
        stop.set()
        thread.join()
        complete = counters['completed'] == len(items) and not error and not counters['failed']
        status('RESPONSES_COMPLETE' if complete else 'NEEDS_DIAGNOSIS')
        immutable(out / f'worker_exit_{time.time_ns()}.json', dict(timestamp=now(), complete=complete,
            errors=error, **counters, reserved_upper_usd=reserved))
        if not complete:
            raise RuntimeError('External run incomplete; inspect retained status and attempt records')
    finally:
        stop.set()
        thread.join(timeout=15)
        # Run owns this dedicated tmpfs credential copy. The key never enters an
        # artifact, process command line, environment snapshot, or request body.
        Path(key_file).unlink(missing_ok=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    p.add_argument('--key-file', required=True)
    args = p.parse_args()
    main(args.run, args.key_file)
