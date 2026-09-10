import json
from pathlib import Path
import sys

import pytest
sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
import run_stage5_external as runner
from medical_posttrain.rl.common import durable, read, record


def setup(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    key = tmp_path / 'key'
    key.write_text('unit-test-credential')
    key.chmod(0o600)
    row = dict(id='item1', task='exam', messages=[dict(role='system', content='no tools'), dict(role='user', content='question')])
    requests = tmp_path / 'requests.jsonl'
    requests.write_text(json.dumps(row) + '\n')
    protocol = tmp_path / 'protocol.json'
    durable(protocol, {})
    cfg = dict(model='deepseek-flash', endpoint='https://api.deepseek.com/chat/completions',
        parameters=dict(thinking={'type':'enabled'}, reasoning_effort='high', max_tokens=8192, tool_choice='none', stream=False),
        request_file=record(requests), evaluation_protocol=record(protocol), request_count=1,
        max_transport_attempts=3, timeout_seconds=180, concurrency=1, conservative_reservation_cap_usd=15,
        pricing=dict(peak_usd_per_million=dict(input_cache_miss=.3, output=1.2)))
    durable(out / 'config.json', cfg)
    durable(out / 'manifest.json', dict(config=record(out / 'config.json'), code=[]))
    return out, key, row, cfg


def response(content='A', tools=None):
    msg = dict(role='assistant', content=content)
    if tools:
        msg['tool_calls'] = tools
    body = dict(model='deepseek-flash', id='r1', choices=[dict(message=msg, finish_reason='stop')],
                usage=dict(prompt_tokens=10, completion_tokens=1, total_tokens=11))
    return dict(http_status=200, body=json.dumps(body), seconds=.1, timestamp='test')


def test_payload_has_no_search_tools_or_gold(tmp_path):
    _, _, row, cfg = setup(tmp_path)
    body = runner.payload(row, cfg)
    assert body['tool_choice'] == 'none' and 'tools' not in body
    assert not any('search' in k for k in body)
    row['answer'] = 'B'
    with pytest.raises(AssertionError):
        runner.payload(row, cfg)


def test_valid_wrong_response_is_never_resampled(tmp_path, monkeypatch):
    out, key, _, _ = setup(tmp_path)
    calls = []
    monkeypatch.setattr(runner, 'post', lambda *a: calls.append(1) or response())
    runner.main(out, key)
    assert len(calls) == 1
    assert read(out / 'requests/0000/result.json')['content'] == 'A'
    assert read(out / 'status.json')['status'] == 'RESPONSES_COMPLETE'
    assert not key.exists()


def test_orphan_complete_response_is_adopted_without_another_api_call(tmp_path, monkeypatch):
    out, key, row, cfg = setup(tmp_path)
    f = out / 'requests/0000'
    durable(f / 'request.json', dict(id=row['id'], task=row['task'], body=runner.payload(row, cfg)))
    durable(f / 'attempts/001/reservation.json', dict(reserved_upper_usd=.01))
    durable(f / 'attempts/001/response.json', response())
    def forbidden(*args):
        raise AssertionError('Must not repeat paid response')
    monkeypatch.setattr(runner, 'post', forbidden)
    runner.main(out, key)
    assert read(f / 'result.json')['attempt'] == 1


def test_unexpected_tool_response_stops_and_is_retained(tmp_path, monkeypatch):
    out, key, _, _ = setup(tmp_path)
    monkeypatch.setattr(runner, 'post', lambda *a: response(tools=[dict(type='function')]))
    with pytest.raises(RuntimeError):
        runner.main(out, key)
    assert (out / 'requests/0000/attempts/001/response.json').exists()
    assert not (out / 'requests/0000/result.json').exists()
    assert read(out / 'status.json')['status'] == 'NEEDS_DIAGNOSIS'


def test_auth_error_is_not_retried(tmp_path, monkeypatch):
    out, key, _, _ = setup(tmp_path)
    calls = []
    monkeypatch.setattr(runner, 'post', lambda *a: calls.append(1) or dict(http_status=401, body='invalid credential', seconds=.1))
    with pytest.raises(RuntimeError):
        runner.main(out, key)
    assert len(calls) == 1


def test_budget_guard_prevents_any_request(tmp_path, monkeypatch):
    out, key, _, cfg = setup(tmp_path)
    cfg['conservative_reservation_cap_usd'] = 0
    durable(out / 'config.json', cfg)
    durable(out / 'manifest.json', dict(config=record(out / 'config.json'), code=[]))
    monkeypatch.setattr(runner, 'post', lambda *a: pytest.fail('budget exhausted'))
    with pytest.raises(RuntimeError):
        runner.main(out, key)
    assert not list(out.glob('requests/*/attempts/*/reservation.json'))
