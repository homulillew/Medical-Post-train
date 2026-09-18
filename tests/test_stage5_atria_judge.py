import json
import pytest
from run_stage5_atria_judge import MODEL, DIMENSIONS, extract, normalize
from medical_posttrain.evaluation.atria import JudgeResponseError, decode_response


def response(output, status='completed'):
    return dict(model=MODEL, status=status, output=output)


def message(text):
    return dict(type='message', role='assistant', content=[dict(type='output_text', text=text)])


def test_only_visible_json_is_parsed():
    assert extract(response([dict(type='reasoning', text='Not a judgment'), message('{"value":1}')])) == {'value': 1}


@pytest.mark.parametrize('value', [
    response([message('{}')], 'incomplete'),
    response([dict(type='web_search_call'), message('{}')]),
    response([dict(type='message', role='assistant', content=[dict(type='refusal')])]),
    response([message('```json\n{}\n```')]),
])
def test_invalid_outputs_are_not_repaired(value):
    with pytest.raises(JudgeResponseError):
        extract(value)


def judgment():
    return dict(preference='tie', scores={s: {d: 3 for d in DIMENSIONS} for s in ['A', 'B']},
                rationale='Synthetic fixture only.', critical_safety_flags=[])


def test_safety_evidence_must_quote_actual_answer():
    item=dict(pair_id='fixture', A='Actual answer A', B='Actual answer B')
    value=judgment()
    assert normalize(value, item, {})['judge_is_candidate'] is False
    value['scores']['A']['safety_escalation']=0
    with pytest.raises(JudgeResponseError):normalize(value, item, {})
    flag=dict(side='A', type='dangerous_instruction', response_quote='fabricated quotation', rationale='Fixture')
    value['critical_safety_flags']=[flag]
    with pytest.raises(JudgeResponseError):normalize(value, item, {})
    flag['response_quote']='Actual answer A'
    assert len(normalize(value, item, {})['critical_safety_evidence']) == 1


def test_fractional_scores_are_rejected():
    value=judgment();value['scores']['A']['factual_correctness']=2.5
    with pytest.raises(ValueError):normalize(value, dict(pair_id='fixture', A='A', B='B'), {})


@pytest.mark.parametrize('text,code', [
    ('/0'*100, 'degenerate_repetition'), ('', 'empty_visible_output'),
    ('=`**: user--- broken response', 'non_json_output'), ('{"{"" :""}', 'malformed_judgment_json'),
    ('{"a":1,"a":2}', 'duplicate_json_key'), ('{"a":NaN}', 'invalid_json_constant'),
    ('{"a":"\ufffd"}', 'damaged_unicode'), ('[]', 'invalid_judgment_schema'),
])
def test_failure_categories(text, code):
    with pytest.raises(JudgeResponseError) as exc:extract(response([message(text)]))
    assert exc.value.code == code


def test_reasoning_never_substitutes_for_missing_answer():
    with pytest.raises(JudgeResponseError) as exc:
        extract(response([dict(type='reasoning', text=json.dumps(judgment()))]))
    assert exc.value.code == 'empty_visible_output'


def test_alias_is_lossless_recorded_and_does_not_mutate_input():
    value=judgment();item=dict(pair_id='fixture',A='A',B='B')
    for side in ['A','B']:value['scores'][side]['safety_safety']=value['scores'][side].pop('safety_escalation')
    original=json.dumps(value,sort_keys=True)
    j=normalize(value,item,{})
    assert len(j['parser_normalization']['changes']) == 2
    assert j['scores'] == judgment()['scores']
    assert json.dumps(value,sort_keys=True) == original
    with pytest.raises(JudgeResponseError):normalize(value,item,{},allow_known_alias=False)


def test_ambiguous_alias_and_missing_scores_stay_invalid():
    item=dict(pair_id='fixture',A='A',B='B');value=judgment()
    value['scores']['A']['safety_safety']=3
    with pytest.raises(JudgeResponseError):normalize(value,item,{})
    del value['scores']['A']['safety_escalation'];del value['scores']['A']['factual_correctness']
    with pytest.raises(JudgeResponseError):normalize(value,item,{})


def test_undecodable_bytes_are_preserved_but_not_scored():
    import base64
    from run_stage5_atria_judge import capture_response
    raw=b'\xffsecret';wrapped=capture_response(raw,200,1,{'Content-Type':'application/json','Set-Cookie':'private'},'secret')
    assert wrapped['body'] is None and wrapped['credential_redacted']
    assert base64.b64decode(wrapped['body_base64']) == b'\xff[REDACTED]'
    assert 'Set-Cookie' not in wrapped['headers']
    with pytest.raises(JudgeResponseError) as exc:decode_response(raw)
    assert exc.value.code == 'invalid_utf8'


def test_explicit_message_boundaries_and_tools_disabled():
    from run_stage5_atria_judge import build_request
    item=dict(pair_id='fixture',question='中文',A='A',B='B')
    p=dict(model=MODEL,prompt='Frozen rubric',parameters=dict(tools=[],tool_choice='none'))
    body=build_request(p,item)
    assert body['tools']==[] and body['tool_choice']=='none'
    assert [x['role'] for x in body['input']]==['system','user']
    assert json.loads(body['input'][1]['content'][0]['text'])==item
    assert 'instructions' not in body


def test_resume_replays_saved_response_without_network(tmp_path):
    from run_stage5_atria_judge import restore_result, capture_response
    from medical_posttrain.evaluation.core import freeze
    item=dict(pair_id='fixture',A='A',B='B');dest=tmp_path/'saved';dest.mkdir()
    freeze(dest/'request.json',dict(pair_id='fixture'))
    raw=json.dumps(response([message(json.dumps(judgment()))])).encode()
    freeze(dest/'raw_response.json',capture_response(raw,200,1,{},'fake-secret'))
    assert restore_result(dest,item)=='VALID'
    assert restore_result(dest,item)=='VALID'
    assert (dest/'judgment.json').exists()


def test_resume_skips_invalid_but_blocks_uncertain_request(tmp_path):
    from run_stage5_atria_judge import restore_result, capture_response
    from medical_posttrain.evaluation.core import freeze
    item=dict(pair_id='fixture',A='A',B='B');dest=tmp_path/'saved';dest.mkdir()
    freeze(dest/'request.json',dict(pair_id='fixture'))
    with pytest.raises(RuntimeError,match='Uncertain'):restore_result(dest,item)
    raw=json.dumps(response([message('/0'*100)])).encode()
    freeze(dest/'raw_response.json',capture_response(raw,200,1,{},'fake-secret'))
    assert restore_result(dest,item)=='INVALID'
    assert restore_result(dest,item)=='INVALID'


def test_progress_counts_valid_invalid_and_all_usage(tmp_path):
    from run_stage5_atria_judge import publish_progress, capture_response
    from medical_posttrain.evaluation.core import freeze,read
    for i,kind in enumerate(['judgment.json','invalid.json']):
        dest=tmp_path/'out'/'requests'/str(i)
        freeze(dest/'request.json',{})
        freeze(dest/kind,{})
        freeze(dest/'raw_response.json',capture_response(json.dumps(dict(usage=dict(input_tokens=3,output_tokens=2,total_tokens=5))).encode(),200,1,{},'fake-secret'))
    publish_progress(tmp_path/'out',tmp_path/'index',1347,current_index=4)
    s=read(tmp_path/'index'/'status.json')
    assert s['completed']==1 and s['invalid_outputs']==1 and s['attempted']==2
    assert s['usage']['total_tokens']==10
