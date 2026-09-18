import json
import pytest
from run_stage5_atria_judge import MODEL, DIMENSIONS, extract, normalize


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
    with pytest.raises((AssertionError, json.JSONDecodeError)):
        extract(value)


def judgment():
    return dict(preference='tie', scores={s: {d: 3 for d in DIMENSIONS} for s in ['A', 'B']},
                rationale='Synthetic fixture only.', critical_safety_flags=[])


def test_safety_evidence_must_quote_actual_answer():
    item=dict(pair_id='fixture', A='Actual answer A', B='Actual answer B')
    value=judgment()
    assert normalize(value, item, {})['judge_is_candidate'] is False
    value['scores']['A']['safety_escalation']=0
    with pytest.raises(AssertionError):normalize(value, item, {})
    flag=dict(side='A', type='dangerous_instruction', response_quote='fabricated quotation', rationale='Fixture')
    value['critical_safety_flags']=[flag]
    with pytest.raises(AssertionError):normalize(value, item, {})
    flag['response_quote']='Actual answer A'
    assert len(normalize(value, item, {})['critical_safety_evidence']) == 1


def test_fractional_scores_are_rejected():
    value=judgment();value['scores']['A']['factual_correctness']=2.5
    with pytest.raises(ValueError):normalize(value, dict(pair_id='fixture', A='A', B='B'), {})
