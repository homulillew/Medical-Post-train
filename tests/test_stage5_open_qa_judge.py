import copy
from pathlib import Path
import sys

import pytest
sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from stage5_open_qa_judge import visible, validate_judgment, DIMENSIONS, system_prompt


def judgment():
    return dict(scores={d:dict(score=3,rationale='具体依据') for d in DIMENSIONS},critical_flags=[],
                reference_assessment=dict(status='CONSISTENT',rationale='参考可辅助'))


def test_hidden_reasoning_is_not_judged():
    assert visible('<think>secret</think> visible')=='visible'
    assert visible('<think>unfinished')==''
    assert visible('plain answer')=='plain answer'


@pytest.mark.parametrize('score',[True,5,-1,2.5])
def test_invalid_score_is_rejected(score):
    value=judgment();value['scores']['factual_correctness']['score']=score
    with pytest.raises(AssertionError):validate_judgment(value,'answer','source')


def test_critical_flag_requires_real_quote():
    value=judgment();value['critical_flags']=[dict(type='dangerous_instruction',response_quote='invented',source_quote='',rationale='why')]
    with pytest.raises(AssertionError):validate_judgment(value,'answer','source')
    value['critical_flags'][0]['response_quote']='answer'
    assert validate_judgment(value,'answer','source') is value


def test_safety_zero_without_critical_evidence_is_rejected():
    value=judgment();value['scores']['safety_escalation']['score']=0
    with pytest.raises(AssertionError):validate_judgment(value,'answer','source')


def test_judge_prompt_does_not_disclose_candidate_identity():
    prompt=system_prompt().lower()
    assert 'deepseek' not in prompt and 'flash' not in prompt
    assert 'json' in prompt and '内部推理' in prompt
