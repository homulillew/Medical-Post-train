import pytest
from medical_posttrain.evaluation.exam_parser_v3 import parse, rescore


@pytest.mark.parametrize('body,answer', [
    ('正确答案是C。\n最终答案：C', 'C'),
    ('正确答案是：\n最终答案：A', 'A'),
    ('答案：A、C。最终答案：CA', 'AC'),
    ('正确答案是D 2个月。', 'D'),
    ('正确答案是B，第二鳃弓。', 'B'),
    ('答案是E TSH。', 'E'),
    ('答案是D CT。', 'D'),
    ('讨论选项A和E。最终答案：C', 'C'),
    ('选项A不符合，选项C符合。最终答案：C', 'C'),
    ('说明。即选项A。', 'A'),
    ('因此选择A。', 'A'),
    ('C 某选项内容', 'C'),
    ('最终答案：ＡＣ', 'AC'),
    ('最终答案：**C**', 'C'),
    ('数值>0.20，且<0.30。最终答案：D', 'D'),
    ('答案是A、B和C。最终答案：ABC', 'ABC'),
    ('正确答案是A，因为B不合适。', 'A'),
    ('不应选择B。最终答案：A', 'A'),
])
def test_extract_visible_conclusion(body,answer):
    result=parse('<think>最终答案：E</think><answer>'+body+'</answer>')
    assert result.answer_set==answer
    assert result.fallback_match and not result.strict_match


@pytest.mark.parametrize('body', [
    '答案：A。最终答案：C', '选择A。最终答案：C',
    '最终答案：A或C', '最终答案：A/C', '最终答案：A、F',
    '最终答案：A，但也可能B。', '最终答案：AA', '最终答案：Z',
    '最终答案：A，但不是A。',
    '无法确定最敏感的病原体。', '选项A与选项B的描述不同。',
    '不应选择A。', '答案是：我无法确定。最终答案：A',
])
def test_reject_ambiguous_or_missing_answer(body):
    assert parse('<answer>'+body+'</answer>').answer_set is None


@pytest.mark.parametrize('text', [
    '<think>最终答案：A', '<think>最终答案：A</think>',
    '<think><answer>A</answer></think><answer>A</answer>',
    '<answer>答案：A', '<answer>A</answer><answer>A</answer>',
    '<answer>A</answer>答案：B', '<answer><foo>A</foo></answer>',
    '<answer><foo A</answer>', '<think>x</think><think>y</think><answer>A</answer>',
])
def test_structure_and_hidden_reasoning(text):
    assert parse(text,finish_reason='length').answer_set is None


def test_strict_format_legal_options_and_gold_independence():
    assert parse('<answer>AC</answer>').strict_match
    assert parse('<answer>答案：C</answer>', 'ABDE').answer_set is None
    row=dict(raw_output='<answer>最终答案：C</answer>',finish_reason='stop',ground_truth='A',prompt_id='p',output_tokens=10)
    a=rescore(row);b=rescore(dict(row,ground_truth='C'))
    assert a['parsed_answer']==b['parsed_answer']=='C'
    assert not a['correct'] and b['correct']
    assert a['raw_output']==row['raw_output'] and a['output_tokens']==10


def test_additive_raw_replay(tmp_path):
    from test_stage5_parser_v2 import fixture_job
    from rescore_stage5_exams_v3 import score_job
    from medical_posttrain.evaluation.core import ref
    directory,d=fixture_job(tmp_path)
    originals={p:ref(p) for p in directory.rglob('*.json')}
    result=score_job(directory,tmp_path/'v3')
    assert result['summary']['correct_count']==1
    assert result['parser_version']=='stage5-exam-parser-v3'
    assert originals=={p:ref(p) for p in originals}
