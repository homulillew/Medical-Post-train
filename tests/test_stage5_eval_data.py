import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
from stage5_eval_data import balanced, request, score_exam, safety_tags, retention_quality_reason


def test_balanced_2000_and_nested_280_are_order_invariant():
    rows = [dict(id=f'{s}:{i}', stratum=str(s)) for s in range(28) for i in range(100)]
    a, quotas = balanced(rows, 2000, 'test', lambda r: r['stratum'])
    b, _ = balanced(list(reversed(rows)), 2000, 'test', lambda r: r['stratum'])
    assert a == b
    assert sorted(quotas.values()) == [71] * 16 + [72] * 12
    probe, quotas = balanced(a, 280, 'probe', lambda r: r['stratum'])
    assert set(quotas.values()) == {10}
    assert {r['id'] for r in probe} <= {r['id'] for r in a}


def test_small_strata_redistribute_without_duplication():
    rows = [dict(id=str(i), stratum='small' if i < 2 else 'big') for i in range(20)]
    chosen, quotas = balanced(rows, 10, 'small', lambda r: r['stratum'])
    assert quotas == {'small': 2, 'big': 8}
    assert len({r['id'] for r in chosen}) == 10


def exam():
    return dict(id='x', task='exam', question='示例问题', options={'A': '选项一', 'B': '选项二'},
                answer='B', reference_explanation='GOLD_REFERENCE_DO_NOT_EXPORT', scorable=True)


def test_request_does_not_export_gold_or_reference():
    row = exam()
    assert set(request(row)) == {'id', 'task', 'messages'}
    assert 'GOLD_REFERENCE_DO_NOT_EXPORT' not in str(request(row))
    row['answer'] = 'A'
    row['reference_explanation'] = 'changed'
    assert request(row) == request(exam())


def test_clinical_prompt_excludes_reference_and_previous_answers():
    row = dict(id='c:q0', task='open_qa', question='问诊问题', context='病例原始描述',
               reference_answer='SECRET_REFERENCE', previous_reference_answers=['SECRET_HISTORY'])
    assert 'SECRET' not in str(request(row))
    assert '病例原始描述' in request(row)['messages'][1]['content']


def test_common_parser_and_conflicting_answers():
    row = exam()
    assert score_exam('最终答案：B', row)['correct']
    assert score_exam('<think>分析</think><answer>B</answer>', row)['correct']
    assert not score_exam('<answer>A</answer><answer>B</answer>', row)['correct']
    assert not score_exam('<think>最终答案：B', row, 'length')['correct']
    assert not score_exam('最终答案：BB', row)['correct']


def test_safety_tags_ignore_reference_answer():
    row = dict(question='一般问题', context='', reference_answer='休克孕妇用药')
    assert not safety_tags(row)
    row['question'] = '孕妇出现呼吸困难'
    tags = {r['tag'] for r in safety_tags(row)}
    assert 'emergency_red_flag' in tags and 'pregnancy_pediatrics_elderly' in tags


def test_cmb_six_option_alphabet_is_scored_without_dropping_f():
    row = exam()
    row['options'] = {k: k + '内容' for k in 'ABCDEF'}
    row['answer'] = 'AF'
    assert score_exam('最终答案：FA', row)['correct']
    assert not score_exam('最终答案：AG', row)['correct']


def test_negated_history_and_insulin_like_biomarker_are_not_positive_risk_tags():
    row = dict(question='诊断依据是什么？', context='否认药物过敏史。无明显胸闷、胸痛。胰岛素样生长因子升高。否认长期服药史。')
    assert safety_tags(row) == []
    row['context'] += '现突发胸痛，服用阿司匹林。'
    assert {t['tag'] for t in safety_tags(row)} == {'emergency_red_flag', 'medication_dose_contraindication'}


def test_open_qa_excludes_choice_and_unavailable_visual_sources():
    assert retention_quality_reason('诊断是什么？\nA. 答案一\nB. 答案二') == 'choice_format_not_open_qa'
    assert retention_quality_reason('请根据提供的化学结构式识别该药物。') == 'unavailable_visual_or_attachment_reference'
    assert retention_quality_reason('根据以下症状分析原因：发热三天。') is None


def test_simile_is_not_pregnancy_context():
    assert not safety_tags(dict(question='肚子变大好像怀孕的妇女一样。', context=''))
