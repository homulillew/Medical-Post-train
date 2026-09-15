import pytest

from medical_posttrain.evaluation.exam_parser_v2 import parse, rescore
from medical_posttrain.reward.parser import parse as parse_v1


@pytest.mark.parametrize("body,answer", [
    ("最终答案：C", "C"), ("最终答案：BD", "BD"),
    ("正确答案为 A、C。", "AC"), ("Answer: D", "D"),
    ("分析完毕。\n最终答案：C", "C"),
])
def test_marker_inside_answer(body, answer):
    text = "<think>答案：E，不作为最终答案。</think>\n<answer>" + body + "</answer>"
    result = parse(text)
    assert result.answer_set == answer
    assert result.fallback_match and not result.valid_format
    assert parse_v1(text).answer_set is None
    start, end = result.matched_span
    assert answer[0] in text[start:end]


@pytest.mark.parametrize("text", [
    "<answer>最终答案：A，最终答案：C</answer>",
    "<answer>最终答案：A，最终答案：A</answer>",
    "<answer>选择A。最终答案：C</answer>",
    "<answer>最终答案：Z</answer>",
    "<answer>最终答案：AA</answer>",
    "<answer>最终答案：A或C</answer>",
    "<answer>最终答案：A</answer>另外答案C",
    "<answer>最终答案：A</answer><answer>C</answer>",
    "<answer>最终答案：A",
    "<think><answer>最终答案：A</answer></think>",
    "<think>最终答案：A</think>",
    "<think>分析<answer>最终答案：A</answer>",
    "<answer><answer>最终答案：A</answer></answer>",
    "<answer>分析A很好</answer>",
])
def test_reject_ambiguity_and_malformed_structure(text):
    assert parse(text, finish_reason="length").answer_set is None


@pytest.mark.parametrize("text", [
    "<think>分析</think><answer>AC</answer>", "<answer>B</answer>",
    "最终答案：D", "C", "<think>未闭合", "<answer>AA</answer>",
])
def test_legacy_behavior_unchanged(text):
    assert parse(text) == parse_v1(text)


def test_legal_options_and_no_mutation():
    assert parse("<answer>最终答案：C</answer>", "ABDE").answer_set is None
    old = dict(raw_output="<answer>最终答案：C</answer>", finish_reason="stop",
               ground_truth="C", correct=False, parsed_answer=None,
               output_tokens=10, prompt_id="example")
    new = rescore(old)
    assert new["correct"] and not old["correct"]
    assert new["raw_output"] == old["raw_output"]
    assert new["output_tokens"] == old["output_tokens"]
    assert new["prompt_id"] == old["prompt_id"]


def fixture_job(tmp_path):
    from medical_posttrain.evaluation.core import freeze, ref, score, digest, summarize
    item = dict(id='example', options={'A':'one','C':'two'}, answer='C')
    items = tmp_path/'items.jsonl'
    import json
    items.write_text(json.dumps(item)+'\n')
    directory=tmp_path/'job';directory.mkdir()
    freeze(directory/'execution_spec_001.json',dict(items=ref(items), checkpoint_id='sft',
        run_id='test', dataset='test', adapter=dict(sha256='adapter')))
    d=directory/digest('example')[:24];d.mkdir()
    raw=dict(raw_output='<answer>最终答案：C</answer>',finish_reason='stop',
             prompt_tokens=10,output_tokens=8,adapter_sha256='adapter')
    response=score(raw,item,'sft','test')
    freeze(d/'raw_attempt_001.json',raw);freeze(d/'response.json',response)
    freeze(d/'receipt.json',dict(raw=ref(d/'raw_attempt_001.json'),response=ref(d/'response.json'),attempts=1))
    freeze(directory/'complete.json',dict(n=1,prompt_ids=['example'],response_refs=[ref(d/'response.json')],summary=summarize([response])))
    return directory,d


def test_completed_job_replay_preserves_original(tmp_path):
    from rescore_stage5_exams_v2 import score_job
    from medical_posttrain.evaluation.core import ref
    directory,d=fixture_job(tmp_path)
    before={p:ref(p) for p in directory.rglob('*.json')}
    result=score_job(directory,tmp_path/'new')
    assert result['summary']['correct_count']==1
    assert result['original_summary']['correct_count']==0
    assert result['changed_answers']==1 and result['new_generations']==0
    assert before=={p:ref(p) for p in before}


def test_replay_rejects_modified_raw(tmp_path):
    from rescore_stage5_exams_v2 import score_job
    directory,d=fixture_job(tmp_path)
    p=d/'raw_attempt_001.json'
    p.write_text(p.read_text().replace('答案：C','答案：A'))
    with pytest.raises(ValueError,match='hash/size'):
        score_job(directory,tmp_path/'new')


def test_replay_never_overwrites_derived_result(tmp_path):
    from rescore_stage5_exams_v2 import score_job
    directory,d=fixture_job(tmp_path)
    score_job(directory,tmp_path/'new')
    with pytest.raises(FileExistsError):
        score_job(directory,tmp_path/'new')
