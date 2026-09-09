import itertools
import pytest
from medical_posttrain.reward.parser import parse, canonical_answer, reasoning_text
from medical_posttrain.reward.hybrid import reward, compute_score


@pytest.mark.parametrize("text,answer,strict", [
    ("<answer>C</answer>", "C", True), ("<answer>c</answer>", "C", True),
    ("<answer>A,C</answer>", "AC", True), ("<answer>CA</answer>", "AC", True),
    ("答案：C", "C", False), ("答案是C", "C", False), ("Answer: C", "C", False),
    ("<think>我曾考虑D。</think>答案：C", "C", False),
    ("<think></think><answer>C</answer>", "C", True),
    ("<think>A 或 B？</think><answer>C,A</answer>", "AC", True),
])
def test_parse_valid(text, answer, strict):
    result = parse(text)
    assert result.answer_set == answer and result.strict_match == strict
    assert result.valid_format == strict and result.fallback_match != strict
    assert result.matched_span is not None


@pytest.mark.parametrize("text", [
    "<answer>C</answer><answer>C</answer>", "<answer>C", "<think>C</think>",
    "<think>答案是C", "<answer>F</answer>", "<answer>AA</answer>",
    "答案是C，最终答案是D", "<answer>C</answer>答案是D", "前面选C，答案：D",
    "<think><answer>C</answer></think><answer>D</answer>",
    "<think>C</think><think>D</think><answer>C</answer>",
    "<answer>C</answer>其他文字", "分析中出现 C 和 D", "<answer></answer>",
    "<answer>C或D</answer>", "</think><answer>C</answer>",
])
def test_invalid(text):
    assert parse(text).answer_set is None


def test_lengths_and_legal_options():
    assert parse("<think>C", finish_reason="length").error_type == "truncated_thinking"
    assert parse("<answer>C", finish_reason="length").error_type == "truncated_answer"
    assert parse("<answer>E</answer>", "ABCD").answer_set is None
    assert reasoning_text("<think>abc") == "abc"
    assert parse("<answer>C</answer>", finish_reason="length").answer_set == "C"


def test_reward_separation_and_no_partial_credit():
    for sem, text, truth in itertools.product([0, .2, .9, 1], ["<answer>A</answer>", "答案是A", "<answer>AC</answer>", "<think>A"], ["A", "AC", "B"]):
        r = reward(text, truth, sem)
        assert r["acc"] in (0, 1)
        if r["acc"]:
            assert .8 <= r["score"] <= 1
        else:
            assert r["semantic_contribution"] == 0 and r["score"] <= .05
    assert reward("<answer>A</answer>", "AC", 1)["acc"] == 0
    assert reward("答案是C", "C", 1)["format"] == 0
    assert reward("<answer>CA</answer>", "AC", 1)["acc"] == 1


def test_semantic_errors_and_verl_dict():
    for value in [float("nan"), float("inf"), -.1, 1.1]:
        with pytest.raises(ValueError): reward("<answer>A</answer>", "A", value)
    with pytest.raises(ValueError): compute_score("CMExam", "A", "A")
    result = compute_score("CMExam", "A", "A", {"semantic": .4})
    assert isinstance(result, dict) and result["acc"] == 1 and result["score"] != result["acc"]
