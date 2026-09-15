"""Evaluation-only compatibility for final markers inside a single answer block.

The frozen training/selection parser remains the v1 provenance authority.
Never search thinking text or relax structural/ambiguity checks.
"""
from dataclasses import replace
import re

from medical_posttrain.reward.parser import parse as parse_v1

VERSION = "stage5-exam-parser-v2"


def parse(text, legal_options="ABCDE", finish_reason=None):
    original = parse_v1(text, legal_options, finish_reason)
    if original.error_type != "illegal_or_empty_options":
        return original
    # v1 has already validated the complete outer control structure. Match it
    # again explicitly so the extension cannot unwrap a block inside thinking.
    block = re.fullmatch(
        r"\s*(?:<think>[\s\S]*?</think>\s*)?<answer>([^<>]*)</answer>\s*",
        text,
    )
    if block is None:
        return original
    inner = parse_v1(block[1], legal_options, finish_reason)
    if inner.answer_set is None:
        return inner
    start, end = inner.matched_span
    return replace(inner, strict_match=False, fallback_match=True,
                   valid_format=False,
                   matched_span=(block.start(1) + start, block.start(1) + end))


def rescore(response, legal_options="ABCDE"):
    """Return a new derived score; preserve raw text, IDs, tokens and v1 files."""
    parsed = parse(response["raw_output"], legal_options, response["finish_reason"])
    return dict(response, parsed_answer=parsed.answer_set,
                correct=parsed.answer_set == response["ground_truth"],
                strict_format=parsed.valid_format,
                parse_method="strict" if parsed.strict_match else "fallback" if parsed.fallback_match else "unparseable",
                ambiguous=parsed.ambiguous, unparseable=parsed.answer_set is None)
