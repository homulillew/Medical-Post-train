"""Versioned binary correctness with strictly gated semantic shaping."""
import math
from .parser import canonical_answer, parse

WEIGHTS = {"acc": 0.8, "sem": 0.15, "format": 0.05}


def reward(text, ground_truth, semantic, legal_options="ABCDE", finish_reason=None):
    if not math.isfinite(semantic) or not 0 <= semantic <= 1:
        raise ValueError("Semantic score must be finite and clipped before reward")
    parsed = parse(text, legal_options, finish_reason)
    truth = canonical_answer(ground_truth, legal_options)
    acc = int(parsed.answer_set == truth)
    fmt = int(parsed.valid_format)
    contribution = WEIGHTS["sem"] * acc * semantic
    return dict(score=WEIGHTS["acc"] * acc + contribution + WEIGHTS["format"] * fmt,
                acc=acc, sem=semantic, format=fmt, semantic_contribution=contribution,
                parse_status="strict" if parsed.strict_match else "fallback" if parsed.fallback_match else parsed.error_type,
                parser=parsed.to_dict())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """verl NaiveRewardManager-compatible dict, never scalar-as-accuracy.

    Reward-side caller must supply a score from the frozen semantic encoder.
    Missing/failed embeddings are errors, never quietly replaced with zero.
    """
    if extra_info is None or "semantic" not in extra_info:
        raise ValueError("Missing frozen reward-side semantic evidence")
    result = reward(solution_str, ground_truth, extra_info["semantic"],
                    extra_info.get("legal_options", "ABCDE"), extra_info.get("finish_reason"))
    return {k: result[k] for k in ("score", "acc", "sem", "format", "semantic_contribution", "parse_status")}
