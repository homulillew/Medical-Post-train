"""Conservative option extraction; reasoning is never an answer search region."""
from dataclasses import asdict, dataclass
import re
import unicodedata


@dataclass(frozen=True)
class ParseResult:
    answer_set: str | None = None
    strict_match: bool = False
    fallback_match: bool = False
    valid_format: bool = False
    ambiguous: bool = False
    error_type: str | None = None
    matched_span: tuple[int, int] | None = None

    def to_dict(self):
        return asdict(self)


def canonical_answer(value, legal_options="ABCDE"):
    value = unicodedata.normalize("NFKC", value).strip().upper()
    letters = re.sub(r"[\s,，、;；]+", "", value)
    if not letters or any(c not in legal_options for c in letters):
        raise ValueError("illegal_or_empty_options")
    if len(set(letters)) != len(letters):
        raise ValueError("duplicate_options")
    return "".join(sorted(letters))


def reasoning_text(text):
    match = re.search(r"<think>([\s\S]*?)</think>", text)
    if match:
        return match[1].strip()
    if text.count("<think>") == 1 and "</think>" not in text:
        return text.split("<think>", 1)[1].strip()
    return ""


def parse(text, legal_options="ABCDE", finish_reason=None):
    def fail(error, ambiguous=False):
        return ParseResult(error_type=error, ambiguous=ambiguous)

    # Reject malformed control structure, including answer blocks hidden in think.
    opening, closing = text.count("<think>"), text.count("</think>")
    if opening != closing:
        return fail("truncated_thinking" if finish_reason == "length" else "unclosed_thinking")
    if opening > 1:
        return fail("multiple_thinking_blocks", True)
    start = 0
    if opening:
        m = re.match(r"\s*<think>([\s\S]*?)</think>", text)
        if not m or "<answer" in m[1] or "</answer" in m[1]:
            return fail("invalid_thinking_structure", True)
        start = m.end()
    final = text[start:]
    if text.count("<answer>") > 1 or text.count("</answer>") > 1:
        return fail("multiple_answer_blocks", True)
    if "<answer" in final or "</answer" in final:
        m = re.fullmatch(r"\s*<answer>([^<>]*)</answer>\s*", final)
        if not m:
            if "</answer>" in final:
                return fail("trailing_or_conflicting_answer", True)
            return fail("truncated_answer" if finish_reason == "length" else "unclosed_answer")
        try:
            answer = canonical_answer(m[1], legal_options)
        except ValueError as error:
            return fail(str(error))
        return ParseResult(answer, True, False, True, False, None,
                           (start + m.start(1), start + m.end(1)))
    if "<" in final or ">" in final:
        return fail("unknown_or_malformed_tags")

    # Explicit conclusion markers only. Multiple markers are conservatively
    # ambiguous even when equal. Bare letters must constitute the whole final.
    markers = list(re.finditer(r"(?:最终答案|正确答案|答案|最终选项|Answer)\s*(?:是|为|[:：])\s*", final, re.I))
    if len(markers) > 1:
        return fail("multiple_final_conclusions", True)
    if markers:
        marker = markers[0]
        prefix = final[:marker.start()].strip()
        # A preceding explicit option conclusion also invalidates final picking.
        if re.search(r"(?:选|选择|选项|结论|应为)\s*[A-Za-z]", prefix):
            return fail("conflicting_final_region", True)
        begin = marker.end()
        tail = final[begin:]
    else:
        begin, tail = 0, final
    m = re.fullmatch(r"\s*([A-Za-z\s,，、;；]+)[。.!！]?\s*", tail)
    if not m:
        return fail("ambiguous_final_region" if markers else "no_final_answer", bool(markers))
    try:
        answer = canonical_answer(m[1], legal_options)
    except ValueError as error:
        return fail(str(error))
    return ParseResult(answer, False, True, False, False, None,
                       (start + begin + m.start(1), start + begin + m.end(1)))
