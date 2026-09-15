"""Visible-answer extraction, independent of gold labels and model identity.

Repeated equal conclusions are one answer. Descriptions after a letter are not
format errors. Different explicit conclusions remain ambiguous; never pick the
last answer or search hidden thinking. Frozen training/selection code is separate.
"""
import re
import unicodedata
from medical_posttrain.reward.parser import ParseResult, canonical_answer, parse as legacy_parse

VERSION = 'stage5-exam-parser-v3'
MARKER = re.compile(r'(?:最终答案|正确答案|答案|最终选项|Answer)\s*(?:(?:应该|应当|应)?(?:是|为)|[:：])\s*[:：]?\s*', re.I)
CHOICE = re.compile(r'(?:因此选择|所以选择|应当选择|应该选择|应选择|因此选|所以选|故选|应当选|应该选|应选|选择|即(?:为)?选项)\s*[:：]?\s*', re.I)


def fail(reason, ambiguous=False):
    return ParseResult(error_type=reason, ambiguous=ambiguous)


def visible_region(text, finish_reason):
    if text.count('<think>') != text.count('</think>'):
        return None, fail('truncated_thinking' if finish_reason == 'length' else 'unclosed_thinking')
    if text.count('<think>') > 1:
        return None, fail('multiple_thinking_blocks', True)
    if '<think>' in text:
        block = re.match(r'\s*<think>([\s\S]*?)</think>', text)
        if not block or '<answer' in block[1] or '</answer' in block[1]:
            return None, fail('invalid_thinking_structure', True)
        text = text[block.end():]
    if text.count('<answer>') > 1 or text.count('</answer>') > 1:
        return None, fail('multiple_answer_blocks', True)
    if '<answer' in text or '</answer' in text:
        block = re.fullmatch(r'\s*<answer>([\s\S]*?)</answer>\s*', text)
        if not block:
            return None, fail('invalid_answer_structure', True)
        text = block[1]
    # Numeric inequalities are content, while unknown/nested control tags fail.
    if re.search(r'<\s*/?\s*[A-Za-z!]', text, re.I):
        return None, fail('unknown_or_malformed_tags')
    return text, None


def option_prefix(text, legal_options):
    """Extract an explicit initial option set; never infer from option semantics."""
    text = text.strip().lstrip('*').strip()
    match = re.match(r'([A-Za-z]+)(?![A-Za-z])', text)
    if not match:
        return None, 'missing_explicit_option'
    value = match[1].upper()
    end = match.end()
    while True:
        more = re.match(r'(?:\s*[,，、;；和及与]\s*|\s+)([A-Za-z]+)(?![A-Za-z])', text[end:])
        if not more:
            break
        token = more[1].upper()
        # Whitespace followed by e.g. CT/TSH is description, not another option.
        if text[end:][0].isspace() and not set(token) <= set(legal_options):
            break
        value += token
        end += more.end()
    try:
        answer = canonical_answer(value, legal_options)
    except ValueError as exc:
        return None, str(exc)
    tail = text[end:]
    if re.match(r'\s*(?:或者?|还是|或许|也可能(?:是)?|也许(?:是)?|/|、)\s*[A-Za-z]', tail):
        return None, 'alternative_options'
    if re.search(r'(?:但|不过).{0,12}(?:也可能|也许|不能确定|不确定|无法确定)', tail):
        return None, 'uncertain_conclusion'
    for retraction in re.finditer(r'(?:不是|并非|不应选)\s*([A-Za-z]+)', tail):
        if retraction[1].upper() == answer:
            return None, 'retracted_conclusion'
    return answer, None


def parse(text, legal_options='ABCDE', finish_reason=None):
    final, error = visible_region(text, finish_reason)
    if error:
        return error
    final = unicodedata.normalize('NFKC', final).strip()
    markers = list(MARKER.finditer(final))
    choices = list(CHOICE.finditer(final))
    claims = sorted([(m, 'marker') for m in markers] + [(m, 'choice') for m in choices], key=lambda x:x[0].start())
    answers = []
    for i, (claim, kind) in enumerate(claims):
        end = claims[i+1][0].start() if i+1 < len(claims) else len(final)
        region = final[claim.end():end]
        answer, problem = option_prefix(region, legal_options)
        if problem:
            # A heading such as "正确答案是：\n最终答案：A" is not a second answer.
            if kind == 'marker' and not region.strip(' \n\t:：*') and i+1 < len(claims):
                continue
            if kind == 'choice' and problem == 'missing_explicit_option':
                continue  # "选择合适的治疗" is not an option assertion.
            return fail(problem, True)
        prefix = final[max(0, claim.start()-5):claim.start()]
        if re.search(r'(?:不|非|并非|不能|不应|不要)\s*$', prefix):
            continue  # Negative mentions do not assert a selected option.
        answers.append(answer)
    if not answers:
        if claims:
            return fail('no_positive_conclusion')
        # A bare letter/letter-plus-description at the start of the visible answer.
        bare = re.sub(r'^选项\s*', '', final)
        try:
            canonical_answer(bare.rstrip('。.!！'), legal_options)
        except ValueError:
            if bare != final or not re.match(r'^[A-Za-z]+\s+', bare) or re.search(r'不正确|错误|不符合|不选|不是', bare):
                return fail('no_explicit_final_answer')
        answer, problem = option_prefix(bare, legal_options)
        if problem:
            return fail('no_explicit_final_answer')
        answers.append(answer)
    if len(set(answers)) != 1:
        return fail('conflicting_explicit_answers', True)
    old = legacy_parse(text, legal_options, finish_reason)
    if old.strict_match and old.answer_set == answers[0]:
        return old
    return ParseResult(answer_set=answers[0], fallback_match=True, valid_format=False)


def rescore(response, legal_options='ABCDE'):
    result = parse(response['raw_output'], legal_options, response['finish_reason'])
    return dict(response, parsed_answer=result.answer_set,
                correct=result.answer_set == response['ground_truth'],
                strict_format=result.valid_format,
                parse_method='strict' if result.strict_match else 'fallback' if result.fallback_match else 'unparseable',
                ambiguous=result.ambiguous, unparseable=result.answer_set is None,
                parser_error=result.error_type)
