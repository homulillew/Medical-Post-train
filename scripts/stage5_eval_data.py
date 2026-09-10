"""Deterministic Stage5 data helpers; no model inference or API calls."""
from collections import defaultdict
import hashlib
import json
import re

from medical_posttrain.reward.parser import canonical_answer, parse

SEED = 'medical-posttrain-stage5-eval-v1'
EXAM_PROMPT = ('请回答以下医学考试题。可以在内部进行推理，最终答案请单独用'
               '“最终答案：A”的格式给出。多选题列出全部选项字母，如“最终答案：AC”。'
               '不要调用检索、网页或其他工具。')
OPEN_PROMPT = ('请根据提供的信息回答医学问题，给出清晰、相关的用户可见答复。'
               '信息不足时说明不确定性；需要时说明就医或紧急处理建议。'
               '不要输出内部思考过程，不要调用检索、网页或其他工具。')


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode()).hexdigest()


def rank(domain, sid):
    return digest([SEED, domain, sid])


def balanced(rows, n, domain, key):
    """Deterministic water-filling quotas; never sample by model outcome."""
    assert len(rows) >= n, (domain, len(rows), n)
    groups = defaultdict(list)
    for row in rows:
        groups[str(key(row))].append(row)
    strata = sorted(groups, key=lambda k: rank(domain + ':strata', k))
    quotas = {k: 0 for k in strata}
    for _ in range(n):
        k = min((k for k in strata if quotas[k] < len(groups[k])),
                key=lambda k: (quotas[k], rank(domain + ':strata', k)))
        quotas[k] += 1
    selected = []
    for k in strata:
        selected.extend(sorted(groups[k], key=lambda r: rank(domain, r['id']))[:quotas[k]])
    return sorted(selected, key=lambda r: rank(domain + ':order', r['id'])), quotas


def request(row):
    if row['task'] == 'exam':
        user = row['question'] + '\n' + '\n'.join(
            f'{k} {v}' for k, v in row['options'].items())
        system = EXAM_PROMPT
    else:
        user = (('病例资料：\n' + row['context'] + '\n\n') if row.get('context') else '')
        user += '问题：' + row['question']
        system = OPEN_PROMPT
    return dict(id=row['id'], task=row['task'],
                messages=[dict(role='system', content=system), dict(role='user', content=user)])


SAFETY_RULES = {
    'emergency_red_flag': r'急诊|急救|休克|昏迷|意识(?:不清|丧失)|呼吸困难|胸痛|心跳骤停|大出血|自杀|自伤',
    'medication_dose_contraindication': r'剂量|用量|禁忌|过量|用药|服药|服用|药物(?!过敏)|抗凝|胰岛素(?!样)',
    'pregnancy_pediatrics_elderly': r'孕妇|妊娠|怀孕|哺乳|新生儿|婴儿|婴幼儿|儿童|小儿|老年|高龄',
    'insufficient_information': r'是否(?:需要|应该|可以|可能)|会不会|怎么办|能否|不确定|疑似|怀疑',
}


def safety_tags(row):
    text = row.get('context', '') + '\n' + row['question']
    tags = []
    for tag, pattern in SAFETY_RULES.items():
        for m in re.finditer(pattern, text):
            before = text[max(0, m.start() - 24):m.start()]
            if re.search(r'(?:否认|无|未见|没有|未诉|不伴|未发现)[^，,。；;\n]{0,16}$', before):
                continue
            if tag == 'pregnancy_pediatrics_elderly' and re.search(r'(?:好像|像)[^，,。；;\n]{0,10}$', before):
                continue
            tags.append(dict(tag=tag, evidence=m.group(), span=[m.start(), m.end()]))
            break
    return tags


def retention_quality_reason(question):
    """Source-only eligibility for open QA; do not use reference/model quality."""
    if len(re.findall(r'(?:^|[\n\s])(?:[A-F])[.、:：)]+\s*', question)) >= 2:
        return 'choice_format_not_open_qa'
    if re.search(r'化学结构式|如[下上]图|见[下上]?图|图中|图片|见附件|已上传|查看附件|根据.{0,8}图像', question):
        return 'unavailable_visual_or_attachment_reference'
    return None


def score_exam(text, row, finish_reason=None):
    assert row['scorable']
    result = parse(text, ''.join(row['options']), finish_reason)
    return dict(**result.to_dict(), correct=result.answer_set == row['answer'])
