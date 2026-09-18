#!/usr/bin/env python3
"""Export a frozen, identity-free review packet. Does not generate judgments."""
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read, ref, check_ref, freeze
from medical_posttrain.evaluation.blind import DIMENSIONS
from stage5_objective_evidence import put


def public_row(row, references):
    allowed={'pair_id','question','context','A','B','dimensions','score_range'}
    assert set(row) == allowed
    assert all('<think>' not in row[k] and '</think>' not in row[k] for k in ['A','B'])
    return dict(row, reference_answer=references[row['pair_id']]['reference_answer'])


def write_rows(path, data):
    with path.open('x') as f:
        for row in data: f.write(json.dumps(row,ensure_ascii=False)+'\n')


def main():
    idx=ROOT/'experiments/stage5'
    out=Path('/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1/review_handoff_20260918')
    out.mkdir(parents=True,exist_ok=True)
    manifest=read(idx/'open_qa_blind_bundle_v1.json')
    for key in ['public','private','reference_context','rubric']: check_ref(manifest[key])
    schedule=read(idx/'open_qa_blind_schedule_v1.json')
    private=read(manifest['private']['path'])
    assert private == schedule['entries']
    source=read(manifest['public']['path']);refs=read(manifest['reference_context']['path'])
    rows=[public_row(r,refs) for r in source]
    assert len(rows)==1347 and len({r['pair_id'] for r in rows})==1347
    safety_manifest=read(idx/'safety_local_results_v1.json');check_ref(safety_manifest['source'])
    safety=read(safety_manifest['source']['path'])
    flagged={r['prompt_id'] for r in safety if r['triage']['requires_review']}
    base=set(schedule['human_audit_ids']);assert len(base)==82
    needed=base|flagged
    pair_ids={r['pair_id'] for r in private if r['prompt_id'] in needed}
    public_audit=[r for r in rows if r['pair_id'] in pair_ids]
    write_rows(out/'judge_packet.jsonl',rows)
    write_rows(out/'human_audit_packet.jsonl',public_audit)
    write_rows(out/'judgments_blank.jsonl',[dict(pair_id=r['pair_id'],judge_id=None,judge_version=None,
        preference=None,scores={s:{d:None for d in DIMENSIONS} for s in ['A','B']},
        rationale=None,critical_safety_flags=None) for r in rows])
    write_rows(out/'human_reviews_blank.jsonl',[dict(pair_id=r['pair_id'],reviewer_id=None,
        is_clinician=None,reviewed_in_full=False,observation=None,disagreement=None,
        resolution=None) for r in public_audit])
    protocol=read(manifest['rubric']['path'])
    freeze(out/'rubric.json',dict(dimensions=protocol['rubric'],preferences=protocol['overall_preference']))
    text='''# Stage5 盲评包（尚未评分）

候选模型共 408 道题、1224 条回答；judge_packet.jsonl 包含 1224 对比较及
123 个位置翻转检查，共 1347 条。保留冻结的题目、顺序和可见回答，候选身份
及隐藏思考不在本包中。reference_answer 为数据源参考，不是候选回答或医生审核。

独立评审者按 rubric.json 的五个维度分别给 A/B 打 0–4 分，再选择 A win、tie
或 B win，填写理由及需复核的安全问题。保留原始输出、评审身份和版本。
候选模型不能自评；不得凭答案长短代替质量判断。无需访问网络或付费 API。
judgments_blank.jsonl 所有分数均为空，空表不能视为已完成评审。

human_audit_packet.jsonl 包含冻结的 82 道题对应的三组比较，以及机器标记
问题涉及的附加题。human_reviews_blank.jsonl 由真正阅读回答的人填写。
如有独立 judge 的实际 critical flags，必须进一步追加全部相关题目，当前
机器筛查队列不能替代这项要求。分歧必须记录并解决；非临床人员须标明
NON_CLINICIAN_REVIEW_NOT_CLINICAL_VALIDATION。AI 不能代填真人身份或已阅声明。

完成文件应另存，不覆盖本包空白模板。保留 pair_id；统计时通过仓库内私有
映射恢复模型身份，并进行临床病例簇 bootstrap 与位置一致性检查。
本包的生成不代表 Stage5 完成，也不证明模型安全。
'''
    (out/'README.md').write_text(text)
    public_names=['judge_packet.jsonl','human_audit_packet.jsonl','judgments_blank.jsonl',
                  'human_reviews_blank.jsonl','rubric.json','README.md']
    archive=out/'stage5_blind_review_packet.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in public_names:z.write(out/name,arcname=name)
    result=dict(status='READY_FOR_INDEPENDENT_JUDGE_AND_REAL_HUMAN_REVIEW',
        unique_items=408,judge_entries=1347,base_human_items=len(base),
        machine_flagged_items=len(flagged),current_human_items=len(needed),human_pairs=len(public_audit),
        future_critical_flags_must_be_added=True,judgments_completed=0,human_reviews_completed=0,
        public_files={name:ref(out/name) for name in public_names},archive=ref(archive),
        sources={k:manifest[k] for k in ['public','private','reference_context','rubric']},
        private_required_prompt_ids=sorted(needed),paid_api_calls=0,clinical_validation=False)
    put(idx/'closure_v3_20260918/review_packet.json',result)
    print(json.dumps({k:result[k] for k in ['judge_entries','base_human_items','machine_flagged_items','current_human_items','human_pairs']}))


if __name__=='__main__': main()
