#!/usr/bin/env python3
"""Publish the verified v3 measurement without overwriting historical evidence."""
from pathlib import Path
import sys
import json

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from medical_posttrain.evaluation.core import read,rows,ref,check_ref
from medical_posttrain.rl.common import durable
from stage5_objective_evidence import put,roles,EXAMS

IDX=ROOT/'experiments/stage5'
CLOSE=IDX/'closure_v3_20260918'
REV=IDX/'parser_correction_v3'
OUT=Path('/data/WSH/medical-post-train-artifacts/evaluation/stage5_project_v1')


def write_once(path,text):
    if path.exists(): assert path.read_text()==text
    else:
        with path.open('x') as f:f.write(text)


def remine_cases():
    primary,_=roles();result={};counts={}
    for ds in EXAMS:
        manifest=read(IDX/'manifests'/f'{ds}.json')
        files={k:OUT/'rescored_v3'/v.replace(':','_')/ds/'predictions.jsonl' for k,v in primary.items()}
        data={k:rows(p) for k,p in files.items()}
        assert all([x['prompt_id'] for x in v]==manifest['ids'] for v in data.values())
        buckets={k:[] for k in ['sft_wrong_both_rl_correct','sft_wrong_dynamic_only_correct',
            'vanilla_correct_dynamic_wrong','dynamic_correct_vanilla_wrong',
            'both_rl_regress_from_sft','sft_parseable_rl_unparseable','medium_hard_dynamic_gain']}
        difficult=set()
        if ds==EXAMS[0]:
            d=read(IDX/'cmexam_slice_manifest_v1.json')['difficulty'];difficult=set(d['medium']+d['hard'])
        for s,v,d in zip(data['sft'],data['selected_vanilla'],data['selected_dynamic']):
            flags=[not s['correct'] and v['correct'] and d['correct'],
                   not s['correct'] and not v['correct'] and d['correct'],
                   v['correct'] and not d['correct'],d['correct'] and not v['correct'],
                   s['correct'] and not v['correct'] and not d['correct'],
                   not s['unparseable'] and (v['unparseable'] or d['unparseable']),
                   d['prompt_id'] in difficult and d['correct'] and not v['correct']]
            for key,yes in zip(buckets,flags):
                if yes:buckets[key].append(s['prompt_id'])
        result[ds]=dict(buckets=buckets,predictions={k:ref(p) for k,p in files.items()},
            source_items=manifest['data'],review_status='MACHINE_CANDIDATES_NOT_MANUAL_REVIEW')
        counts[ds]={k:len(v) for k,v in buckets.items()}
    put(CLOSE/'case_candidates_v3.json',dict(parser='v3',datasets=result,counts=counts,human_reviews_completed=0))
    return counts


def main():
    verified=read(CLOSE/'objective_verification_v3.json')
    assert verified['result']=='PASS' and verified['responses']==35236 and verified['jobs']==8
    for r in verified['results'].values():check_ref(r)
    packet=read(CLOSE/'review_packet.json');check_ref(packet['archive'])
    cm=read(REV/'cmexam_final_results_v3.json');cmb=read(REV/'cmb_final_results_v3.json')
    primary,endpoints=roles();counts=remine_cases()
    efficiency=read(ROOT/'experiments/stage4/postformal_analysis_v1.json')
    states={a:efficiency['runs'][a]['state'] for a in ['vanilla','dynamic']}
    tokens={a:s['prompt_tokens']+s['output_tokens'] for a,s in states.items()}
    ratio=tokens['dynamic']/tokens['vanilla']
    report=ROOT/'docs/stage_reports/05_objective_evaluation_v3.md'
    lines=['# Stage5 objective evaluation — v3 verified measurement','',
        'As of 2026-09-18: OBJECTIVE_EVAL_PASS (v3); OPEN_QA_JUDGE_PENDING. Stage5 remains FULL_RUNNING and Stage6 NOT_STARTED. This is the active exam report; v1/v2 reports are retained historical measurements.','',
        'All 35,236 unique final exam generations (four checkpoints × 8,809 questions) were rescored from original responses. No valid output was regenerated. The v3 verifier checked raw/response receipt hashes and all predictions, and independently reproduced summaries, secondary subsets, slices and paired statistics. Historical full token replay was linked through unchanged evidence.','',
        '## Primary models selected on validation only','',
        '| Model | Checkpoint groups | CMExam 6809 | CMB 2000 | CMExam clean 6732 | CMB medical 1929 |',
        '|---|---:|---:|---:|---:|---:|']
    for role,cid in primary.items():
        values=[cm['primary']['summary'][role],cmb['primary']['summary'][role],cm['secondary']['primary']['summary'][role],cmb['secondary']['primary']['summary'][role]]
        lines.append('| '+role+' | '+('fixed SFT' if cid=='sft' else cid.split(':')[1])+' | '+' | '.join(f"{v['correct_count']}/{v['n']} ({v['accuracy']*100:.2f}%)" for v in values)+' |')
    lines+=['','## Equal-update comparison','',
        '| Endpoint | CMExam | CMB |','|---|---:|---:|']
    for role in endpoints:
        a=cm['scientific_endpoints']['summary'][role];b=cmb['scientific_endpoints']['summary'][role]
        lines.append(f"| {role} 5000 groups | {a['accuracy']*100:.2f}% | {b['accuracy']*100:.2f}% |")
    lines+=['','Both formal runs used 5,000 groups, 625 policy windows and 1,250 optimizer steps. Checkpoint selection remains frozen at Vanilla 4608 and Dynamic 5000; final tests did not select a new checkpoint.','',
        '## Paired uncertainty','', '| Dataset / comparison | Delta (pp) | 95% paired bootstrap CI | Exact McNemar p |', '|---|---:|---|---:|']
    for label,d in [('CMExam',cm),('CMB',cmb)]:
        for scope in ['primary','scientific_endpoints']:
            for name,p in d[scope]['paired'].items():
                lo,hi=p['ci95_pp'];lines.append(f"| {label} / {scope}: {name} | {p['delta_pp']:+.3f} | [{lo:.3f}, {hi:.3f}] | {p['exact_mcnemar']:.4f} |")
    lines+=['','Bootstrap: 10,000 paired prompt resamples, seed 20260914. Primary pairwise intervals cross zero. These data do not establish a Dynamic advantage, equivalence, or consistent RL improvement over SFT. Multiple comparisons/slices are exploratory and unadjusted; one training seed limits generalization.','',
        '## Fixed CMExam difficulty slices','', '| Model | Easy (4525) | Medium (1500) | Hard (784) |','|---|---:|---:|---:|']
    for role,cid in primary.items():
        values=cm['slices'][cid]['difficulty'];lines.append('| '+role+' | '+' | '.join(f"{values[k]['accuracy']*100:.2f}%" for k in ['easy','medium','hard'])+' |')
    lines+=['','Dynamic has a higher hard-slice point estimate than selected Vanilla, but a lower medium-slice point estimate. Do not infer a medium-difficulty mechanism or a subgroup win from point estimates. Complete fixed category membership, counts and interval results remain in the linked JSON tables.','',
        '## Parser correction and costs','',
        'Final exam prompts asked for “最终答案：A”, whereas selection prompts required letter-only answer tags. The original parser rejected many clear answers. V2 fixed marker-in-tag handling; v3 additionally handles repeated equal conclusions, option descriptions and ordinary option mentions. V3 was specified after observing test outputs, with tests and code pinned before its aggregate scores were computed. This is a disclosed measurement correction, not a pre-test frozen parser.','',
        '| Model | CMExam unparseable | CMB unparseable |','|---|---:|---:|']
    for role in primary:lines.append(f"| {role} | {cm['primary']['summary'][role]['unparseable_rate']*100:.2f}% | {cmb['primary']['summary'][role]['unparseable_rate']*100:.2f}% |")
    lines += ['',f"Formal training generated {tokens['vanilla']:,} prompt-plus-output tokens for Vanilla and {tokens['dynamic']:,} for Dynamic ({ratio:.5f}×). This ratio excludes validation and optimizer compute and is not a GPU-hour/currency ratio. The held-out results do not establish a rollout-compute efficiency advantage.",'',
        '## Open QA, safety and remaining gates','',
        'The 74 clinical cases/208 questions and 200 retention prompts have 1,224 complete local responses. The frozen blind schedule has 1,224 pairs plus 123 position flips (1,347 judging entries). No independent judgments or human reviews are complete. There are no preference win-rates or rubric quality scores yet.','',
        f"Safety is machine triage only: 111 source-defined items, 333 reused responses, four flagged responses across three items. These are review candidates, not confirmed unsafe answers; unflagged outputs are not certified safe. The human packet currently covers {packet['current_human_items']} items / {packet['human_pairs']} pairs: 82 frozen items plus additional flagged items. All future judge critical flags must also be reviewed.",'',
        'Remaining Stage5 gates: independent scoring, position-consistency analysis, required real human audit, safety review/adjudication, and final integrated narrative. Stage4 owner waiver covered Stage4 manual documentation only; it does not waive Stage5 review. No paid API calls were made for this closure.','',
        '## Evidence','',
        '- [V3 verification](../../experiments/stage5/closure_v3_20260918/objective_verification_v3.json)',
        '- [CMExam tables](../../experiments/stage5/parser_correction_v3/cmexam_final_results_v3.json) and [CMB tables](../../experiments/stage5/parser_correction_v3/cmb_final_results_v3.json)',
        '- [V3 case index](../../experiments/stage5/closure_v3_20260918/case_candidates_v3.json): all gain/regression buckets retained, including unfavorable cases.',
        '- [Blind-review package manifest](../../experiments/stage5/closure_v3_20260918/review_packet.json)',
        '- [Historical v1 report](05_objective_evaluation.md) and [initial v3 snapshot](05_exam_rescoring_v3.md); do not use either as the current complete report.',
        '- [Cleanup round 1](../../experiments/maintenance/disk_cleanup_20260918/summary.json) and [round 2](../../experiments/maintenance/disk_cleanup_20260918_round2/summary.json). Intermediate native training states were pruned after owner authorization. All adapters and 77 protected full checkpoints remain. Historical Stage4 full-file verification cannot be repeated unchanged for pruned nodes; original verification receipts and removal manifests are retained.','']
    write_once(report,'\n'.join(lines))
    story=ROOT/'docs/stage_reports/05_interview_story_v3.md'
    write_once(story,'# Stage5 evidence-based project narrative\n\n'
        'The project completed matched Vanilla/Dynamic GSPO training from the same SFT initialization and evaluated validation-selected checkpoints plus equal-update endpoints. On held-out CMExam/CMB, Dynamic had small selected-checkpoint gains but all main paired confidence intervals crossed zero. Equal-update results were nearly tied, while Dynamic used about 2.925 times the rollout tokens. The evidence therefore supports an inconclusive capability effect with a measured cost increase, not a compute-efficiency win.\n\n'
        'A major evaluation failure was found in the interface between prompt and answer parser. Apparently severe model regressions changed after correcting falsely rejected explicit answers. The response was to retain every raw output and earlier score, test and version the revised parser, replay all models uniformly, and disclose that correction occurred after test outputs were observed. No checkpoint was reselected and no wrong answer was regenerated.\n\n'
        'Open medical QA is still pending independent judgment and real human audit. Generation completion is not clinical-quality validation. The next delivery is a frozen anonymous review package, followed by adjudication and final Stage5 verification before serving benchmarks. See [the active report](05_objective_evaluation_v3.md) for numbers and evidence.\n')
    handoff=dict(status='OBJECTIVE_EVAL_PASS',parser_version='v3',stage5_status='FULL_RUNNING',
        open_qa_judge_status='OPEN_QA_JUDGE_PENDING',selected=read(IDX/'selected_checkpoints_v1.json'),
        cmexam=cm,cmb=cmb,objective_verification=ref(CLOSE/'objective_verification_v3.json'),
        report=ref(report),interview_story=ref(story),cases=ref(CLOSE/'case_candidates_v3.json'),
        review_packet=ref(CLOSE/'review_packet.json'),supersedes=ref(ROOT/'experiments/handoffs/stage5_results_to_chatgpt_v1.json'),
        training_tokens=tokens,training_token_ratio=ratio,
        human_reviews_completed=0,clinical_validation=False,paid_api_calls=0,stage6='NOT_STARTED',
        unsupported_claims=['Dynamic capability superiority','Compute-efficiency benefit','Clinical safety','Open-QA quality or preference without judgments','Multi-seed robustness'],
        maintenance=[ref(ROOT/'experiments/maintenance'/n/'summary.json') for n in ['disk_cleanup_20260918','disk_cleanup_20260918_round2']])
    handoff_path=ROOT/'experiments/handoffs/stage5_results_to_chatgpt_v3.json';put(handoff_path,handoff)
    current=dict(active_exam_parser='v3',report=ref(report),handoff=ref(handoff_path),
        objective_verification=ref(CLOSE/'objective_verification_v3.json'),
        open_qa_status='OPEN_QA_JUDGE_PENDING',stage5_done=False,stage6_ready=False)
    put(IDX/'current_results_v3.json',current)
    state=read(ROOT/'project_state.json')
    assert state['stages']['6']['status']=='NOT_STARTED'
    state['stages']['5'].update(status='FULL_RUNNING',objective_evaluation_status='OBJECTIVE_EVAL_PASS',
        objective_parser_version='v3',objective_verification_receipt=current['objective_verification'],
        stage_report=str(report.relative_to(ROOT)),interview_story=str(story.relative_to(ROOT)),
        handoff=str(handoff_path.relative_to(ROOT)),open_qa_status='OPEN_QA_JUDGE_PENDING',
        execution_status='LOCAL_TASK_COMPLETE',verification_receipt=None,
        pending_requirements=['Independent open-QA judging','Real human audit and safety adjudication','Full Stage5 acceptance'],
        progress=dict(checkpoints_evaluated=20,selection_generations=20480,exam_generations=35236,open_qa_generations=1224,exam_jobs_verified=8))
    durable(ROOT/'project_state.json',state)
    print(json.dumps(dict(status='V3_HANDOFF_READY',cases=counts,stage5_done=False)))


if __name__=='__main__':main()
