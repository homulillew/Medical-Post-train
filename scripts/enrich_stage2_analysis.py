"""Post-frozen analysis: correctness contrast, source incompleteness, exact cases."""
from collections import Counter
from datetime import date,timedelta
from pathlib import Path
import re
from medical_posttrain.evidence import write_json
from medical_posttrain.evidence.stage2 import selected_path,jsonlines,read,record,INDEX
from medical_posttrain.rollout.statistics import distribution
from medical_posttrain.reward.parser import reasoning_text

root=selected_path('formal');s=read(root/'summary.json');assert s['completed_responses']==4000
rows=jsonlines(root/'trajectories.jsonl');groups=jsonlines(root/'groups.jsonl');pool=jsonlines(selected_path('data')/'candidate_pool.jsonl')
byid={r['trajectory_id']:r for r in rows}
contrasts=Counter();details=[]
for g in groups:
    rs=[byid[t] for t in g['trajectory_ids']]
    parsed_wrong=sum(r['acc']==0 and r['parsed_answer'] is not None for r in rs)
    unparsed=sum(r['parsed_answer'] is None for r in rs)
    if g['classification']=='mixed':
        kind='has_parsed_wrong_answers' if parsed_wrong else 'contrast_only_from_unparseable_responses'
        contrasts[kind]+=1
        details.append(dict(group_id=g['group_id'],acc_vector=g['acc_vector'],parsed_wrong_count=parsed_wrong,unparseable_count=unparsed,category=kind))

def image_related(r):return bool(re.search(r'暂无图|无图|如图|见图|下图|图示',r['question']))
missing_images=[r['prompt_id'] for r in pool if image_related(r)]
image_set=set(missing_images)
missing_reference_ids={r['prompt_id'] for r in pool if not r['reference_explanation'].strip()}
result=dict(run_id=root.name,mixed_group_contrast_counts=dict(contrasts),mixed_groups=details,
            high_semantic_wrong_parse_status=dict(Counter('parsed_wrong' if r['parsed_answer'] is not None else 'unparseable' for r in rows if r['acc']==0 and r['semantic']>.9)),
            all_wrong_hybrid_std_nonzero=sum(g['classification']=='all_wrong' and g['hybrid_reward_std']>1e-12 for g in groups),
            all_correct_hybrid_std_nonzero=sum(g['classification']=='all_correct' and g['hybrid_reward_std']>1e-12 for g in groups),
            empty_reasoning_count=sum(r['reasoning_tokens']==0 for r in rows),
            near_empty_reasoning_le10=sum(r['reasoning_tokens']<=10 for r in rows),
            format_valid_empty_reasoning=sum(r['format_valid'] and r['reasoning_tokens']==0 for r in rows),
            pool_image_reference_prompts=len(missing_images),pool_image_reference_ids=missing_images,
            formal_image_reference_prompts=sum(g['prompt_id'] in image_set for g in groups),
            image_reference_trajectory_accuracy=distribution([r['acc'] for r in rows if r['prompt_id'] in image_set]),
            formal_missing_reference_prompts=sum(g['prompt_id'] in missing_reference_ids for g in groups),
            semantic_encoded_only_by_accuracy={str(a):distribution([r['semantic'] for r in rows if r['acc']==a and r['semantic_reason']=='encoded']) for a in (0,1)},
            exact_arithmetic_case=dict(prompt_id='cmexam_train:fadb22c89beb1b7115dc36460ba792eb96b7b972:30338',member=2,
                output_claim='2002-04-18 + 14 days = 2002-04-30',recomputed_date=str(date(2002,4,18)+timedelta(days=14)),
                option_distance_from_may3_days={'A_May2':1,'B_May4':1,'C_May6':3},scope='calendar arithmetic only; no clinical ovulation-date validation'),
            caveats=['Mixed groups can arise from malformed outputs as well as disagreements among parseable answers; acc is the defined task-level reward.',
                     'All-correct groups may retain semantic ranking signal under hybrid reward. Filtering them removes that signal by design, not because all advantages must be zero.',
                     'Missing-image regex is descriptive triage, not a new exclusion or category/difficulty annotation.',
                     'Raw semantic correlations and qualitative source disagreements are not causal or clinical judgments.'],
            sources=[record(root/'trajectories.jsonl'),record(root/'groups.jsonl')])
write_json(INDEX/'supplementary_analysis.json',result)
# Append focused manual cases based on already-read immutable trajectories.
coverage=read(INDEX/'case_coverage.json');focused=[]
for subtype,source_row,member,observation in [
 ('correct_reference_disagreement',31425,0,'Final D matches GT; reasoning calls urine specific gravity high and attributes to low volume, while reference describes low specific gravity and rejects low volume. Clinical truth NOT_ASSESSED.'),
 ('correct_internal_contradiction',19412,2,'Final E is correct, yet preceding sentence denies the host-weight increase in E. Reference explanation absent; semantic is zero for missing reference.'),
 ('date_arithmetic_error',30338,2,'Calendar addition and nearest-option distances are inconsistent with the raw reasoning. Final C differs from GT B.'),
 ('missing_image',27961,2,'Prompt explicitly says no image; response discusses hypothetical imaging findings. Correct B does not establish visual reasoning.')]:
    r=next(r for r in rows if r['prompt_id'].endswith(':'+str(source_row)) and r['member_index']==member)
    cid='S2-MANUAL-'+subtype.upper().replace('_','-')
    item=dict(case_id=cid,run_id=root.name,stage=2,category='REWARD_CASE',subtype=subtype,prompt_id=r['prompt_id'],trajectory_id=r['trajectory_id'],
              prompt=r['question'],ground_truth=r['ground_truth'],parsed_answer=r['parsed_answer'],raw_output_excerpt=r['raw_output'],
              reward_components={k:r[k] for k in ('acc','semantic','semantic_contribution','format','total_reward')},
              observation=observation,why_interesting='Specific manually read limitation of option-level reward or source completeness',
              hypothesis=None,alternative_explanations=['Reference explanation itself is not clinically adjudicated'],followup='Retain fixed prompt case for future Stage3/4 review; no reward changes in this profiling.')
    focused.append(item)
write_json(INDEX/'manual_cases.json',focused)
# Original automatic case inventory remains unchanged; manual cases have a separate file.
print('Supplementary mixed attribution',dict(contrasts))
