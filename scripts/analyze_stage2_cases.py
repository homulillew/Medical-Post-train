"""Mine reproducible cases and prepare a fixed qualitative review packet."""
from collections import Counter
from pathlib import Path
import re
from medical_posttrain.evidence import write_json
from medical_posttrain.evidence.stage2 import selected_path,jsonlines,read,INDEX,dump_lines,record
from medical_posttrain.data.stage1 import text_hash
from medical_posttrain.reward.parser import reasoning_text

root=selected_path('formal');rows=jsonlines(root/'trajectories.jsonl');groups=jsonlines(root/'groups.jsonl')
bygroup={g['group_id']:[r for r in rows if r['group_id']==g['group_id']] for g in groups}
lookup={r['prompt_id']:r for r in jsonlines(selected_path('data')/'candidate_pool.jsonl')}
categories={};cases=[];review={}

def take(category,selected,limit=3):
    selected=selected[:limit];caseids=[]
    for row in selected:
        cid='S2-'+category.upper().replace('_','-')+'-'+str(len(caseids)+1).zfill(3)
        caseids.append(cid)
        case=dict(case_id=cid,run_id=root.name,stage=2,category='REWARD_CASE' if 'semantic' in category else 'BOUNDARY_CASE',
                  subtype=category,prompt_id=row['prompt_id'],group_id=row['group_id'],trajectory_id=row['trajectory_id'],
                  prompt=lookup[row['prompt_id']]['question'],ground_truth=row['ground_truth'],parsed_answer=row['parsed_answer'],
                  raw_output_excerpt=row['raw_output'][:1400],raw_output_sha256=text_hash(row['raw_output']),
                  reward_components={k:row[k] for k in ('acc','semantic','semantic_contribution','format','total_reward')},
                  why_interesting=f'Deterministic representative of {category}; inspect full trajectory in the sealed bulk file.',
                  observation=f"acc={row['acc']}, semantic={row['semantic']:.6f}, format={row['format']}, finish={row['finish_reason']}",
                  hypothesis='Selection illustrates a boundary; not an estimate of clinical reasoning quality.',
                  alternative_explanations=['Source explanation may be missing or incorrect','Lexical overlap is not factual entailment'],
                  followup='Track fixed prompt and policy lineage in future Stage3/4, without training in Stage2.',
                  raw_source=str(root/'trajectories.jsonl'))
        cases.append(case);review[row['trajectory_id']]=row
    categories[category]=dict(status='OBSERVED' if selected else 'NOT_OBSERVED',case_ids=caseids)

for number,category in enumerate(['all_wrong','one_of_four','two_of_four','three_of_four','all_correct']):
    selected=[g for g in groups if g['correct_count']==number]
    take(category,[bygroup[g['group_id']][0] for g in selected])
    for group in selected[:3]:
        for r in bygroup[group['group_id']]:review[r['trajectory_id']]=r

take('high_semantic_wrong',sorted([r for r in rows if r['acc']==0 and r['semantic']>.9],key=lambda r:-r['semantic']))
take('highest_semantic_wrong_relative',sorted([r for r in rows if r['acc']==0],key=lambda r:-r['semantic']))
take('correct_low_semantic',sorted([r for r in rows if r['acc']==1 and r['semantic']<.5],key=lambda r:r['semantic']))
take('correct_low_semantic_nonempty_reference',sorted([r for r in rows if r['acc']==1 and r['semantic']<.5 and r['semantic_reason']=='encoded'],key=lambda r:r['semantic']))
take('parser_ambiguous',[r for r in rows if r['ambiguous']])
take('fallback_correct',[r for r in rows if r['parse_method']=='fallback' and r['acc']==1])
take('format_failure',[r for r in rows if not r['format_valid']])
take('truncation',[r for r in rows if r['finish_reason']=='length'])
take('long_reasoning',sorted(rows,key=lambda r:-r['reasoning_tokens']))
take('short_reasoning',sorted(rows,key=lambda r:r['reasoning_tokens']))
take('multi_select',[r for r in rows if len(r['ground_truth'])>1])
take('format_gaming_candidate',[r for r in rows if r['format_valid'] and r['reasoning_tokens']<10])
# This is a triage flag, never a medical error label or additional reward score.
suspicious=[]
for r in rows:
    if not r['acc']:continue
    reasoning=reasoning_text(r['raw_output']);reference=lookup[r['prompt_id']]['reference_explanation']
    numbers=set(re.findall(r'\d+(?:\.\d+)?',reasoning))-set(re.findall(r'\d+(?:\.\d+)?',reference))
    if numbers and reference.strip():suspicious.append(r)
take('correct_reasoning_numeric_triage',suspicious)
for r in sorted(rows,key=lambda r:text_hash('42:manual:'+r['trajectory_id'])):
    if len(review)>=80:break
    review[r['trajectory_id']]=r
assert 50<=len(review)<=100
packet=[]
for r in review.values():packet.append(dict(**r,reference_explanation=lookup[r['prompt_id']]['reference_explanation']))
write_json(INDEX/'case_coverage.json',dict(run_id=root.name,categories=categories,cases=cases,
          thresholds=dict(high_semantic_wrong='acc=0 and sem>0.9',correct_low_semantic='acc=1 and sem<0.5'),
          clinical_reasoning_quality='NOT_ASSESSED; numeric triage is not an error judgment'))
write_json(root/'manual_review_packet.json',packet)
write_json(INDEX/'manual_review_selection.json',dict(run_id=root.name,selection='predefined category coverage, full representative groups, then seed42 SHA order to80',
          count=len(packet),trajectory_ids=[r['trajectory_id'] for r in packet],packet=record(root/'manual_review_packet.json')))
print('Prepared',len(packet),'complete trajectories for manual qualitative reading')
