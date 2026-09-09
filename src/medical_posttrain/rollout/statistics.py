"""Recomputable Stage 2 descriptive summaries, with no training claims."""
from collections import Counter
import math
import numpy as np
from medical_posttrain.evidence.stage2 import jsonlines, read


def distribution(values):
    if not values:return dict(count=0,mean=None,p50=None,p90=None,p95=None,p99=None,max=None,min=None)
    return dict(count=len(values),total=float(sum(values)),mean=float(np.mean(values)),
                **{f'p{p}':float(np.percentile(values,p)) for p in (50,90,95,99)},min=float(min(values)),max=float(max(values)))


def group_summary(rows):
    assert len(rows)==4 and len({r['trajectory_id'] for r in rows})==4
    for k in ('prompt_id','group_id','policy_version','config_sha256','reward_version'):
        assert len({r[k] for r in rows})==1,k
    assert all(r['acc'] in (0,1) and r['finish_reason'] in ('stop','length') for r in rows)
    acc=[r['acc'] for r in rows];correct=sum(acc);rewards=[r['total_reward'] for r in rows]
    return dict(prompt_id=rows[0]['prompt_id'],group_id=rows[0]['group_id'],policy_version=rows[0]['policy_version'],
                trajectory_ids=[r['trajectory_id'] for r in rows],acc_vector=acc,group_accuracy=correct/4,correct_count=correct,
                classification='all_wrong' if correct==0 else 'all_correct' if correct==4 else 'mixed',
                reward_mean=float(np.mean(rewards)),reward_std=float(np.std(rewards)),correctness_std=float(np.std(acc)),
                hybrid_reward_std=float(np.std(rewards)),std_definition='population ddof=0; no actual advantages computed',
                response_length_mean=float(np.mean([r['output_tokens'] for r in rows])),response_length_max=max(r['output_tokens'] for r in rows),
                format_rate=sum(r['format'] for r in rows)/4,distinct_output_texts=len({r['raw_output'] for r in rows}),
                answer_cardinality=len(rows[0]['ground_truth']),prompt_tokens=rows[0]['prompt_tokens'],sampling_metric='acc')


def correlation(a,b):
    if len(a)<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])


def length_summary(rows):
    return {k:distribution([r[k] for r in rows]) for k in ('reasoning_tokens','answer_tokens','output_tokens')}


def summarize(rows,groups,path):
    n=len(rows);ng=len(groups);assert n==ng*4
    assert len({r['trajectory_id'] for r in rows})==n and len({g['prompt_id'] for g in groups})==ng
    counts=Counter(g['classification'] for g in groups);buckets=Counter(g['correct_count'] for g in groups)
    a=counts['mixed']/ng
    classes={g['prompt_id']:g['classification'] for g in groups}
    starts,ends=[],[];generation_attempts=[]
    for attempt in sorted(path.glob('attempt_*')):
        metrics=attempt/'metrics.jsonl'
        if metrics.exists():
            events=jsonlines(metrics);starts.extend(r for r in events if r['event']=='request_started');ends.extend(r for r in events if r['event']=='request_completed')
        if (attempt/'generation_summary.json').exists():generation_attempts.append(read(attempt/'generation_summary.json'))
    done={r['request_id'] for r in ends};unknown=[s['request_id'] for s in starts if s['request_id'] not in done]
    wall=sum(r['seconds'] for r in ends);tokens=sum(r['batch_output_tokens'] for r in ends)
    control_cost=[c for r in generation_attempts for c in r['identity']['controls']]
    classification_lengths={c:length_summary([r for r in rows if classes[r['prompt_id']]==c]) for c in ('all_wrong','mixed','all_correct')}
    return dict(unique_prompts=ng,completed_responses=n,trajectory_accuracy=sum(r['acc'] for r in rows)/n,
                any_correct_rate=sum(g['correct_count']>0 for g in groups)/ng,
                correct_count_distribution={str(k):buckets[k] for k in range(5)},
                group_counts={k:counts[k] for k in ('all_wrong','mixed','all_correct')},
                group_fractions={k:counts[k]/ng for k in ('all_wrong','mixed','all_correct')},
                expected_sampling_amplification=1/a if a else None,amplification_infinite=a==0,
                amplification_scope='First-order fixed SFT policy estimate 1/P_mixed, not measured Dynamic refill or final training cost',
                sampling_metric='acc',lengths=length_summary(rows),lengths_by_group=classification_lengths,
                closure_rate=sum(r['answer_closed'] for r in rows)/n,
                think_closure_rate=sum(r['think_closed'] for r in rows)/n,
                truncation_rate=sum(r['finish_reason']=='length' for r in rows)/n,
                strict_format_rate=sum(r['format_valid'] for r in rows)/n,
                fallback_rate=sum(r['parse_method']=='fallback' for r in rows)/n,
                ambiguous_rate=sum(r['ambiguous'] for r in rows)/n,
                unparseable_rate=sum(r['parsed_answer'] is None for r in rows)/n,
                parse_errors=dict(Counter(r['parse_error'] for r in rows if r['parse_error'])),
                correct_format_failure=sum(r['acc']==1 and not r['format_valid'] for r in rows),
                wrong_format_valid=sum(r['acc']==0 and r['format_valid'] for r in rows),
                semantic_by_accuracy={str(a):distribution([r['semantic'] for r in rows if r['acc']==a]) for a in (0,1)},
                semantic_reasons=dict(Counter(r['semantic_reason'] for r in rows)),
                high_semantic_wrong_gt09=sum(r['acc']==0 and r['semantic']>.9 for r in rows),
                high_semantic_wrong_gt08=sum(r['acc']==0 and r['semantic']>.8 for r in rows),
                correct_low_semantic_lt05=sum(r['acc']==1 and r['semantic']<.5 for r in rows),
                reward_components={k:distribution([r[k] for r in rows]) for k in ('acc','semantic','semantic_contribution','format','total_reward')},
                correlations=dict(acc_sem=correlation([r['acc'] for r in rows],[r['semantic'] for r in rows]),
                                  acc_format=correlation([r['acc'] for r in rows],[r['format'] for r in rows]),
                                  reward_acc=correlation([r['total_reward'] for r in rows],[r['acc'] for r in rows])),
                distinct_output_texts_per_group=dict(Counter(str(g['distinct_output_texts']) for g in groups)),
                answer_cardinality={str(k):dict(groups=sum(g['answer_cardinality']==k for g in groups),
                    accuracy=float(np.mean([r['acc'] for r in rows if len(r['ground_truth'])==k]))) for k in sorted({len(r['ground_truth']) for r in rows})},
                exposure=dict(generated_unique_prompts=ng,accepted_unique_prompts=0,accepted_scope='No Stage3/4 acceptance or optimizer updates performed',
                              correctness_contrast_eligible_unique_prompts=counts['mixed'],prompt_exposure_count={g['prompt_id']:1 for g in groups},
                              generated_repeat_histogram={'1':ng},max_prompt_exposure=1),
                costs=dict(generated_prompt_tokens_shared_prefill=sum(r['prompt_tokens'] for r in starts),
                           logical_prompt_tokens_all_trajectories=sum(r['prompt_tokens'] for r in rows),
                           generated_output_tokens=tokens,canonical_output_tokens=sum(r['output_tokens'] for r in rows),
                           generation_wall_seconds=wall,output_tokens_per_second=tokens/wall if wall else None,
                           control_prompt_tokens=sum(r['prompt_tokens'] for r in control_cost),control_output_tokens=sum(r['output_tokens'] for r in control_cost),
                           control_generation_seconds=sum(r['seconds'] for r in control_cost),
                           generation_process_wall_seconds=sum(r['wall_seconds'] for r in generation_attempts),
                           gpu_reserved_time_scope='Generation worker wall includes engine startup and sleep/wake; embedding process time separately reported',
                           incomplete_requests=unknown,failed_request_tokens='UNKNOWN' if unknown else 0,
                           retry_output_tokens=tokens-sum(r['output_tokens'] for r in rows) if not unknown else 'UNKNOWN',
                           attempt_count=len(generation_attempts)))
