"""Recompute inference-only refill statistics from immutable batch evidence."""
from collections import Counter
from pathlib import Path
from medical_posttrain.evidence.stage2 import read
from medical_posttrain.rollout.statistics import distribution, group_summary
from .dynamic import DISPOSITIONS, initial_state, apply_batch


def summarize(out):
    out=Path(out);config=read(out/'config.json');state=initial_state(config)
    groups=[];rows=[];generation_seconds=0.;scoring_seconds=0.
    for batch in sorted((out/'batches').glob('*')):
        commit=read(batch/'commit.json');scored=read(batch/'scored.json')['groups'];raw=read(batch/'raw.json')
        state,decisions=apply_batch(state,scored,config,15000)
        assert state==commit['state_after'] and decisions==commit['decisions']
        generation_seconds+=raw['generation_seconds'];scoring_seconds+=commit['scoring_seconds']
        for g,d in zip(scored,decisions):
            info=group_summary(g['responses']) if d['valid'] else {}
            groups.append(dict(info,**d));rows.extend(dict(r,disposition=d['disposition']) for r in g['responses'])
    counts=Counter(g['classification'] for g in groups);accepted=[g for g in groups if g['disposition']=='accepted']
    valid=state['valid_generated_groups'];n=state['accepted_mixed_groups'];tokens=sum(r['output_tokens'] for r in rows)
    mixed=lambda gg:dict(Counter(g['mixed_subtype'] for g in gg if g['classification']=='mixed'))
    exposures=state['prompt_exposures'];ae=state['accepted_exposures']
    identity=[];runtimes=[]
    for attempt in sorted(out.glob('attempt_*')):
        if (attempt/'identity_receipt.json').exists():identity.extend(read(attempt/'identity_receipt.json')['controls'])
        path=attempt/'runtime_summary.json'
        if not path.exists():path=attempt/'runtime_progress.json'
        if path.exists():runtimes.append(read(path))
    s2=read(config['selection']['path'])
    return dict(run_id=out.name,state=state,generated_attempts=state['generated_groups'],generated_attempts_unit='group requests, each requesting four trajectories',
        generated_groups=state['generated_groups'],valid_generated_groups=valid,completed_trajectories=state['completed_trajectories'],
        completed_valid_trajectories=state['completed_valid_trajectories'],group_counts={k:counts[k] for k in ('all_wrong','mixed','all_correct','invalid')},
        accepted_mixed_groups=n,overflow_mixed_groups=state['overflow_mixed_groups'],invalid_groups=state['invalid_groups'],
        correct_count_distribution={str(k):sum(g.get('correct_count')==k for g in groups) for k in range(5)},
        accepted_correct_count_distribution={str(k):sum(g['correct_count']==k for g in accepted) for k in (1,2,3)},
        group_fractions={k:counts[k]/valid for k in ('all_wrong','mixed','all_correct')} if valid else {},
        sampling_metric='acc',sampling_amplification=valid/n if n else None,attempt_amplification=len(groups)/n if n else None,
        mixed_subtypes=mixed(groups),accepted_mixed_subtypes=mixed(accepted),
        mixed_with_parsed_wrong=sum(g.get('mixed_subtype') in ('mixed_parsed_wrong','mixed_both') for g in groups),
        accepted_with_parsed_wrong=sum(g.get('mixed_subtype') in ('mixed_parsed_wrong','mixed_both') for g in accepted),
        trajectory_accuracy=sum(r['acc'] for r in rows)/len(rows),
        strict_format_rate=sum(r['format_valid'] for r in rows)/len(rows),unparseable_rate=sum(r['parsed_answer'] is None for r in rows)/len(rows),
        ambiguous_rate=sum(r['ambiguous'] for r in rows)/len(rows),parse_errors=dict(Counter(r['parse_error'] for r in rows if r['parse_error'])),
        truncation_rate=sum(r['finish_reason']=='length' for r in rows)/len(rows),
        lengths=distribution([r['output_tokens'] for r in rows]),
        lengths_by_disposition={k:distribution([r['output_tokens'] for r in rows if r['disposition']==k]) for k in DISPOSITIONS},
        reward_std_by_class={k:distribution([g['hybrid_reward_std'] for g in groups if g['classification']==k]) for k in ('all_wrong','mixed','all_correct')},
        correctness_std_by_class={k:distribution([g['correctness_std'] for g in groups if g['classification']==k]) for k in ('all_wrong','mixed','all_correct')},
        homogeneous_acc_nonzero_hybrid_variance={k:sum(g['classification']==k and g['hybrid_reward_std']>0 for g in groups) for k in ('all_wrong','all_correct')},
        answer_cardinality_by_class={str(k):{c:sum(g.get('answer_cardinality')==k and g['classification']==c for g in groups) for c in ('all_wrong','mixed','all_correct')} for k in sorted({g['answer_cardinality'] for g in groups if g['valid']})},
        exposure=dict(generated_unique_prompts=len(exposures),accepted_unique_prompts=len(ae),prompt_repeat_histogram=dict(Counter(map(str,exposures.values()))),
            accepted_repeat_histogram=dict(Counter(map(str,ae.values()))),max_prompt_exposure=max(exposures.values()),prompt_exposure_count=exposures,accepted_exposure_count=ae,
            stage2_formal_overlap=len(set(exposures)&set(s2)),candidate_epoch=state['epoch'],candidate_cursor=state['cursor']),
        costs=dict(generated_output_tokens=tokens,output_tokens_by_disposition=state['output_tokens_by_disposition'],
            rejected_output_tokens=sum(state['output_tokens_by_disposition'][k] for k in ('rejected_all_wrong','rejected_all_correct')),
            accepted_token_fraction=state['output_tokens_by_disposition']['accepted']/tokens if tokens else None,
            prompt_tokens_shared_prefill=state['prompt_tokens_shared_prefill'],logical_prompt_tokens_all_trajectories=sum(r['prompt_tokens'] for r in rows),
            generation_wall_seconds=generation_seconds,output_tokens_per_second=tokens/generation_seconds if generation_seconds else None,
            semantic_scoring_seconds=scoring_seconds,gpu_worker_active_seconds=sum(r['wall_seconds'] for r in runtimes),
            worker_time_scope='Sum of active worker runtime through final/pause checkpoints; excludes human pause gap, includes engine loading and scoring; not measured GPU kernel busy time',
            control_output_tokens=sum(c['output_tokens'] for c in identity),control_prompt_tokens=sum(c['prompt_tokens'] for c in identity),
            control_generation_seconds=sum(c['seconds'] for c in identity),unknown_inflight_attempts=0),
        runtimes=runtimes,refill_batches=state['batches'],optimizer_updates=0,groups=groups)
