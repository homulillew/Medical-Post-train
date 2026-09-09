"""Pure window selection and committed-lineage checks for Stage4."""
from collections import Counter
from copy import deepcopy

from medical_posttrain.sampling.dynamic import classify


def select_groups(groups, policy_version, mode, already=0, target=8):
    if mode not in ('vanilla','dynamic') or not 0<=already<=target:
        raise ValueError('Invalid window mode or accepted count')
    decisions, selected = [], []
    for group in groups:
        rows = group['responses']
        c = classify(rows,policy_version)
        if any(r.get('group_id') != group['group_id'] or r.get('prompt_id') != group['prompt_id'] for r in rows):
            c = dict(valid=False,classification='invalid',eligible=False,reason='envelope_identity',mixed_subtype=None)
        if not c['valid']:
            disposition = 'invalid'
        elif mode == 'dynamic' and not c['eligible']:
            disposition = 'rejected_'+c['classification']
        elif already+len(selected) == target:
            disposition = 'overflow_eligible'
        else:
            disposition = 'selected'
            selected.append(group)
        decisions.append(dict(group_id=group['group_id'],prompt_id=group['prompt_id'],
                              encounter_index=group['encounter_index'],disposition=disposition,**c))
    return selected,decisions


def window_metrics(groups, decisions):
    from .actor import array_stats
    rows = [r for g in groups for r in g['responses']]
    valid = [d for d in decisions if d['valid']]
    classes = Counter(d['classification'] for d in valid)
    dispositions = {d['group_id']:d['disposition'] for d in decisions}
    tokens = Counter()
    prompt_tokens = 0
    for g in groups:
        tokens[dispositions[g['group_id']]] += sum(r['output_tokens'] for r in g['responses'])
        prompt_tokens += g['prompt_tokens']
    selected = sum(d['disposition']=='selected' for d in decisions)
    return dict(generated_groups=len(groups),valid_generated_groups=len(valid),
        generated_trajectories=len(rows),selected_groups=selected,
        group_counts=dict(classes),correct_count_histogram=dict(Counter(str(d['correct_count']) for d in valid)),
        mixed_subtypes=dict(Counter(d['mixed_subtype'] for d in valid if d['mixed_subtype'])),
        output_tokens=sum(tokens.values()),output_tokens_by_disposition=dict(tokens),
        prompt_tokens=prompt_tokens,logical_trajectory_prompt_tokens=prompt_tokens*4,
        sampling_amplification=len(valid)/selected if selected else None,
        reward=array_stats([r['total_reward'] for r in rows]),
        acc=sum(r['acc'] for r in rows)/len(rows),semantic=sum(r['semantic'] for r in rows)/len(rows),
        semantic_contribution=sum(r['semantic_contribution'] for r in rows)/len(rows),
        strict_format=sum(r['format'] for r in rows)/len(rows),
        unparseable=sum(r['parsed_answer'] is None for r in rows)/len(rows),
        think_closure=sum(r['think_closed'] for r in rows)/len(rows),
        answer_closure=sum(r['answer_closed'] for r in rows)/len(rows),
        truncation=sum(r['finish_reason']=='length' for r in rows)/len(rows),
        response_length=array_stats([r['output_tokens'] for r in rows]))


def advance(state, groups, decisions, output_policy, optimizer_steps):
    if len([d for d in decisions if d['disposition']=='selected']) != 8:
        raise ValueError('Cannot commit an incomplete training window')
    if output_policy == state['policy_version']:
        raise ValueError('Policy digest did not change')
    if any(r['policy_version'] != state['policy_version'] for g in groups for r in g['responses']):
        raise ValueError('Mixed policy window')
    after = deepcopy(state)
    assert [g['encounter_index'] for g in groups] == list(range(state['cursor'],state['cursor']+len(groups)))
    after['cursor'] += len(groups)
    after['policy_windows'] += 1
    after['optimizer_steps'] += optimizer_steps
    after['training_groups'] += 8
    after['generated_groups'] += len(groups)
    after['output_tokens'] += sum(r['output_tokens'] for g in groups for r in g['responses'])
    after['prompt_tokens'] += sum(g['prompt_tokens'] for g in groups)
    after['policy_version'] = output_policy
    selected = {d['group_id'] for d in decisions if d['disposition']=='selected'}
    for g in groups:
        pid = g['prompt_id']
        after['generated_exposure'][pid] = after['generated_exposure'].get(pid,0)+1
        if g['group_id'] in selected:
            after['training_exposure'][pid] = after['training_exposure'].get(pid,0)+1
    return after
