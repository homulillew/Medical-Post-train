"""Pure accuracy-only group filtering and deterministic encounter stream."""
from collections import Counter
from copy import deepcopy
import hashlib

IDENTITY = ('prompt_id', 'group_id', 'policy_version', 'config_sha256', 'reward_version')
DISPOSITIONS = ('accepted', 'rejected_all_wrong', 'rejected_all_correct', 'overflow_eligible', 'invalid')


def validity(rows, policy_version):
    if len(rows) != 4:
        return False, 'incomplete_group'
    if any(any(k not in r for k in (*IDENTITY, 'trajectory_id', 'acc', 'finish_reason')) for r in rows):
        return False, 'missing_field'
    if len({r['trajectory_id'] for r in rows}) != 4:
        return False, 'duplicate_trajectory'
    for key in IDENTITY:
        if len({r[key] for r in rows}) != 1:
            return False, 'mixed_' + key
    if rows[0]['policy_version'] != policy_version:
        return False, 'policy_barrier'
    if any(r.get('transport_error') or r['finish_reason'] not in ('stop', 'length') for r in rows):
        return False, 'incomplete_generation'
    if any(type(r['acc']) not in (int, float) or r['acc'] not in (0, 1) for r in rows):
        return False, 'nonbinary_acc'
    return True, None


def eligible(acc_vector):
    if len(acc_vector) != 4 or any(type(a) not in (int, float) or a not in (0, 1) for a in acc_vector):
        raise ValueError('Eligibility requires exactly four binary accuracy values')
    return 0 < sum(acc_vector) < 4


def classify(rows, policy_version):
    valid, reason = validity(rows, policy_version)
    if not valid:
        return dict(valid=False, reason=reason, eligible=False, classification='invalid', mixed_subtype=None)
    acc = [r['acc'] for r in rows]
    count = int(sum(acc))
    mixed = eligible(acc)
    parsed_wrong = any(r['acc'] == 0 and r.get('parsed_answer') is not None for r in rows)
    unparsed = any(r['acc'] == 0 and r.get('parsed_answer') is None for r in rows)
    subtype = ('mixed_both' if parsed_wrong and unparsed else 'mixed_parsed_wrong' if parsed_wrong else 'mixed_unparseable_only') if mixed else None
    return dict(valid=True, reason=None, eligible=mixed, correct_count=count,
                classification='mixed' if mixed else 'all_wrong' if count == 0 else 'all_correct', mixed_subtype=subtype)


class Stream:
    def __init__(self, prompt_ids, seed, domain):
        if not prompt_ids or len(prompt_ids) != len(set(prompt_ids)):
            raise ValueError('Pool must be nonempty and unique')
        self.ids, self.seed, self.domain = list(prompt_ids), seed, domain
        self.orders = {}

    def order(self, epoch):
        if epoch not in self.orders:
            self.orders[epoch] = sorted(self.ids, key=lambda pid: hashlib.sha256(f'{self.seed}:{self.domain}:{epoch}:{pid}'.encode()).digest())
        return self.orders[epoch]

    def encounter(self, index, run_id, policy_version):
        epoch, cursor = divmod(index, len(self.ids))
        pid = self.order(epoch)[cursor]
        gid = f'{run_id}:{policy_version[:12]}:encounter:{index:08d}'
        seed = int.from_bytes(hashlib.sha256(f'{self.seed}:{self.domain}:{policy_version}:{index}:{pid}'.encode()).digest()[:4], 'big')
        return dict(prompt_id=pid, group_id=gid, encounter_index=index, candidate_epoch=epoch, candidate_cursor=cursor, request_seed=seed)


def initial_state(config):
    return dict(seed=config['seed'], domain=config['stream_domain'], policy_version=config['policy_version'],
                epoch=0, cursor=0, encounter_index=0, batches=0, generated_groups=0, valid_generated_groups=0,
                completed_trajectories=0, completed_valid_trajectories=0, accepted_mixed_groups=0,
                all_wrong_groups=0, all_correct_groups=0, mixed_groups=0, overflow_mixed_groups=0, invalid_groups=0,
                output_tokens_by_disposition={k:0 for k in DISPOSITIONS}, prompt_tokens_shared_prefill=0,
                prompt_exposures={}, accepted_exposures={}, dispositions={})


def apply_batch(state, groups, config, pool_size):
    state = deepcopy(state)
    if state['policy_version'] != config['policy_version']:
        raise ValueError('Policy version barrier: create a fresh window/state for a new policy')
    decisions = []
    for group in groups:
        rows = group['responses']; gid = group['group_id']; pid = group['prompt_id']
        if gid in state['dispositions'] or group['encounter_index'] != state['encounter_index']:
            raise ValueError('Duplicate/out-of-order encounter')
        result = classify(rows, config['policy_version'])
        if any(r.get('group_id') != gid or r.get('prompt_id') != pid for r in rows):
            result = dict(valid=False, reason='envelope_identity', eligible=False, classification='invalid', mixed_subtype=None)
        disposition = 'invalid'
        if result['valid']:
            state['valid_generated_groups'] += 1
            state['completed_valid_trajectories'] += 4
            state[result['classification'] + '_groups'] += 1
            if result['eligible']:
                if config['target_accepted'] is None or state['accepted_mixed_groups'] < config['target_accepted']:
                    disposition = 'accepted'; state['accepted_mixed_groups'] += 1
                    state['accepted_exposures'][pid] = state['accepted_exposures'].get(pid, 0) + 1
                else:
                    disposition = 'overflow_eligible'; state['overflow_mixed_groups'] += 1
            else:
                disposition = 'rejected_' + result['classification']
        else:
            state['invalid_groups'] += 1
        state['generated_groups'] += 1
        state['completed_trajectories'] += sum(r.get('finish_reason') in ('stop', 'length') and not r.get('transport_error') for r in rows)
        state['output_tokens_by_disposition'][disposition] += sum(r['output_tokens'] for r in rows)
        state['prompt_tokens_shared_prefill'] += group['prompt_tokens']
        state['prompt_exposures'][pid] = state['prompt_exposures'].get(pid, 0) + 1
        state['dispositions'][gid] = disposition
        state['encounter_index'] += 1
        state['epoch'], state['cursor'] = divmod(state['encounter_index'], pool_size)
        decisions.append(dict(group_id=gid, prompt_id=pid, encounter_index=group['encounter_index'], disposition=disposition, **result))
    state['batches'] += 1
    return state, decisions


def stop_reason(state, config):
    if state['invalid_groups']:
        return 'FAILED_INVALID_GROUP'
    if config['mode'] == 'smoke' and state['generated_groups'] >= config['smoke_prompts']:
        return 'SMOKE_PASS'
    if config['target_accepted'] is not None and state['accepted_mixed_groups'] == config['target_accepted']:
        return 'FULL_PASS'
    if state['batches'] >= config['max_num_generation_batches']:
        return 'BLOCKED_STARVATION'
    return None
