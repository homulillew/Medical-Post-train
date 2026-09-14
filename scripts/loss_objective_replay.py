"""Native token objective replay derived from the frozen Stage4 evidence checks."""
import json
import numpy as np
from pathlib import Path
from medical_posttrain.rl.common import read,record
from medical_posttrain.evidence import sha256
def update(path,groups,cfg):
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,compute_policy_loss_vanilla
    from verl.workers.config import ActorConfig
    old = np.load(path/'old.npz')
    assert all(np.isfinite(old[k]).all() for k in old.files)
    frozen = read(path/'old_frozen.json')
    assert record(path/'old.npz')==frozen['artifact']
    rows = [r for g in groups for r in g['responses']]
    assert frozen['trajectory_ids']==[r['trajectory_id'] for r in rows]
    mask = old['mask']
    rewards = np.zeros_like(mask)
    for i,r in enumerate(rows):
        length = len(r['token_ids'])
        assert (mask[i,:length]==1).all() and (mask[i,length:]==0).all()
        rewards[i,length-1]=r['total_reward']
    assert np.array_equal(rewards,old['rewards'])
    adv,_ = compute_grpo_outcome_advantage(torch.tensor(rewards),torch.tensor(mask),np.repeat(np.arange(8),4),epsilon=1e-6)
    # A four-element FP32 reduction's rounding is amplified by division by a
    # small reward std. Check independently in FP64 with a forward-error bound,
    # rather than requiring CPU/GPU reduction trees to agree bitwise.
    for start in range(0,32,4):
        r=rewards[start:start+4].astype(np.float64).sum(1)
        std=r.std(ddof=1)
        reference=(r-r.mean())/(std+1e-6)
        observed=old['advantages'][start:start+4,0]
        if std==0:
            assert np.max(np.abs(observed))<=1e-6
        else:
            tolerance=max(1e-6,8*np.finfo(np.float32).eps*max(abs(r))/(std+1e-6))
            assert np.max(np.abs(reference-observed))<=tolerance
        assert np.allclose(old['advantages'][start:start+4],observed[:,None]*mask[start:start+4],atol=0,rtol=0)
    # Recompute the actual native objective using the now independently checked
    # GPU advantage artifact, not a numerically different CPU normalization.
    adv=torch.tensor(old['advantages'])
    config = ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=cfg['clip_ratio_low'],clip_ratio_high=cfg['clip_ratio_high'],
        ppo_mini_batch_size=cfg['mini_prompts'],ppo_epochs=1)
    summary = read(path/'update.json')
    events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    assert frozen['timestamp'] < events[0]['timestamp']
    for j,mini in enumerate(summary['minibatches']):
        evidence = np.load(path/f'mini_{j:02d}.npz')
        assert all(np.isfinite(evidence[k]).all() for k in evidence.files)
        idx = evidence['indices']
        assert idx.tolist()==list(range(j*16,(j+1)*16))
        current = evidence['current_logprobs']
        ratios = np.exp(((current-old['old_logprobs'][idx])*mask[idx]).sum(1)/mask[idx].sum(1))
        assert np.allclose(ratios,evidence['ratios'],atol=1e-7,rtol=1e-6)
        losses,clips = [],[]
        for local,i in enumerate(idx):
            loss,metrics = compute_policy_loss_vanilla(torch.tensor(old['old_logprobs'][i:i+1]),
                torch.tensor(current[local:local+1]),adv[i:i+1],torch.tensor(mask[i:i+1]),config=config,loss_agg_mode='seq-mean-token-mean')
            losses.append(loss.item())
            clips.append(metrics['actor/pg_clipfrac'])
        assert np.allclose(losses,evidence['losses'],atol=1e-6)
        assert np.allclose(clips,evidence['clips'],atol=1e-6)
        assert abs(np.mean(clips)-mini['clip_fraction'])<1e-6
        assert abs(np.mean(losses)-mini['policy_loss'])<1e-6
        assert np.isfinite(mini['grad_norm']) and mini['grad_norm']>=0
        if j==0:
            assert np.max(np.abs(ratios-1))<=1e-6
    assert summary['parameter_delta_l2']>0 and summary['initial_trainable_digest']!=summary['final_trainable_digest']
    return summary
