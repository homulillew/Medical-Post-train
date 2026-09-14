"""Auxiliary objective contract tests (no model generation)."""
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
from loss_objective_metrics import length_bin,select

def test_length_boundaries():
    assert [length_bin(n) for n in [191,192,255,256,383,384,1024]]==['<192','[192,256)','[192,256)','[256,384)','[256,384)','>=384','>=384']

def test_selection_fixed_order():
    rows=[dict(learning_rate=lr,clip=c,health=dict(healthy=True,active_clipping=True)) for lr in [1e-5,1e-6,3e-6] for c in [.1,.2]]
    assert select(rows,{})['learning_rate']==1e-6
    assert select(rows,{})['clip']==.2
    rows[2]['health']['healthy']=rows[3]['health']['healthy']=False
    assert select(rows,{})['learning_rate']==3e-6

def test_native_token_objective_is_not_sequence_alias():
    import torch
    from verl.trainer.ppo.core_algos import get_policy_loss_fn,compute_policy_loss_vanilla,compute_policy_loss_gspo,compute_grpo_outcome_advantage
    from verl.workers.config import ActorConfig
    assert get_policy_loss_fn('vanilla') is compute_policy_loss_vanilla
    assert compute_policy_loss_vanilla is not compute_policy_loss_gspo
    cfg=ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,clip_ratio_low=.2,clip_ratio_high=.2)
    old=torch.zeros(1,2);cur=torch.tensor([[.5,-.5]],requires_grad=True);a=torch.ones(1,2);mask=torch.ones(1,2)
    token,_=compute_policy_loss_vanilla(old,cur,a,mask,config=cfg,loss_agg_mode='seq-mean-token-mean')
    seq,_=compute_policy_loss_gspo(old,cur,a,mask,config=cfg)
    assert abs(float(token.detach()-seq.detach()))>.05
    tg=torch.autograd.grad(token,cur,retain_graph=True)[0];sg=torch.autograd.grad(seq,cur)[0]
    assert not torch.allclose(tg,sg)
    import ast
    for path in ['scripts/loss_objective_actor.py','src/medical_posttrain/rl/actor.py']:
        src=(ROOT/path).read_text()
        assert 'compute_grpo_outcome_advantage(rewards,mask,np.repeat(np.arange(8),4),' in src
        assert src.index("immutable(directory/'old_frozen.json'")<src.index('for start in range(0,32')
