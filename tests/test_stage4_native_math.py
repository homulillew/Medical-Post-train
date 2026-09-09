"""Independent oracles for native GSPO accumulation and full-reward advantage."""
import numpy as np
import pytest
torch=pytest.importorskip('torch')
pytest.importorskip('verl')
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo,compute_grpo_outcome_advantage
from verl.workers.config import ActorConfig


def test_native_sequence_gradient_and_microbatch_weighting():
    cfg=ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,
        ppo_mini_batch_size=4,clip_ratio_low=.0003,clip_ratio_high=.0004)
    mask=torch.tensor([[1.,0,0],[1,1,1],[1,1,0],[1,1,1]],dtype=torch.float64)
    old=torch.full_like(mask,-2.)
    changes=torch.tensor([[.1,99,99],[-.1,-.1,-.1],[.2,-.1998,99],[0,0,0]],dtype=torch.float64)
    adv=torch.tensor([1.,-1.,1.,-1.],dtype=torch.float64)[:,None]*mask
    x=(old+changes).requires_grad_()
    loss,_=compute_policy_loss_gspo(old,x,adv,mask,config=cfg)
    loss.backward()
    ratio=(((x.detach()-old)*mask).sum(1)/mask.sum(1)).exp()
    # Native seq token-mean retains an explicit1e-8 denominator stabilizer.
    length=mask.sum(1)
    objective=(torch.maximum(-adv[:,0]*ratio,-adv[:,0]*ratio.clamp(.9997,1.0004))*length/(length+1e-8)).mean()
    assert torch.allclose(loss,objective,atol=1e-12)
    gradient=x.grad.clone()
    assert (gradient[mask==0]==0).all()
    assert (gradient[:2]==0).all()
    y=x.detach().clone().requires_grad_()
    micro_losses=[]
    for i in range(4):
        micro,_=compute_policy_loss_gspo(old[i:i+1],y[i:i+1],adv[i:i+1],mask[i:i+1],config=cfg)
        (micro/4).backward()
        micro_losses.append(micro.detach())
    assert torch.allclose(torch.stack(micro_losses).mean(),loss)
    assert torch.allclose(y.grad,gradient,atol=1e-12)


def test_full_hybrid_sample_std_not_binary_accuracy():
    rewards=torch.tensor([[.80],[.92],[.0],[.05],[.85],[.86],[.88],[.94]])
    advantage,_=compute_grpo_outcome_advantage(rewards,torch.ones_like(rewards),np.repeat(np.arange(2),4))
    for i in (0,4):
        group=rewards[i:i+4,0]
        expected=(group-group.mean())/(group.std(correction=1)+1e-6)
        assert torch.allclose(advantage[i:i+4,0],expected)
    # The all-correct group still retains semantic shaping variance.
    assert advantage[4:8,0].std()>0
    assert advantage[0,0]!=advantage[1,0]
