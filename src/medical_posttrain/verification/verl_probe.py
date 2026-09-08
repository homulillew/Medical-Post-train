import asyncio
import json
import os
from pathlib import Path
import time
from medical_posttrain.evidence import write_json

def actor_config():
    from verl.workers.config import ActorConfig
    return ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,clip_ratio_low=.0003,clip_ratio_high=.0004,ppo_mini_batch_size=8,ppo_epochs=1)

def numeric(ev):
    import torch,numpy as np
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo,compute_grpo_outcome_advantage
    result=[]
    for device in ('cpu','cuda'):
        mask=torch.tensor([[1,1,0],[1,1,1],[1,0,0],[1,1,0],[1,1,1],[1,0,0]],device=device,dtype=torch.float64)
        ratios=torch.tensor([1.2,.8,1.2,.8,1.,1.0001],device=device,dtype=torch.float64)
        adv=torch.tensor([1,-1,-1,1,1,-1],device=device,dtype=torch.float64)[:,None].expand_as(mask)
        old=torch.full_like(mask,-2)
        delta=ratios.log()[:,None].expand_as(mask).clone(); delta[0,:2]+=torch.tensor([.1,-.1],device=device)
        delta[mask==0]=99 # adversarial padding must not affect ratio/loss/gradient
        current=(old+delta).requires_grad_()
        loss,metrics=compute_policy_loss_gspo(old,current,adv,mask,config=actor_config())
        expected=torch.maximum(-adv[:,0]*ratios,-adv[:,0]*ratios.clamp(1-.0003,1+.0004)).mean()
        assert torch.allclose(loss,expected,atol=1e-10)
        loss.backward(); assert torch.isfinite(current.grad).all() and (current.grad[mask==0]==0).all()
        assert (current.grad[:2]==0).all() and current.grad[2,0]>0 and current.grad[3,0]<0
        rewards=torch.tensor([[0.,0.,0.],[0,0,1.],[0,0,0.],[0,0,1.]],device=device)
        a,_=compute_grpo_outcome_advantage(rewards,torch.ones_like(rewards),np.array([0,0,1,1]),norm_adv_by_std_in_grpo=True)
        assert a[0,0]<0 and a[1,0]>0 and torch.isfinite(a).all()
        result.append(ev.metric(event='gspo_numeric',device=device,loss=float(loss.detach()),expected=float(expected),ratios=ratios.tolist(),finite_grad=True,padding_grad_zero=True,**metrics))
    return dict(gates={'gspo_numeric':True},results=result)

def compose(ev):
    from hydra import compose,initialize_config_module
    from omegaconf import OmegaConf
    with initialize_config_module(config_module='verl.trainer.config',version_base=None):
        cfg=compose(config_name='ppo_trainer',overrides=[
            'actor_rollout_ref.actor.strategy=fsdp2', 'algorithm.adv_estimator=grpo',
            'actor_rollout_ref.actor.policy_loss.loss_mode=gspo',
            'actor_rollout_ref.actor.clip_ratio_low=0.0003','actor_rollout_ref.actor.clip_ratio_high=0.0004',
            'actor_rollout_ref.actor.ppo_mini_batch_size=8','actor_rollout_ref.actor.ppo_epochs=1',
            'actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean',
            f'actor_rollout_ref.model.path={ev.config["model"]}',
            'actor_rollout_ref.model.lora_rank=32','actor_rollout_ref.model.lora_alpha=64',
            'actor_rollout_ref.model.use_remove_padding=False',
            'actor_rollout_ref.rollout.name=vllm','actor_rollout_ref.rollout.n=4',
            'actor_rollout_ref.rollout.tensor_model_parallel_size=1','+actor_rollout_ref.rollout.enable_sleep_mode=True',
            'trainer.n_gpus_per_node=1','trainer.nnodes=1'])
    (ev.path/'verl_composed.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=False))
    assert cfg.actor_rollout_ref.actor.policy_loss.loss_mode=='gspo'
    return cfg

async def reward_interface(ev,cfg,tok):
    import torch,numpy as np
    from verl import DataProto
    from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager
    from medical_posttrain.data.format import parse_answer
    def score(data_source,solution_str,ground_truth,extra_info):
        assert data_source=='stage0_synthetic'
        parsed=parse_answer(solution_str)
        # Interface fixture: no semantic model or medical reward quality claimed.
        acc=float(parsed==ground_truth)
        return dict(score=acc,acc=acc,sem=0.,format=float(parsed is not None))
    ids=torch.tensor([tok.encode('<answer>C</answer>',add_special_tokens=False)])
    data=DataProto.from_dict(tensors={'responses':ids,'attention_mask':torch.ones_like(ids)},non_tensors={'data_source':np.array(['stage0_synthetic'],dtype=object),'reward_model':np.array([{'ground_truth':'C'}],dtype=object)})
    got=await NaiveRewardManager(cfg,tok,score).run_single(data)
    assert got['reward_extra_info']==dict(score=1.,acc=1.,sem=0.,format=1.)
    ev.metric(event='reward_dict',result=got)

def run(ev):
    if ev.config['purpose']=='numeric': return numeric(ev)
    import torch
    from transformers import AutoTokenizer
    from verl.workers.config import HFModelConfig,FSDPEngineConfig,FSDPOptimizerConfig
    from verl.trainer.config import CheckpointConfig
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
    from medical_posttrain.training.lora import TARGETS,MemoryMonitor,batch
    cfg=compose(ev)
    tok=AutoTokenizer.from_pretrained(ev.config['model'],local_files_only=True)
    asyncio.run(reward_interface(ev,cfg,tok))
    torch.distributed.init_process_group('nccl',init_method='file://'+str(ev.bulk/'dist_init'),rank=0,world_size=1)
    try:
        mc=HFModelConfig(path=ev.config['model'],lora_rank=32,lora_alpha=64,target_modules=TARGETS,use_remove_padding=False,use_fused_kernels=False,override_config={'attn_implementation':'sdpa'},lora_adapter_path=ev.config['adapter'])
        ec=FSDPEngineConfig(strategy='fsdp2',model_dtype='bf16',param_offload=False,optimizer_offload=False,offload_policy=False,use_torch_compile=False)
        oc=FSDPOptimizerConfig(lr=1e-5,total_training_steps=3)
        engine=FSDPEngineWithLMHead(mc,ec,oc,CheckpointConfig())
        t=time.monotonic()
        with MemoryMonitor() as mm:
            engine.initialize()
            model=engine.module
            model.config.use_cache=False
            engine.optimizer_zero_grad()
            data=batch(tok,128)
            loss=model(**data).loss; loss.backward()
            grad=engine.optimizer_step()
            torch.cuda.synchronize()
        assert torch.isfinite(loss)
        ev.metric(event='native_verl_fsdp2',loss=float(loss.detach()),grad_norm=float(grad),seconds=time.monotonic()-t,**mm.result())
        return dict(gates={'verl_import_compose':True,'verl_fsdp2':True,'reward_dict':True},strategy='fsdp2',attention='sdpa',remove_padding=False,loss=float(loss.detach()))
    finally:
        torch.distributed.destroy_process_group()
