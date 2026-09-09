"""Native verl FSDP2/GRPO-advantage/GSPO actor with explicit old-policy barrier.

The wrapper controls durable windows and microbatch evidence. Loss and optimizer
math are delegated to installed verl; no separate reference model is instantiated.
"""
import json
import os
from pathlib import Path
import time

from .common import read, immutable, event, record, durable


def config_value(value):
    """Serialize native runtime configs without pretending tokenizer objects are JSON."""
    import torch
    from dataclasses import is_dataclass,asdict
    if is_dataclass(value):
        return config_value(asdict(value))
    if isinstance(value,dict):
        return {str(k):config_value(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,set)):
        return [config_value(v) for v in value]
    if isinstance(value,torch.dtype):
        return str(value)
    if hasattr(value,'to_dict'):
        return config_value(value.to_dict())
    if hasattr(value,'name_or_path'):
        return dict(runtime_class=type(value).__name__,name_or_path=value.name_or_path)
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError(f'Unsupported config field type: {type(value)}')


def array_stats(values):
    import numpy as np
    a = np.asarray(values,dtype=np.float64)
    assert a.size and np.isfinite(a).all()
    return dict(mean=float(a.mean()),std=float(a.std()),min=float(a.min()),max=float(a.max()),
                p05=float(np.percentile(a,5)),p50=float(np.percentile(a,50)),p95=float(np.percentile(a,95)),
                p99=float(np.percentile(a,99)))


def trainable_state(model):
    import torch
    from torch.distributed.tensor import DTensor
    return {n:(p.full_tensor() if isinstance(p,DTensor) else p).detach().cpu().clone()
            for n,p in model.named_parameters() if p.requires_grad}


def optimizer_digest(engine):
    import hashlib
    import torch
    from torch.distributed.tensor import DTensor
    h = hashlib.sha256()
    for name,p in engine.module.named_parameters():
        if not p.requires_grad:
            continue
        h.update(name.encode())
        for key,value in sorted(engine.optimizer.state.get(p,{}).items()):
            h.update(key.encode())
            if isinstance(value,DTensor):
                value=value.full_tensor()
            if isinstance(value,torch.Tensor):
                h.update(str(value.dtype).encode())
                h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
            else:
                h.update(repr(value).encode())
    return h.hexdigest()


def compose_config(cfg, directory, adapter):
    from hydra import compose, initialize_config_module
    from omegaconf import OmegaConf
    with initialize_config_module(config_module='verl.trainer.config',version_base=None):
        c = compose(config_name='ppo_trainer',overrides=[
            'actor_rollout_ref.actor.strategy=fsdp2','algorithm.adv_estimator=grpo',
            'algorithm.use_kl_in_reward=False','actor_rollout_ref.actor.use_kl_loss=False',
            'actor_rollout_ref.actor.entropy_coeff=0',
            'actor_rollout_ref.actor.policy_loss.loss_mode=gspo',
            f'actor_rollout_ref.actor.clip_ratio_low={cfg["clip_ratio_low"]}',
            f'actor_rollout_ref.actor.clip_ratio_high={cfg["clip_ratio_high"]}',
            f'actor_rollout_ref.actor.ppo_mini_batch_size={cfg["mini_prompts"]}',
            'actor_rollout_ref.actor.ppo_epochs=1',
            'actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean',
            f'actor_rollout_ref.actor.optim.lr={cfg["learning_rate"]}',
            f'actor_rollout_ref.model.path={cfg["model"]}',
            'actor_rollout_ref.model.lora_rank=32','actor_rollout_ref.model.lora_alpha=64',
            'actor_rollout_ref.model.use_remove_padding=False',
            'actor_rollout_ref.rollout.name=vllm','actor_rollout_ref.rollout.n=4',
            'actor_rollout_ref.rollout.temperature=0.6','actor_rollout_ref.rollout.tensor_model_parallel_size=1',
            'trainer.n_gpus_per_node=1','trainer.nnodes=1'])
    (directory/'verl_composed.yaml').write_text(OmegaConf.to_yaml(c,resolve=False))
    immutable(directory/'orchestration.json',dict(reference_worker_created=False,critic_created=False,
        driver='local single-rank FSDP2 engine; no Ray trainer instantiated',
        adapter=str(adapter),ppo_mini_batch_size_unit='prompts; wrapper expands by G4',
        old='teacher-forced generating adapter before ALL optimizer steps',
        logprob='FP32 log_softmax(logits.float()/0.6); only response tokens including EOS'))


class Actor:
    def __init__(self, cfg, directory, adapter, checkpoint=None):
        import torch
        from verl.workers.config import HFModelConfig,FSDPEngineConfig,FSDPOptimizerConfig,ActorConfig
        from verl.trainer.config import CheckpointConfig
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngineWithLMHead
        from medical_posttrain.training.lora import seed_all,TARGETS
        self.cfg,self.directory = cfg,Path(directory)
        self.directory.mkdir(parents=True,exist_ok=True)
        seed_all(cfg['seed'])
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group('nccl',init_method='file://'+str(self.directory.resolve()/'dist_init'),rank=0,world_size=1)
        compose_config(cfg,self.directory,adapter)
        mc = HFModelConfig(path=cfg['model'],lora_rank=32,lora_alpha=64,target_modules=TARGETS,
            use_remove_padding=False,use_fused_kernels=False,override_config={'attn_implementation':'sdpa'},
            lora_adapter_path=str(adapter),enable_gradient_checkpointing=True)
        ec = FSDPEngineConfig(strategy='fsdp2',model_dtype='bf16',param_offload=False,
            optimizer_offload=False,offload_policy=False,use_torch_compile=False)
        oc = FSDPOptimizerConfig(lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'],
            betas=tuple(cfg['betas']),clip_grad=cfg['grad_clip'],lr_warmup_steps=0,
            lr_scheduler_type='constant',total_training_steps=1250,
            override_optimizer_config={'eps':cfg['eps']})
        cc = CheckpointConfig(save_lora_only=True)
        self.engine = FSDPEngineWithLMHead(mc,ec,oc,cc)
        self.engine.initialize()
        self.model = self.engine.module
        self.model.config.use_cache = False
        self.model.train()
        self.loss_config = ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,
            clip_ratio_low=cfg['clip_ratio_low'],clip_ratio_high=cfg['clip_ratio_high'],
            ppo_mini_batch_size=cfg['mini_prompts'],ppo_epochs=1)
        immutable(self.directory/'actual_native_configs.json',dict(
            model=config_value(mc),engine=config_value(ec),optimizer=config_value(oc),
            checkpoint=config_value(cc),loss=config_value(self.loss_config)))
        if checkpoint:
            self.engine.load_checkpoint(str(Path(checkpoint)/'native'),del_local_after_load=False)
        params = trainable_state(self.model)
        assert sum(v.numel() for v in params.values()) == 87293952
        assert all(v.dtype == torch.float32 for v in params.values())
        assert all(getattr(m,'p',0)==0 for m in self.model.modules() if isinstance(m,torch.nn.Dropout))
        immutable(self.directory/'loaded_identity.json',dict(
            trainable_digest=self.digest(params),trainable_parameters=87293952,
            optimizer_digest=optimizer_digest(self.engine),
            trainable_dtypes=sorted({str(v.dtype) for v in params.values()}),
            optimizer_steps=sorted({int(v['step'].item()) for v in self.engine.optimizer.state.values() if 'step' in v}),
            scheduler=self.engine.lr_scheduler.state_dict(),checkpoint=str(checkpoint) if checkpoint else None))

    @staticmethod
    def digest(params):
        from medical_posttrain.training.lora import digest
        return digest(params)

    def logprobs(self, row, entropy=False):
        import torch
        p,r = row['prompt_token_ids'],row['token_ids']
        ids = torch.tensor([p+r],device='cuda')
        # Unpadded microbatch1 avoids EOS/PAD ambiguity; the last generated EOS
        # remains supervised even if the tokenizer also uses it for padding.
        output = self.model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False)
        logits = output.logits[0,len(p)-1:-1].float()
        assert logits.shape[0] == len(r)
        labels = torch.tensor(r,device='cuda')[:,None]
        raw = torch.log_softmax(logits,dim=-1).gather(1,labels).squeeze(1) if not torch.is_grad_enabled() else None
        lp_all = torch.log_softmax(logits/self.cfg['sampling']['temperature'],dim=-1)
        lp = lp_all.gather(1,labels).squeeze(1)
        ent = -(lp_all.detach().exp()*lp_all.detach()).sum(-1) if entropy else None
        return lp,raw,ent

    def update(self, groups, directory, fixed_old=None, check_parity=False, step_before=0):
        import numpy as np
        import torch
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,compute_policy_loss_gspo
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        rows = [r for g in groups for r in g['responses']]
        assert len(groups) == 8 and len(rows) == 32
        assert len({r['policy_version'] for r in rows}) == 1
        initial = trainable_state(self.model)
        max_r = max(len(r['token_ids']) for r in rows)
        mask = torch.zeros(32,max_r,device='cuda')
        rewards = torch.zeros_like(mask)
        for i,r in enumerate(rows):
            mask[i,:len(r['token_ids'])] = 1
            rewards[i,len(r['token_ids'])-1] = r['total_reward']
        advantages,_ = compute_grpo_outcome_advantage(rewards,mask,np.repeat(np.arange(8),4),
            epsilon=1e-6,norm_adv_by_std_in_grpo=True)
        assert torch.isfinite(advantages).all()
        old = torch.zeros_like(mask)
        raws = torch.zeros_like(mask)
        start = time.monotonic()
        with torch.no_grad():
            for i,r in enumerate(rows):
                lp,raw,_ = self.logprobs(r)
                old[i,:len(lp)] = lp
                raws[i,:len(raw)] = raw
        old_seconds = time.monotonic()-start
        if fixed_old:
            frozen = np.load(fixed_old)
            # Same full batch and freshly restored SFT must reproduce old exactly.
            assert np.array_equal(old.cpu().numpy(),frozen['old_logprobs'])
            assert np.array_equal(advantages.cpu().numpy(),frozen['advantages'])
            assert np.array_equal(mask.cpu().numpy(),frozen['mask'])
            old = torch.from_numpy(frozen['old_logprobs']).to('cuda')
        np.savez_compressed(directory/'old.npz',old_logprobs=old.cpu().numpy(),
            raw_logprobs=raws.cpu().numpy(),advantages=advantages.cpu().numpy(),
            mask=mask.cpu().numpy(),rewards=rewards.cpu().numpy())
        immutable(directory/'old_frozen.json',dict(artifact=record(directory/'old.npz'),
            policy_version=rows[0]['policy_version'],trajectory_ids=[r['trajectory_id'] for r in rows],
            optimizer_step=step_before,seconds=old_seconds,timestamp=__import__('medical_posttrain.evidence',fromlist=['now']).now()))
        if check_parity:
            diffs = [raws[i,:len(r['token_ids'])].cpu().numpy()-np.asarray(r['rollout_raw_logprobs']) for i,r in enumerate(rows)]
            absolute = np.abs(np.concatenate(diffs))
            seq = np.abs([d.mean() for d in diffs])
            limits = self.cfg['parity_limits']
            ok = absolute.mean()<=limits['token_mean_abs'] and np.percentile(absolute,99)<=limits['token_p99_abs'] and max(seq)<=limits['sequence_mean_abs_max']
            immutable(directory/'parity.json',dict(result='PASS' if ok else 'FAIL',
                raw_token_abs=array_stats(absolute),sequence_mean_abs=array_stats(seq),limits=limits,
                raw_temperature=1,optimization_temperature=.6,old_source='actor recomputation before any update'))
            assert ok, 'Cross-engine raw logprob parity investigation required'
        minibatches = []
        for start in range(0,32,self.cfg['mini_prompts']*4):
            end = start+self.cfg['mini_prompts']*4
            self.engine.optimizer_zero_grad()
            began = time.monotonic()
            ratios,losses,clips,entropies,drifts = [],[],[],[],[]
            currents = []
            for i in range(start,end):
                lp,_,ent = self.logprobs(rows[i],entropy=True)
                current = torch.nn.functional.pad(lp,(0,max_r-len(lp)))[None]
                loss,m = compute_policy_loss_gspo(old[i:i+1],current,advantages[i:i+1],mask[i:i+1],config=self.loss_config)
                assert torch.isfinite(loss)
                (loss/(end-start)).backward()
                delta = (current.detach()-old[i:i+1])[0,:len(lp)]
                ratios.append(float(delta.mean().exp()))
                losses.append(float(loss.detach()))
                clips.append(m['actor/pg_clipfrac'])
                entropies.append(float(ent.mean()))
                drifts.append(float(delta.abs().mean()))
                currents.append(current.detach().cpu().numpy()[0])
            if start == 0:
                assert max(abs(r-1) for r in ratios) <= 1e-6, 'First mini is not current==old'
            grad = self.engine.optimizer_step()
            assert np.isfinite(grad), 'Native optimizer skipped nonfinite gradient: abort window'
            lr = self.engine.lr_scheduler_step()
            torch.cuda.synchronize()
            np.savez_compressed(directory/f'mini_{start//(end-start):02d}.npz',
                current_logprobs=np.stack(currents),indices=np.arange(start,end),
                ratios=np.asarray(ratios),losses=np.asarray(losses),clips=np.asarray(clips),entropy=np.asarray(entropies))
            row = dict(mini_index=start//(end-start),optimizer_step=step_before+len(minibatches)+1,
                trajectories=end-start,ratio=array_stats(ratios),ratios=ratios,
                clip_fraction=float(np.mean(clips)),ratio_bound_exceedance=float(np.mean(
                    [(r<1-self.cfg['clip_ratio_low'] or r>1+self.cfg['clip_ratio_high']) for r in ratios])),
                policy_loss=float(np.mean(losses)),grad_norm=grad,entropy=float(np.mean(entropies)),
                mean_absolute_logprob_drift=float(np.mean(drifts)),lr=lr,seconds=time.monotonic()-began)
            minibatches.append(row)
            event(directory,'optimizer_step',**row)
        after = trainable_state(self.model)
        delta = sum((after[k]-v).double().square().sum().item() for k,v in initial.items())**.5
        assert delta > 0
        summary = dict(old_logprob_seconds=old_seconds,minibatches=minibatches,
            parameter_delta_l2=delta,initial_trainable_digest=self.digest(initial),
            final_trainable_digest=self.digest(after),advantage=array_stats(advantages[:,0].cpu().numpy()),
            optimizer_steps_after=step_before+len(minibatches),training_groups=8,training_trajectories=32)
        immutable(directory/'update.json',summary)
        return summary

    def save(self, directory, step, metadata):
        import torch
        directory = Path(directory)
        assert not directory.exists()
        temp = directory.with_name('.tmp-'+directory.name)
        temp.mkdir()
        start = time.monotonic()
        self.engine.save_checkpoint(str(temp/'native'),global_step=step)
        state = trainable_state(self.model)
        self.model.save_pretrained(temp/'adapter',state_dict=state,safe_serialization=True)
        # Preserve Python/NumPy/CPU/CUDA explicitly as well as upstream extra.
        from medical_posttrain.training.lora import rng_state
        torch.save(rng_state(),temp/'rng.pt')
        immutable(temp/'controller.json',metadata)
        records = []
        for p in sorted(temp.rglob('*')):
            if p.is_file():
                with p.open('rb') as f:
                    os.fsync(f.fileno())
                ref = record(p)
                ref['path'] = str(p.relative_to(temp))
                records.append(ref)
        immutable(temp/'COMMITTED.json',dict(files=records,optimizer_step=step,
            trainable_digest=self.digest(state),optimizer_digest=optimizer_digest(self.engine),
            scheduler=self.engine.lr_scheduler.state_dict(),seconds=time.monotonic()-start))
        os.rename(temp,directory)
        fd = os.open(directory.parent,os.O_DIRECTORY)
        os.fsync(fd)
        os.close(fd)
        return record(directory/'adapter/adapter_model.safetensors')

    def close(self):
        import torch
        torch.distributed.destroy_process_group()
