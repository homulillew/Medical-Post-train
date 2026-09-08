"""Stage 0 only: native verl GSPO loss, 8 synthetic groups, exactly 3 Adam updates."""
import json
import time
from medical_posttrain.evidence import write_json

def run(ev):
    import torch,numpy as np
    from medical_posttrain.training.lora import load,parameters,seed_all
    from medical_posttrain.verification.verl_probe import actor_config
    from verl.trainer.ppo.core_algos import compute_policy_loss_gspo,compute_grpo_outcome_advantage
    model,tok=load(ev,ev.config['adapter'])
    initial=parameters(model)
    samples=[]
    for group in range(8):
        prompt=tok.apply_chat_template([{'role':'user','content':f'合成诊断 {group}：请选择 C，使用 answer 标签。'}],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
        for member in range(4):
            response=tok.encode('<think>\n格式检查。\n</think>\n\n<answer>'+('C' if member%2 else 'A')+'</answer>',add_special_tokens=False)
            samples.append((prompt,response))
    max_r=max(len(r) for p,r in samples)
    masks=torch.zeros(32,max_r,device='cuda')
    rewards=torch.zeros_like(masks)
    for i,(p,r) in enumerate(samples): masks[i,:len(r)]=1; rewards[i,len(r)-1]=float(i%2)
    adv,_=compute_grpo_outcome_advantage(rewards,masks,np.repeat(np.arange(8),4),norm_adv_by_std_in_grpo=True)
    def lp(i):
        p,r=samples[i]
        ids=torch.tensor([p+r],device='cuda')
        logits=model(input_ids=ids,attention_mask=torch.ones_like(ids)).logits[0,len(p)-1:-1].float()/.6
        values=torch.log_softmax(logits,dim=-1).gather(1,torch.tensor(r,device='cuda')[:,None]).squeeze(1)
        return torch.nn.functional.pad(values,(0,max_r-len(r)))
    conditions=[]
    for mini_prompts in (8,4):
        with torch.no_grad():
            for n,p in model.named_parameters():
                if p.requires_grad:p.copy_(initial[n])
        seed_all(ev.config['seed']); model.train()
        opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
        with torch.no_grad(): old=torch.stack([lp(i) for i in range(32)])
        rows=[]
        for start in range(0,32,mini_prompts*4):
            count=mini_prompts*4; opt.zero_grad(set_to_none=True)
            ratios=[]; metrics=[]; losses=[]; t=time.monotonic()
            for i in range(start,start+count):
                current=lp(i)[None]
                ratio=torch.exp(((current-old[i:i+1])*masks[i:i+1]).sum()/masks[i].sum()).item()
                loss,m=compute_policy_loss_gspo(old[i:i+1],current,adv[i:i+1],masks[i:i+1],config=actor_config())
                (loss/count).backward(); ratios.append(ratio); metrics.append(m); losses.append(float(loss.detach()))
            grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
            assert torch.isfinite(grad)
            opt.step()
            row=ev.metric(event='gspo_minibatch',mini_prompts=mini_prompts,trajectories=count,update_index=start//count+1,ratio_mean=float(np.mean(ratios)),ratio_min=min(ratios),ratio_max=max(ratios),ratios=ratios,clip_fraction=float(np.mean([m['actor/pg_clipfrac'] for m in metrics])),loss=float(np.mean(losses)),grad_norm=float(grad),seconds=time.monotonic()-t)
            rows.append(row)
            if start==0: assert max(abs(r-1) for r in ratios)<1e-6
            ev.case('gspo_ratio_one' if start==0 else 'gspo_after_update',f"mini_prompts={mini_prompts}, update={start//count+1}, clip={row['clip_fraction']}",category='SYSTEM_CASE',metrics=row)
        with torch.no_grad():
            after=torch.stack([lp(i) for i in range(32)])
            drift=(((after-old)*masks).sum(1)/masks.sum(1)).abs().mean().item()
        conditions.append(dict(mini_prompts=mini_prompts,epochs=1,updates=32//(mini_prompts*4),metrics=rows,mean_absolute_sequence_logprob_drift=drift))
    write_json(ev.path/'trajectories.json',dict(source='synthetic fixed trajectories; teacher-forced diagnostic, not on-policy RL',samples=[dict(prompt_ids=p,response_ids=r) for p,r in samples]))
    return dict(gates={'minibatch_diagnostic':True},conditions=conditions,orchestration='bounded local microbatch accumulation; native verl GSPO/GRPO; full Ray trainer not run')
