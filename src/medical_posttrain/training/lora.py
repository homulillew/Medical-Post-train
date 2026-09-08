import gc
import hashlib
import json
from pathlib import Path
import random
import threading
import time
from medical_posttrain.evidence import write_json

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

def seed_all(seed):
    import torch
    import numpy as np
    torch.set_num_threads(8)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class MemoryMonitor:
    def __init__(self):
        import pynvml
        pynvml.nvmlInit()
        self.nv = pynvml
        self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.peak = 0
        self.stop = threading.Event()
    def __enter__(self):
        def sample():
            while not self.stop.is_set():
                self.peak = max(self.peak, self.nv.nvmlDeviceGetMemoryInfo(self.handle).used)
                self.stop.wait(.02)
        self.thread=threading.Thread(target=sample,daemon=True); self.thread.start()
        return self
    def __exit__(self,*args):
        self.stop.set(); self.thread.join()
    def result(self):
        import torch
        total = self.nv.nvmlDeviceGetMemoryInfo(self.handle).total
        return dict(nvml_peak_bytes=self.peak, allocated_peak_bytes=torch.cuda.max_memory_allocated(),reserved_peak_bytes=torch.cuda.max_memory_reserved(), remaining_at_peak_bytes=total-self.peak,total_bytes=total)

def load(ev, adapter=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model, PeftModel
    seed_all(ev.config["seed"])
    t=time.monotonic()
    tokenizer=AutoTokenizer.from_pretrained(ev.config["model"], local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(ev.config["model"], dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda", local_files_only=True)
    assert model.config.model_type=="qwen3" and model.config.hidden_size==4096
    assert next(model.parameters()).dtype==torch.bfloat16
    if adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
    else:
        model=get_peft_model(model,LoraConfig(r=32,lora_alpha=64,lora_dropout=0,target_modules=TARGETS,task_type="CAUSAL_LM"))
    model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    ev.metric(event="load",seconds=time.monotonic()-t,base_dtype="bfloat16",attention="sdpa",trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),total_parameters=sum(p.numel() for p in model.parameters()))
    return model,tokenizer

def parameters(model):
    return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def digest(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode()); h.update(v.contiguous().view(__import__('torch').uint8).numpy().tobytes())
    return h.hexdigest()

def rng_state():
    import torch,numpy as np
    return dict(python=random.getstate(),numpy=np.random.get_state(),cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

def restore_rng(state):
    import torch,numpy as np
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["cpu"]); torch.cuda.set_rng_state_all(state["cuda"])

def batch(tokenizer,length):
    import torch
    from medical_posttrain.data.format import sft_tokens
    ids,labels=sft_tokens(tokenizer,"这是合成格式诊断，不用于医疗建议。请选择字母 C。"+("诊断输入。"*length),"C","仅用于验证输出格式和梯度。"*(length//8+1))
    # Bounded sequence capacity probe supervises the final half, independent of corpus.
    ids=(ids*(length//len(ids)+1))[:length]
    x=torch.tensor([ids],device="cuda")
    y=x.clone(); y[:,:length//2]=-100
    return dict(input_ids=x,labels=y,attention_mask=torch.ones_like(x))

def step(model,optimizer,scheduler,data,ev,tag):
    import torch
    model.train(); optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    before=parameters(model)
    with MemoryMonitor() as mm:
        torch.cuda.synchronize(); t=time.monotonic()
        loss=model(**data).loss
        torch.cuda.synchronize(); forward=time.monotonic()-t; t=time.monotonic()
        loss.backward()
        torch.cuda.synchronize(); backward=time.monotonic()-t; t=time.monotonic()
        grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
        assert torch.isfinite(loss) and torch.isfinite(grad)
        optimizer.step(); scheduler.step(); torch.cuda.synchronize()
        optim=time.monotonic()-t
    after=parameters(model)
    delta=sum((after[k]-v).double().square().sum().item() for k,v in before.items())**.5
    assert delta>0
    row=ev.metric(event=tag,sequence_length=data["input_ids"].shape[1],microbatch=1,forward_seconds=forward,backward_seconds=backward,optimizer_seconds=optim,tokens_per_second=data["input_ids"].numel()/(forward+backward+optim),loss=loss.item(),grad_norm=float(grad),parameter_delta_l2=delta,**mm.result())
    return row

def identity_logits(model,tokenizer):
    import torch
    model.eval()
    ids=tokenizer("<|im_start|>user\n请选择 C。<|im_end|>\n<|im_start|>assistant\n",return_tensors="pt").to("cuda")
    with torch.no_grad():
        adapted=model(**ids).logits[0,-1].float().cpu()
        with model.disable_adapter():
            base=model(**ids).logits[0,-1].float().cpu()
    return dict(adapter=adapted,base=base,input_ids=ids.input_ids.cpu())

def run(ev):
    import torch
    model,tokenizer=load(ev)
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1/(1+s*.1))
    rows=[]
    for length in (512,1024,2048):
        rows.append(step(model,opt,sched,batch(tokenizer,length),ev,"memory_sweep"))
    adapter=ev.bulk/"adapter_step3"
    model.save_pretrained(adapter,safe_serialization=True)
    identity=identity_logits(model,tokenizer)
    assert (identity['adapter']-identity['base']).abs().max().item()>0
    torch.save(identity,ev.bulk/"identity.pt")
    state=dict(optimizer=opt.state_dict(),scheduler=sched.state_dict(),rng=rng_state(),global_step=3,trainable_digest=digest(parameters(model)),adapter=str(adapter),sample_cursor=None)
    checkpoint=ev.bulk/"checkpoint_step3.pt"; torch.save(state,checkpoint)
    write_json(ev.path/"checkpoint_refs.json",[dict(path=str(checkpoint),global_step=3,adapter=str(adapter))])
    # A controlled continuation supplies the reference for a separate process.
    reference=step(model,opt,sched,batch(tokenizer,128),ev,"reference_continuation")
    torch.save(parameters(model),ev.bulk/"reference_step4.pt")
    write_json(ev.bulk/"reference.json",dict(step=4,scheduler=sched.state_dict(),loss=reference["loss"]))
    for p in ev.bulk.rglob("*"):
        if p.is_file(): ev.artifact(p,"checkpoint_or_identity")
    return dict(gates={"bf16_load":True,"lora_backward":True,"lora_save":True},sweep=rows,checkpoint=str(checkpoint),adapter=str(adapter),identity_max_abs_delta=(identity['adapter']-identity['base']).abs().max().item())

def reload(ev):
    import torch
    checkpoint=Path(ev.config["resume"])
    state=torch.load(checkpoint,map_location="cpu",weights_only=False) # trusted, locally produced checkpoint only
    model,tokenizer=load(ev,state["adapter"])
    assert digest(parameters(model))==state["trainable_digest"]
    expected=torch.load(checkpoint.parent/"identity.pt",weights_only=True)
    actual=identity_logits(model,tokenizer)
    max_error=(actual['adapter']-expected['adapter']).abs().max().item()
    assert torch.allclose(actual['adapter'],expected['adapter'],atol=.001,rtol=.001)
    assert (actual['adapter']-actual['base']).abs().max().item()>0
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:1/(1+s*.1))
    opt.load_state_dict(state['optimizer']); sched.load_state_dict(state['scheduler']); restore_rng(state['rng'])
    assert torch.equal(torch.get_rng_state(),state['rng']['cpu'])
    assert all(torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),state['rng']['cuda']))
    result=step(model,opt,sched,batch(tokenizer,128),ev,"resumed_continuation")
    reference=torch.load(checkpoint.parent/"reference_step4.pt",weights_only=True)
    params=parameters(model)
    errors={n:(params[n]-v).abs().max().item() for n,v in reference.items()}
    assert all(torch.allclose(params[n],v,atol=1e-6,rtol=1e-4) for n,v in reference.items())
    ref=json.loads((checkpoint.parent/"reference.json").read_text())
    assert sched.state_dict()==ref['scheduler']
    assert abs(result['loss']-ref['loss'])<1e-5
    ev.metric(event="resume_identity",reload_logits_max_error=max_error,continued_parameter_max_error=max(errors.values()),global_step=state['global_step']+1,optimizer_state_entries=len(opt.state),scheduler_equal=True,rng_restored=True)
    return dict(gates={"adapter_reload":True,"checkpoint_resume":True},parent_checkpoint=str(checkpoint),reload_logits_max_error=max_error,continued_parameter_max_error=max(errors.values()),global_step=4)
