"""Bounded child process owned by the parent vLLM run and its evidence manifest."""
import json
from pathlib import Path
import sys
from medical_posttrain.evidence import Evidence
from medical_posttrain.training.lora import load,batch,step

def main():
    import torch
    path=Path(sys.argv[1]); config=json.loads((path/'resolved_config.json').read_text())
    child=path/'actor_child'; child.mkdir()
    (child/'command.json').write_text(json.dumps(sys.argv))
    ev=Evidence.__new__(Evidence); ev.path=child; ev.config=config; ev.run_id=path.parent.name; ev.bulk=Path(config['artifact_root'])/ev.run_id/'actor_child'; ev.bulk.mkdir()
    model,tok=load(ev,config['adapter'])
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda _:1.)
    step(model,opt,sched,batch(tok,128),ev,'actor_while_rollout_sleeps')
    model.save_pretrained(ev.bulk/'updated_adapter',safe_serialization=True)

if __name__=='__main__': main()
