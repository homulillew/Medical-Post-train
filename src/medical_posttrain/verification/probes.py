import importlib
import json
from pathlib import Path
from medical_posttrain.evidence import sha256, write_json

def dispatch(ev):
    purpose=ev.config['purpose']
    if purpose=='environment':
        import torch
        from medical_posttrain.runtime import environment,command
        imports={name:str(importlib.import_module(name).__file__) for name in ('torch','transformers','peft','vllm','verl','ray','tensordict')}
        x=torch.randn(64,64,device='cuda',dtype=torch.bfloat16); assert torch.isfinite(x@x).all()
        check=command(['uv','pip','check','--python',__import__('sys').executable]); assert check['returncode']==0,check
        r=environment()|dict(imports=imports,torch_cuda=torch.version.cuda,nccl=torch.cuda.nccl.version(),flash_attn_installed=importlib.util.find_spec('flash_attn') is not None,pip_check=check)
        write_json(ev.path/'runtime.json',r)
        ev.metric(event='environment',torch=torch.__version__,cuda=torch.version.cuda,nccl=torch.cuda.nccl.version())
        return dict(gates={'runtime':True},runtime=r)
    if purpose=='snapshot':
        from medical_posttrain.config import QWEN_REVISION
        root=Path(ev.config['model'])
        source=json.loads(Path('experiments/stage0/bootstrap/qwen-source.json').read_text())
        files=[]
        for row in source['siblings']:
            p=root/row['rfilename']
            h=sha256(p)
            if row.get('lfs'):
                assert h==row['lfs']['sha256'],p
            assert p.stat().st_size==row['size']
            files.append(dict(path=str(p),size=p.stat().st_size,sha256=h))
            p.chmod(0o444)
        result=dict(source='Qwen/Qwen3-8B',revision=QWEN_REVISION,files=files,gates={'snapshot':True})
        write_json(ev.path/'snapshot_manifest.json',result)
        ev.metric(event='snapshot',file_count=len(files),bytes=sum(f['size'] for f in files))
        return result
    if purpose in {'lora','reload'}:
        from medical_posttrain.training import lora
        return getattr(lora,'run' if purpose=='lora' else 'reload')(ev)
    module={'template':'templates','numeric':'verl_probe','verl':'verl_probe','minibatch':'minibatch','vllm':'vllm_probe','length':'vllm_probe','semantic':'semantic'}[purpose]
    return importlib.import_module('medical_posttrain.verification.'+module).run(ev)
