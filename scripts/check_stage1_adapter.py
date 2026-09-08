"""Fresh-process identity/reload plus heldout generation checks for SFT runs."""
import argparse
import json
import math
from pathlib import Path
import re
import time
import torch
from medical_posttrain.evidence import sha256,write_json
from medical_posttrain.training.stage1 import check_checkpoint,load,read_rows,json_digest
from medical_posttrain.training.lora import parameters,digest

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--count',type=int,default=4);args=p.parse_args()
    root=Path(args.run);config=json.loads((root/'config.json').read_text());summary=json.loads((root/'summary.json').read_text())
    output=root/'reload_generation';output.mkdir(exist_ok=False)
    write_json(output/'protocol.json',dict(process='fresh independent Python process',generation=dict(do_sample=False,max_new_tokens=1024,enable_thinking=True,eos_token_id=151645),purpose='Reload and structural sanity; greedy diagnostic, distinct from sampled paired final evaluation',script_sha256=sha256(__file__)))
    start=time.monotonic();integrity=check_checkpoint(summary['final_checkpoint'])
    model,tokenizer=load(config,summary['final_adapter']);model.eval();model.gradient_checkpointing_disable();model.config.use_cache=True
    assert digest(parameters(model))==integrity['trainable_digest']==summary['final_trainable_digest']
    assert sha256(Path(summary['final_adapter'])/'adapter_model.safetensors')==summary['adapter_sha256']
    val=read_rows(config['validation_path']);selected=val[:args.count//2]+val[500:500+args.count-args.count//2]
    records=[];maximum_delta=0.
    with torch.no_grad():
        for row in selected:
            messages=row['messages'][:-1]
            prompt=tokenizer.apply_chat_template(messages,tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
            ids=torch.tensor([prompt],device='cuda');attention=torch.ones_like(ids)
            logits=model(input_ids=ids,attention_mask=attention).logits[0,-1].float().cpu()
            with model.disable_adapter():base=model(input_ids=ids,attention_mask=attention).logits[0,-1].float().cpu()
            delta=float((logits-base).abs().max());maximum_delta=max(maximum_delta,delta)
            assert torch.isfinite(logits).all() and delta>1e-5
            generated=model.generate(input_ids=ids,attention_mask=attention,max_new_tokens=1024,do_sample=False,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id,use_cache=True)[0,len(prompt):].tolist()
            text=tokenizer.decode(generated,skip_special_tokens=True)
            match=re.fullmatch(r'\s*<think>\s*([\s\S]*?)</think>\s*<answer>([\s\S]*?)</answer>\s*',text)
            records.append(dict(sample_id=row['sample_id'],source=row['source'],prompt_messages=messages,prompt_ids=prompt,output_ids=generated,output=text,output_tokens=len(generated),format_valid=bool(match and match[2].strip()),eos_terminated=generated[-1]==tokenizer.eos_token_id,adapter_base_logit_max_delta=delta,reference_answer=row['messages'][-1]['content']))
    with (output/'generations.jsonl').open('x') as f:
        for row in records:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    receipt=dict(status='PASS',run_id=summary['run_id'],run_class=config['run_class'],check='fresh_process_real_adapter_identity_and_generation',adapter_sha256=summary['adapter_sha256'],trainable_digest=integrity['trainable_digest'],identity_logit_max_delta=maximum_delta,generation_count=len(records),format_valid_count=sum(r['format_valid'] for r in records),eos_terminated_count=sum(r['eos_terminated'] for r in records),generations_path=str(output/'generations.jsonl'),generations_sha256=sha256(output/'generations.jsonl'),elapsed_seconds=time.monotonic()-start)
    if config['run_class']=='PILOT':
        resume=json.loads((root/'resume_receipt.json').read_text());assert resume['same_next_sample_ids'] and resume['parameter_max_abs_error']<=1e-5
        receipt['real_data_resume_receipt_sha256']=sha256(root/'resume_receipt.json')
    write_json(output/'receipt.json',receipt);print(json.dumps(receipt))

if __name__=='__main__':main()
