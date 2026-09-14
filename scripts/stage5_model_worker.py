#!/usr/bin/env python3
"""Future GPU worker. Never imported or executed by CPU preflight."""
import argparse,sys,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.evaluation.core import read,rows,ref,check_ref,freeze
from medical_posttrain.evaluation.gates import selection_gate,final_gate,gpu_guard,worker_spec_gate
from medical_posttrain.evaluation.executor import execute_items

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);a=ap.parse_args();spec=read(a.spec)
    worker_spec_gate(spec)
    from importlib.metadata import version
    protocol=read(ROOT/'experiments/stage5/checkpoint_selection_protocol_v1.json')
    if any(version(k)!=v for k,v in protocol['runtime']['versions'].items()):raise PermissionError('Pinned runtime version mismatch')
    gpu_guard();check_ref(spec['adapter']);check_ref(spec['items']);check_ref(spec['requests'])
    # All model imports and engine creation are downstream of every execution gate.
    for k,v in spec['runtime_environment'].items():os.environ[k]=v
    from medical_posttrain.rl.rollout import Rollout
    from vllm import SamplingParams
    from tokenizers.decoders import DecodeStream
    directory=Path(spec['directory']);engine_dir=directory/('engine_'+str(time.time_ns()));engine_dir.mkdir(parents=True)
    model=Rollout(engine_dir,spec['config'],dict(adapter_path=str(Path(spec['adapter']['path']).parent),adapter_sha256=spec['adapter']['sha256']))
    def backend(requests):
        prompts=[];token_lists=[]
        for request in requests:
            text=model.tok.apply_chat_template(request['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=True)
            tokens=model.tok.encode(text,add_special_tokens=False)
            if len(tokens)+spec['decoding']['max_tokens']>spec['config']['engine']['max_model_len']:raise ValueError('Context exceeds frozen limit; no silent truncation')
            prompts.append(text);token_lists.append(tokens)
        t=time.monotonic();outputs=model.llm.generate(prompts,SamplingParams(**spec['decoding']),lora_request=model.request,use_tqdm=False);elapsed=time.monotonic()-t
        if len(outputs)!=len(requests):raise ValueError('Incomplete batch')
        saved=[]
        for result,tokens in zip(outputs,token_lists):
            if len(result.outputs)!=1 or result.prompt_token_ids!=tokens:raise ValueError('Output/prompt mismatch')
            o=result.outputs[0];decoder=DecodeStream(ids=tokens,skip_special_tokens=True)
            if ''.join(decoder.step(model.tok._tokenizer,t) or '' for t in o.token_ids)!=o.text:raise ValueError('Raw decode mismatch')
            saved.append(dict(raw_output=o.text,finish_reason=o.finish_reason,prompt_tokens=len(tokens),output_tokens=len(o.token_ids),prompt_token_ids=tokens,output_token_ids=list(o.token_ids),seconds=elapsed/len(outputs),time_accounting='amortized batch wall time',adapter_sha256=spec['adapter']['sha256']))
        return saved
    try:execute_items(rows(spec['items']['path']),rows(spec['requests']['path']),backend,directory,spec['checkpoint_id'],spec['run_id'],spec['resume'],spec['technical_retry'],spec['exam'])
    finally:model.close()
if __name__=='__main__':main()
