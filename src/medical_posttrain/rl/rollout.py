"""Native vLLM sampling and frozen reward, with durable request boundaries."""
import os
from pathlib import Path
import time

from .common import read, immutable, event, encounter, sha256


class Rollout:
    def __init__(self, out, cfg, initial):
        os.environ.update(VLLM_WORKER_MULTIPROC_METHOD='spawn', VLLM_USE_FLASHINFER_SAMPLER='0',
                          VLLM_USE_V2_MODEL_RUNNER='0', VLLM_BATCH_INVARIANT='1')
        from transformers import AutoTokenizer
        from vllm import LLM
        from vllm.lora.ops.triton_ops.utils import get_lora_op_configs
        self.out, self.cfg = Path(out), cfg
        self.tok = AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
        shrink = get_lora_op_configs('shrink',1,23,4096,32,3)
        assert shrink['split_k'] == 1
        args = dict(model=cfg['model'],tokenizer=cfg['model'],**cfg['engine'])
        immutable(self.out/'engine_args.json',args)
        immutable(self.out/'runtime_env.json',{k:os.environ[k] for k in (
            'VLLM_WORKER_MULTIPROC_METHOD','VLLM_USE_FLASHINFER_SAMPLER','VLLM_USE_V2_MODEL_RUNNER','VLLM_BATCH_INVARIANT')})
        self.llm = LLM(**args)
        self.set_policy(initial['adapter_path'], initial['adapter_sha256'], 1)

    def set_policy(self, path, digest, adapter_id):
        from vllm.lora.request import LoRARequest
        assert sha256(Path(path)/'adapter_model.safetensors') == digest
        self.request = LoRARequest('policy_'+digest[:16],adapter_id,str(path))
        self.policy, self.adapter = digest, str(path)

    def prompt(self, source):
        from medical_posttrain.data.exam import messages
        text = self.tok.apply_chat_template(messages(source),tokenize=False,
                                            add_generation_prompt=True,enable_thinking=True)
        ids = self.tok.encode(text,add_special_tokens=False)
        assert len(ids)+1024 <= self.cfg['engine']['max_model_len']
        return text,ids

    def generate(self, directory, stream, pool, cursor, count, run_id, config_sha):
        from vllm import SamplingParams
        directory = Path(directory)
        directory.mkdir(parents=True,exist_ok=True)
        encounters = [encounter(stream,i,run_id,self.policy) for i in range(cursor,cursor+count)]
        prompts = [self.prompt(pool[e['prompt_id']]) for e in encounters]
        reservation = dict(encounters=encounters,prompt_ids=[p[1] for p in prompts],
                           policy_version=self.policy,adapter=self.adapter)
        if (directory/'reservation.json').exists():
            assert read(directory/'reservation.json') == reservation
            assert (directory/'raw.json').exists(), 'Unreturned generation: output cost UNKNOWN; no silent retry'
            return read(directory/'raw.json')['groups']
        immutable(directory/'reservation.json',reservation)
        params = [SamplingParams(**self.cfg['sampling'], seed=e['request_seed'],logprobs=0) for e in encounters]
        event(self.out,'generation_started',path=str(directory),cursor=cursor,count=count,policy=self.policy)
        started = time.monotonic()
        outputs = self.llm.generate([p[0] for p in prompts],params,lora_request=self.request,use_tqdm=False)
        seconds = time.monotonic()-started
        assert len(outputs) == count
        groups = []
        for e,p,result in zip(encounters,prompts,outputs):
            assert result.prompt_token_ids == p[1]
            source = pool[e['prompt_id']]
            rows = []
            for o in sorted(result.outputs,key=lambda o:o.index):
                rows.append(dict(run_id=run_id,policy_version=self.policy,adapter_sha256=self.policy,
                    config_sha256=config_sha,reward_version=self.cfg['reward_manifest']['sha256'],
                    group_id=e['group_id'],prompt_id=e['prompt_id'],trajectory_id=e['group_id']+':'+str(o.index),
                    encounter_index=e['encounter_index'],member_index=o.index,request_seed=e['request_seed'],
                    question=source['question'],options=source['options'],ground_truth=source['answer_set'],
                    raw_output=o.text,token_ids=list(o.token_ids),prompt_token_ids=p[1],
                    prompt_tokens=len(p[1]),output_tokens=len(o.token_ids),finish_reason=o.finish_reason,
                    stop_reason=o.stop_reason,rollout_raw_logprobs=[float(lp[t].logprob) for t,lp in zip(o.token_ids,o.logprobs)]))
            groups.append(dict(**e,prompt_tokens=len(p[1]),responses=rows))
        immutable(directory/'raw.json',dict(groups=groups,generation_seconds=seconds,timestamp=__import__('medical_posttrain.evidence',fromlist=['now']).now()))
        event(self.out,'generation_completed',path=str(directory),seconds=seconds,
              output_tokens=sum(r['output_tokens'] for g in groups for r in g['responses']))
        return groups

    def close(self):
        self.llm.llm_engine.engine_core.shutdown(timeout=30)


def score(directory, groups, cfg, pool, tok, encoder=None):
    from medical_posttrain.sampling.stage3 import score_groups
    from medical_posttrain.reward.semantic import Encoder
    directory = Path(directory)
    if (directory/'scored.json').exists():
        return read(directory/'scored.json')['groups']
    if encoder is None:
        rm = read(cfg['reward_manifest']['path'])
        encoder = Encoder(rm['encoder_id'],rm['encoder_revision'],device='cuda')
    start = time.monotonic()
    scored = score_groups(groups,encoder,tok,pool,directory)
    immutable(directory/'reward_runtime.json',dict(seconds=time.monotonic()-start))
    return scored
