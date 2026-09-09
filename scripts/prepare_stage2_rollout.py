"""Freeze reviewable Stage 2 configs before any smoke/formal inference."""
import argparse
from pathlib import Path
from medical_posttrain.evidence import write_json,sha256
from medical_posttrain.evidence.stage2 import read,record,selected_path,INDEX
from medical_posttrain.reward.semantic import CHUNK_POLICY
from medical_posttrain.reward.hybrid import WEIGHTS
from medical_posttrain.data.stage1 import text_hash

p=argparse.ArgumentParser();p.add_argument('--mode',choices=['smoke','formal','length'],required=True);p.add_argument('--cap',type=int,default=1024);a=p.parse_args()
if not (INDEX/'reward_manifest.json').exists():
    semantic=selected_path('semantic');s=read(semantic/'summary.json');assert s['status']=='PASS' and s['pair_count']>=100
    encoder=next(r for r in s['results'] if r['model_id']=='BAAI/bge-m3')
    manifest=dict(formula='0.8 * acc + 0.15 * acc * sem + 0.05 * format',weights=WEIGHTS,correctness_gating=True,
                  sampling_metric='acc',parser=record('src/medical_posttrain/reward/parser.py'),reward=record('src/medical_posttrain/reward/hybrid.py'),
                  semantic=record('src/medical_posttrain/reward/semantic.py'),encoder_id=encoder['model_id'],encoder_revision=encoder['revision'],
                  model_files=encoder['model_files'],encoder_mapping='clip(cosine,0,1)',chunk_policy=CHUNK_POLICY,
                  format_rules='single closed answer-only or one think + one answer, no extra text/tags, legal unique options; canonical case/separators/order',
                  ground_truth_canonicalization='NFKC uppercase, allowed separators removed, reject repeats/illegal labels, sort unique exact set; no partial credit',
                  semantic_decision='docs/implementation/STAGE2_DECISIONS.md#d-022',diagnostic=record(semantic/'summary.json'))
    write_json(INDEX/'reward_manifest.json',manifest)
init=read('experiments/stage1/initialization_manifest.json')
parent=read(Path('experiments/stage1')/init['run_id']/'manifest.json')
parentconfig=read(Path(parent['artifact_root'])/'config.json')
data=selected_path('data')
files=['src/medical_posttrain/reward/parser.py','src/medical_posttrain/reward/hybrid.py','src/medical_posttrain/reward/semantic.py',
       'src/medical_posttrain/data/exam.py','src/medical_posttrain/rollout/profile.py','src/medical_posttrain/rollout/statistics.py']
config=dict(stage=2,mode=a.mode,model=parentconfig['model'],policy_version=text_hash(init['base_revision']+':'+init['adapter_sha256']),
            initialization=record('experiments/stage1/initialization_manifest.json'),pool=record(data/'candidate_pool.jsonl'),
            selection=record(data/('profiling_prompt_ids.json' if a.mode=='formal' else 'smoke_prompt_ids.json')),
            planned_prompts=1000 if a.mode=='formal' else 50,planned_responses=4000 if a.mode=='formal' else 200,
            seed=42,request_batch_prompts=4,sampling=dict(n=4,temperature=.6,top_p=1.,top_k=-1,min_p=0.,max_tokens=a.cap,
                     presence_penalty=0.,frequency_penalty=0.,repetition_penalty=1.),
            engine=dict(dtype='bfloat16',tensor_parallel_size=1,max_model_len=4096,max_num_seqs=16,gpu_memory_utilization=.65,
                        enable_lora=True,max_lora_rank=32,enable_sleep_mode=True,enforce_eager=True,seed=42,
                        enable_prefix_caching=False,generation_config='vllm'),
            reward_manifest=record(INDEX/'reward_manifest.json'),execution_hashes={f:sha256(f) for f in files},optimizer_updates=0)
if a.mode=='formal':
    smoke=selected_path('smoke');assert read(smoke/'summary.json')['status']=='PASS'
    config['smoke_receipt']=record(smoke/'summary.json')
    config['freeze_decision']=record('docs/implementation/STAGE2_DECISIONS.md')
path=Path('configs/stages')/f's2_{a.mode}_{a.cap}.json'
assert not path.exists(), 'Use a new config filename/version for any changed experiment'
write_json(path,config);print(path)
