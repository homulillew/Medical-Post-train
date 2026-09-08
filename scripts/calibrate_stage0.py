#!/usr/bin/env python3
"""Transparent conditional estimates from selected diagnostics; not formal throughput."""
import hashlib
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/stage0/compute_calibration.json')
    args=parser.parse_args()
    selected=json.loads((ROOT/'experiments/stage0/selected_runs.json').read_text())['run_ids']
    metrics={}; sources={}
    for run in selected:
        p=ROOT/'experiments/stage0'/run/'attempt_001'
        purpose=json.loads((p.parent/'manifest.json').read_text())['purpose']
        metrics[purpose]=[json.loads(l) for l in (p/'metrics.jsonl').read_text().splitlines()]
        sources[run]=hashlib.sha256((p/'metrics.jsonl').read_bytes()).hexdigest()
    lora_tps=next(m['aggregate_output_tokens_per_second'] for m in metrics['vllm'] if m['event']=='offline_batch_timing' and m['condition']=='adapter')
    sft_tps=next(m['tokens_per_second'] for m in metrics['lora'] if m['event']=='memory_sweep' and m['sequence_length']==1024)
    out=next(m for m in metrics['length'] if m['limit']==1024)
    switch=next(m for m in metrics['vllm'] if m['event']=='wake_identity')
    measured={'sft_1024_update_phase_tps':sft_tps,'lora_batch16_output_tps_32_token_window':lora_tps,'base_model_mean_response_at_1024_limit':out['average_output_tokens'],'cold_sleep_wake_seconds':switch['sleep_seconds']+switch['wake_seconds'],'actor_subprocess_seconds_including_load_and_one_backward':switch['actor_process_seconds']}
    scenarios={}
    for name,factor,acceptance,actor_tps,old_tps,overhead,switch_s in [('optimistic',1.5,.65,1000,1500,1.1,10),('working',1.,.35,700,1000,1.2,25),('adverse',.5,.10,250,400,1.35,60)]:
        decode=lora_tps*factor
        mean_sft=1024;prompt=384;response=out['average_output_tokens']
        effective_sft=sft_tps*{'optimistic':.8,'working':.45,'adverse':.2}[name]
        def hours(seconds):return seconds*overhead/3600
        generating=20000*response/decode
        learning=20000*(prompt+response)*(1/actor_tps+1/old_tps)
        validation=8704*response/decode
        h={'stage1':hours(20000*mean_sft/effective_sft),'stage2':hours(4000*response/decode),'stage3':hours(256*4*response/acceptance/decode),'vanilla':hours(generating+learning+625*switch_s+validation),'dynamic':hours(generating/acceptance+learning+625*switch_s+validation),'stage5':hours((3*(6811+2000)+150)*response/decode),'stage6':{'optimistic':.5,'working':2,'adverse':8}[name]}
        scenarios[name]={'classification':'Estimated','assumptions':{'shared_response_limit_candidate':1024,'mean_sft_tokens':mean_sft,'mean_prompt_tokens':prompt,'mean_response_tokens_transferred_from_base_diagnostic':response,'lora_decode_tps':decode,'effective_sft_tps':effective_sft,'actor_tps':actor_tps,'old_logprob_tps':old_tps,'mixed_acceptance':acceptance,'overhead_factor':overhead,'switch_seconds':switch_s,'updates_per_variant':625},'gpu_hours':h,'total_gpu_hours':sum(h.values())}
    result={'analysis_id':'s0_compute_calibration_20260908_001','stage':0,'classification':'DIAGNOSTIC_ANALYSIS','measured':measured,'estimated_scenarios':scenarios,'unknown':['Post-SFT response distribution','Natural mixed acceptance','Long-run LoRA decode throughput and final validation accuracy','Full Ray actor/rollout scheduling and checkpoint overhead','Dataset mean length/packing efficiency','Additional optimizer-update cost if mini_prompts=4 is adopted'],'sources_metrics_sha256':sources,'caveats':['512 and 1024 length conditions used the same earlier non-batch-invariant base runtime; final LoRA timing uses batch invariance. This cross-condition transfer is an estimate, not a matched benchmark.','32-token batch timing includes shape/activation overhead and does not establish steady decode throughput.','20k/5000/G4 budgets retained. 625 updates assumes the existing 8-prompt setting; D-014 is not adopted.','Worst-case cost remains unbounded as acceptance tends to zero.']}
    p=args.output
    with p.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:round(v['total_gpu_hours'],2) for k,v in scenarios.items()}))

if __name__=='__main__':main()
