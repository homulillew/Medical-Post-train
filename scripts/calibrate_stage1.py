"""Re-estimate later-stage costs from completed SFT and paired heldout evidence."""
import argparse
import json
from pathlib import Path
from medical_posttrain.evidence import sha256,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='experiments/stage1/compute_calibration.json');args=parser.parse_args()
    selected=json.loads(Path('experiments/stage1/selected_runs.json').read_text())
    fdir=Path(json.loads((Path('experiments/stage1')/selected['formal']/'manifest.json').read_text())['artifact_root'])
    formal=json.loads((fdir/'summary.json').read_text())
    edir=Path('experiments/stage1')/selected['evaluation'];evaluation=json.loads((edir/'summary.json').read_text())
    assert formal['run_class']=='FORMAL' and formal['unique_examples']==20000 and evaluation['status']=='PASS'
    good=[cap for cap in evaluation['limits'] if evaluation['results'][f'sft_{cap}']['all']['answer_closed_fraction']>=.95 and evaluation['results'][f'sft_{cap}']['all']['truncated_fraction']<=.05]
    candidate=min(good) if good else max(evaluation['limits'])
    generation=evaluation['results'][f'sft_{candidate}']['all'];timing=evaluation['timings'][f'sft_{candidate}']
    response=generation['total_tokens']['mean'];decode_measured=timing['output_tokens_per_second']
    measured=dict(formal_examples=formal['examples'],formal_total_tokens=formal['processed_tokens'],formal_supervised_tokens=formal['supervised_tokens'],formal_update_seconds=formal['update_seconds'],formal_worker_seconds=formal['attempt_wall_seconds'],formal_update_tokens_per_second=formal['effective_tokens_per_update_second'],formal_peak_nvml_bytes=formal['nvml_peak_bytes'],post_sft_heldout_mean_response_tokens=response,post_sft_heldout_decode_tokens_per_second=decode_measured,post_sft_heldout_response_limit=candidate,post_sft_answer_closure=generation['answer_closed_fraction'],post_sft_truncation=generation['truncated_fraction'])
    scenarios={}
    for name,factor,acceptance,actor,old,overhead,switch in [('optimistic',1.5,.65,1000,1500,1.1,10),('working',1.,.35,700,1000,1.2,25),('adverse',.5,.10,250,400,1.35,60)]:
        decode=decode_measured*factor;prompt=384
        def hours(seconds):return seconds*overhead/3600
        generating=20000*response/decode;learning=20000*(prompt+response)*(1/actor+1/old);validation=8704*response/decode
        values=dict(stage1_measured_worker=formal['attempt_wall_seconds']/3600,stage2=hours(4000*response/decode),stage3=hours(256*4*response/acceptance/decode),vanilla=hours(generating+learning+625*switch+validation),dynamic=hours(generating/acceptance+learning+625*switch+validation),stage5=hours((3*(6811+2000)+150)*response/decode),stage6={'optimistic':.5,'working':2,'adverse':8}[name])
        scenarios[name]=dict(classification='Estimated except completed Stage 1 worker time',assumptions=dict(mean_response_transferred_from_open_ended_validation=response,mean_prompt_tokens=prompt,decode_tokens_per_second=decode,mixed_acceptance=acceptance,actor_tokens_per_second=actor,old_logprob_tokens_per_second=old,switch_seconds=switch,overhead=overhead,updates_per_rl_variant=625),gpu_hours=values,total_gpu_hours=sum(values.values()))
    result=dict(analysis_id='s1_compute_calibration_'+selected['formal'],stage=1,classification='DIAGNOSTIC_ANALYSIS',measured=measured,estimated_scenarios=scenarios,rl_response_candidate=candidate,candidate_meets_heldout_closure_trigger=bool(good),unknown=['Natural CMExam mixed-group acceptance','Post-SFT CMExam response distribution; current inputs are open-ended SFT validation','Formal GSPO actor/old-logprob throughput and Ray switching','Sustained serving throughput and request latency','Any extra update cost if later adopting the mini=4 GSPO proposal'],sources={str(fdir/'summary.json'):sha256(fdir/'summary.json'),str(edir/'summary.json'):sha256(edir/'summary.json')},caveats=['No Stage 2–6 experiment was executed by this analysis.','SFT wall time is measured to worker completion; separate smoke/pilot/diagnostics/reload/generation costs are reported in the stage report.','Open-ended validation speed/length transfer to examination rollouts is an estimate.','Mandatory 20k/5000/G4 budgets and 625-update planning assumption are preserved.','The strict worst case remains unbounded as mixed acceptance tends to zero.'])
    assert not Path(args.output).exists(),'Keep earlier calibration evidence; choose a new output'
    write_json(args.output,result);print(json.dumps({name:round(row['total_gpu_hours'],2) for name,row in scenarios.items()}))

if __name__=='__main__':main()
