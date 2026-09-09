"""Actual Stage 2 accounting and conditional Stage 3/4 rollout projections."""
from pathlib import Path
from medical_posttrain.evidence import write_json
from medical_posttrain.evidence.stage2 import read,record,selected_path,INDEX

formal=selected_path('formal');s=read(formal/'summary.json');assert s['status']=='PASS' and s['completed_responses']==4000
mean=s['lengths']['output_tokens']['mean'];speed=s['costs']['output_tokens_per_second'];acceptance=s['group_fractions']['mixed']
projections={}
for name,groups in [('vanilla',5000),('dynamic',5000/acceptance if acceptance else None)]:
    tokens=groups*4*mean if groups is not None else None
    projections[name]=dict(accepted_training_groups=5000,estimated_generated_groups=groups,estimated_output_tokens=tokens,
                           estimated_generation_gpu_hours=tokens/speed/3600 if tokens is not None else None,
                           optimizer_and_old_logprob_gpu_hours='UNKNOWN: no policy update run in Stage2',
                           switching_and_validation_gpu_hours='UNKNOWN: rollout-only projection excludes future training/evaluation overhead')
scenarios={}
for name,a,factor in [('acceptance_half',acceptance/2,1.),('point_estimate',acceptance,1.),('throughput_half',acceptance,.5)]:
    scenarios[name]=dict(acceptance=a,output_tokens_per_second=speed*factor,
                        dynamic_generated_groups=5000/a if a else None,
                        dynamic_generation_gpu_hours=5000*4*mean/(a*speed*factor)/3600 if a else None)
run_costs=[]
for folder in sorted(INDEX.glob('s2_*')):
    m=read(folder/'manifest.json');root=Path(m['artifact_root']);status=read(root/'status.json')
    gpu_attempts=[];cpu_attempts=[]
    for attempt in sorted(root.glob('attempt_*')):
        state=read(attempt/'status.json');cmd=read(attempt/'command.json');action=cmd[cmd.index('--action')+1]
        entry=dict(attempt=attempt.name,action=action,status=state['status'],measured_wall_seconds=state.get('wall_seconds'),
                   unknown_elapsed=state.get('wall_seconds') is None)
        (gpu_attempts if action in ('rollout','score') else cpu_attempts).append(entry)
    run_costs.append(dict(run_id=root.name,run_class=m['run_class'],purpose=m['purpose'],status=status['status'],gpu_attempts=gpu_attempts,cpu_attempts=cpu_attempts))
actual=sum(a['measured_wall_seconds'] or 0 for r in run_costs for a in r['gpu_attempts'])
result=dict(stage=2,classification='Retrospective accounting plus conditional estimates, not actual RL training',
            measured=dict(mean_output_tokens=mean,p95_output_tokens=s['lengths']['output_tokens']['p95'],
                          mixed_fraction=acceptance,output_tokens_per_second=speed,formal_costs=s['costs'],
                          tracked_gpu_worker_seconds=actual,tracked_gpu_worker_hours=actual/3600,run_costs=run_costs),
            projections=projections,sensitivity=scenarios,
            stage3_256_mixed_groups_estimated_generation_gpu_hours=256*4*mean/(acceptance*speed)/3600 if acceptance else None,
            sources=[record(formal/'summary.json'),record(selected_path('smoke')/'summary.json')],
            caveats=['All estimates use the single fixed SFT policy; acceptance and lengths change after optimization.',
                     'No actual Stage3 refill or Stage4 optimizer execution is included.',
                     'Mandatory 5000 training groups per variant, G4 and original research contract remain unchanged.',
                     'Tracked GPU worker wall includes CPU setup/scoring/bookkeeping and sleep/wake inside those workers, not utilization-integrated kernel hours; process teardown gaps unmeasured.',
                     'Generation tokens/s includes the real bounded-batch generation calls, not cold load or semantic encoding. The projection excludes reference embedding cache startup.',
                     'If acceptance tends to zero, dynamic sampling has no finite worst-case time bound.'])
path=INDEX/'compute_calibration.json';assert not path.exists();write_json(path,result)
p=Path('docs/implementation/COMPUTE_BUDGET.md')
with p.open('a') as f:
    f.write('\n\n## Stage 2 actual profiling calibration\n\n')
    f.write(f"Formal `{formal.name}`: measured mixed fraction {acceptance:.6f}, mean output {mean:.3f} tokens, P95 {s['lengths']['output_tokens']['p95']:.3f}, bounded-batch LoRA generation {speed:.3f} output tokens/s. Tracked Stage2 GPU-worker wall {actual/3600:.4f}h includes smoke/formal generation and sequential semantic scoring; CPU diagnostics separately recorded.\n\n")
    f.write('| Variant | Contract training groups | Estimated generated groups | Estimated output tokens | Estimated generation GPU hours |\n| --- | ---: | ---: | ---: | ---: |\n')
    for name,r in projections.items():
        f.write(f"| {name} | 5000 | {r['estimated_generated_groups']} | {r['estimated_output_tokens']} | {r['estimated_generation_gpu_hours']} |\n")
    f.write('\nThese are fixed-initial-policy rollout-only estimates. Actor updates, old-logprob recomputation, switching and validation costs remain unmeasured in Stage2 and must be added after their stages. Acceptance changes with policy updates; 1/P_mixed is not a measured Dynamic training amplification. No 5000-group budget reduction is made. Stage1 historical scenarios remain above, not overwritten. Machine-readable source hashes, actual run costs and adverse sensitivity: `experiments/stage2/compute_calibration.json`.\n')
print(projections)
