#!/usr/bin/env python3
"""Freeze selected diagnostic-only hyperparameters before fresh auxiliary runs."""
import sys,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,prepare,now

def main():
    idx=ROOT/'experiments/stage4';pre=read(idx/'grpo_diagnostic_preregistration_v1.json');diag=read(idx/'grpo_optimization_diagnostic_v1.json')
    assert diag['status']=='DIAGNOSTIC_PASS' and len(diag['conditions'])==6
    from loss_objective_metrics import select
    chosen=select(diag['conditions'],pre['health_gates']);assert diag['selected']==dict(learning_rate=chosen['learning_rate'],clip=chosen['clip'])
    assert read(idx/'loss_objective_eval_isolation_v1.json')['result']=='PASS'
    pair=read(idx/'formal_pair.json');cfg=read(pair['frozen_config']['path'])
    cfg=dict(cfg,mode='auxiliary_loss_objective',policy_objective='native_token_vanilla',learning_rate=chosen['learning_rate'],
        clip_ratio_low=chosen['clip'],clip_ratio_high=chosen['clip'],target_training_groups=512,pause_after_windows=4)
    runs={}
    for arm in ['vanilla','dynamic']:
        out=prepare('aux_grpo_'+arm,'AUXILIARY_LOSS_OBJECTIVE_ABLATION',dict(cfg,sampling_mode=arm))
        runs[arm]=dict(run_id=out.name,path=str(out),config=record(out/'config.json'))
    init=read(cfg['initialization']['path']);policies={'sft':dict(adapter_path=init['adapter_path'],adapter_sha256=init['adapter_sha256'])}
    for arm,r in pair['runs'].items():
        w=Path(r['path'])/'windows/0063';c=read(w/'commit.json');assert c['state_after']['training_groups']==512 and c['state_after']['optimizer_steps']==128
        policies[arm+'_gspo']=dict(adapter_path=str(w/'checkpoint/adapter'),adapter_sha256=record(w/'checkpoint/adapter/adapter_model.safetensors')['sha256'])
    scripts=['loss_objective_actor.py','loss_objective_metrics.py','loss_objective_replay.py','loss_objective_runtime.py','verify_loss_objective.py','evaluate_loss_objective.py','analyze_loss_objective.py','freeze_loss_objective_protocol.py','freeze_loss_objective_eval.py','stage4_resume_boundary.py','stage4_gpu_release_guard.py']
    p=dict(status='FROZEN_BEFORE_AUXILIARY_TRAINING',timestamp=now(),run_class='AUXILIARY_LOSS_OBJECTIVE_ABLATION',runs=runs,config=cfg,
        selected=diag['selected'],selection_basis='optimization diagnostic only',diagnostic=record(idx/'grpo_optimization_diagnostic_v1.json'),preregistration=record(idx/'grpo_diagnostic_preregistration_v1.json'),
        native_audit=record(idx/'loss_objective_native_audit_v1.json'),formal_pair=record(idx/'formal_pair.json'),
        budget=dict(training_groups=512,accepted_mixed_dynamic=512,windows=64,optimizer_steps=128,G=4,training_trajectories=2048),
        sampling='Original unmodified correctness-only controller. Vanilla exact512 first64 prompt IDs/order/encounter seeds freshgeneration; Dynamic same stream rule but real refill count unconstrained.',
        schedule=record(idx/'loss_objective_vanilla_schedule_v1.json'),
        evaluation_manifest=record(idx/'loss_objective_eval_manifest_v1.json'),evaluation_artifact_path=pre['artifact_path'],evaluation_policies=policies,
        evaluation_order=['sft','vanilla_grpo','vanilla_gspo','dynamic_grpo','dynamic_gspo'],
        evaluation_decoding=dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=1024,seed=20260914),
        evaluation_rule='one completed response per prompt; fail closed on transport errors; no valid-wrong retry; final512 only',
        bootstrap_seed=20260914,bootstrap_resamples=10000,length_bins=pre['length_bins'],interpretation_rules=pre['interpretation_rules'],
        real_resume=dict(arms=['vanilla','dynamic'],pause_groups=32,steps=8,next_groups=40,next_steps=10,optimizer_replay=0),
        project_state_snapshot=read(ROOT/'project_state.json'),project_state=record(ROOT/'project_state.json'),
        selection_protocol=record(ROOT/'experiments/stage5/checkpoint_selection_protocol_v1.json'),
        no_selection_inference=True,no_final_tests=True,no_paid_api=True,no_stage6=True,no_primary_gspo_retraining=True,
        runtime_sources={n:record(ROOT/'scripts'/n) for n in scripts},
        failure_policy='Preserve all failures; no hyperparameter changes from validation; exact budgets or FAILED/BLOCKED; Stage4 FULL_PASS retained',
        runtime_overrides=['auxiliary actor direct native vanilla function and matched aggregation','child actor/native reload routed to same auxiliary actor','bounded GPU release guard','boundary-only resume inspection plus full final raw replay','online monitor disabled'],
        length_gradient_proxy='NOT_IDENTIFIABLE without retained per-trajectory gradients; absolute surrogate is not gradient quality')
    immutable(idx/'grpo_objective_protocol_v1.json',p)
    for r in runs.values():immutable(Path(r['path'])/'auxiliary_protocol.json',p)
    print(runs)
if __name__=='__main__':main()
