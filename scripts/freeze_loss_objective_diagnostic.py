#!/usr/bin/env python3
"""Preregister a bounded native-objective diagnostic without generating responses."""
from pathlib import Path
import sys,ast,subprocess,importlib.metadata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,prepare,now

def main():
    idx=ROOT/'experiments/stage4';pair=read(idx/'formal_pair.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==subprocess.check_output(['git','rev-parse','origin/main'],text=True).strip()
    state=read(ROOT/'project_state.json');assert state['stages']['4']['status']=='FULL_PASS'
    assert all(state['stages'][str(i)]['status']=='NOT_STARTED' for i in [5,6])
    assert read(idx/'deliverables_draft_v1.json')['READY_FOR_STAGE5']=='NO'
    import verl.trainer.ppo.core_algos as native
    from verl.workers.config import ActorConfig
    assert native.get_policy_loss_fn('vanilla') is native.compute_policy_loss_vanilla
    assert native.get_policy_loss_fn('vanilla') is not native.get_policy_loss_fn('gspo')
    source=Path(native.__file__);base=source.parents[2];tree=ast.parse(source.read_text())
    names=['compute_policy_loss_vanilla','compute_policy_loss_gspo','compute_grpo_outcome_advantage','agg_loss','get_policy_loss_fn']
    funcs={n.name:dict(first_line=n.lineno,last_line=n.end_lineno) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names}
    audit=dict(timestamp=now(),verl_version=importlib.metadata.version('verl'),native_source=record(source),functions=funcs,
        actor_config=record(base/'workers/config/actor.py'),actor_loss_dispatch=record(base/'workers/utils/losses.py'),
        installation=read(base.parent/'verl-0.9.0.dist-info/direct_url.json'),
        wheel=record('/data/WSH/medical-post-train-artifacts/wheels/verl-0.9.0-py3-none-any.whl'),
        upstream_git_commit=None,commit_limitation='Installed wheel exposes version and source hashes; no upstream Git commit in installation metadata.',
        grpo_policy_function='verl.trainer.ppo.core_algos.compute_policy_loss_vanilla',registry_key='vanilla',not_gspo_alias=True,
        advantage_function='verl.trainer.ppo.core_algos.compute_grpo_outcome_advantage',G=4,epsilon=1e-6,norm_adv_by_std_in_grpo=True,
        token_objective='d=clamp(logp-old,-20,20); r=exp(d); L=max(-A*r,-A*clip(r,1-eps_low,1+eps_high)); negative A uses min(L,-A*3)',
        sequence_objective='r_seq=exp(clamp_max(mean_masked(logp-old),10)); gradient-preserving detached sequence ratio; clipped surrogate',
        aggregation='seq-mean-token-mean',supported_aggregation=['token-mean','token-sum','seq-mean-token-sum','seq-mean-token-sum-norm','seq-mean-token-mean'],
        knobs=dict(clip_ratio=.2,clip_ratio_low='grid',clip_ratio_high='grid',clip_ratio_c=3.,loss_agg_mode='seq-mean-token-mean',global_batch_info={}),
        objective_package_differences=['token vs sequence ratio/clipping','different clipping scale selected without validation','native token log-ratio clamp [-20,20] vs GSPO sequence clamp max10','native vanilla dual clipping at3 for negative advantage; retained and measured'],
        entropy_coefficient=0,kl_loss=False,kl_reward=False,critic=False,reference_worker=False,
        metrics_caveat='native pg_clipfrac_lower is dual clipping, not ordinary lower-bound exceedance; record these separately')
    immutable(idx/'loss_objective_native_audit_v1.json',audit)
    cfg=read(Path(read(idx/'active_diagnostic.json')['path'])/'config.json');cfg.update(policy_objective='native_token_vanilla',diagnostic_lrs=[1e-6,3e-6,1e-5])
    assert record(cfg['trajectory_source']['path'])==cfg['trajectory_source']
    out=prepare('grpo_optimization_diagnostic','DIAGNOSTIC',cfg)
    p=dict(status='PREREGISTERED_BEFORE_DIAGNOSTIC',timestamp=now(),run_id=out.name,artifact_path=str(out),
        grid=[dict(learning_rate=lr,clip=clip) for lr in [1e-6,3e-6,1e-5] for clip in [.1,.2]],
        health_gates=dict(max_normalized_drift=.02,max_objective_clip=.9,min_p01=.1,max_p99=10.,min_ratio=1e-4,max_ratio=1e4,min_active_clip=.001),
        selection_rule='Reject failed/nonfinite/zero-gradient/inactive/explosive/saturated/pathological candidates. Prefer active objective clipping>=0.001 among healthy. Tail and drift within fixed gates define controlled/close. Select smallest LR then conventional native clip0.2 when healthy at that LR. If none active select smallest healthy LR and record inactive-clipping limitation; no post-hoc search.',
        original_batch=cfg['trajectory_source'],new_rollouts=0,fresh_sft_and_optimizer_each_condition=True,freeze_old_across_conditions=True,
        native_audit=record(idx/'loss_objective_native_audit_v1.json'),length_bins=['<192','[192,256)','[256,384)','>=384'],
        interpretation_rules=dict(GSPO_OBJECTIVE_ADVANTAGE_SUPPORTED='Both objective-effect paired95%CIs strictly positive.',
            GRPO_ADVANTAGE='Both objective-effect paired95%CIs strictly negative.',
            GSPO_STABILITY_ONLY='Both objective-effect CIs include0 and both GSPO arms have fewer windows violating the common drift/nonfinite/inactive/ratio-tail gates than their GRPO arms. Different clip scales prevent direct clip-fraction ranking as stability proof.',
            NO_CLEAR_OBJECTIVE_DIFFERENCE='Both objective-effect CIs include0 and stability-only rule not met; not equivalence.',
            INCONCLUSIVE='Incomplete evidence or discordant/mixed objective-effect CI conclusions.'),
        interaction='(D_GSPO-V_GSPO)-(D_GRPO-V_GRPO), paired descriptive single-seed; bootstrap10000 seed20260914',
        plan=['native audit and six frozen-batch diagnostic conditions','freeze selected config and new disjoint validation512','commit protocol before two freshSFT 512/64/128 runs','real32group interruption/resume and raw native replay','five greedy endpoints on new validation only','paired, length, sampling-cost analysis and machine case candidates','append independent auxiliary docs section and pushmain'],
        resources=dict(gpu='RTX5880 Ada46GiB usable; sequential rollout/actor',estimated_checkpoint_bytes=2*64*1409125569,free_disk_bytes=__import__('shutil').disk_usage(out).free,training_estimate_hours='5-8 plus diagnostic/evaluation/verification; Dynamic refill count unknown'),
        risks=['new native objective can fail diagnostics; retain and block rather than tune on validation','native dual clipping is part of package difference','single seed cannot establish universal objective superiority','no final Stage4 verifier or stage transition'],
        protected_state=record(ROOT/'project_state.json'),selection_protocol=record(ROOT/'experiments/stage5/checkpoint_selection_protocol_v1.json'),
        source_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        diagnostic_sources={n:record(ROOT/'scripts'/n) for n in ['loss_objective_actor.py','loss_objective_metrics.py','loss_objective_replay.py','loss_objective_diagnostic.py','freeze_loss_objective_diagnostic.py']})
    immutable(idx/'grpo_diagnostic_preregistration_v1.json',p);immutable(out/'preregistration.json',p)
    print(out)
if __name__=='__main__':main()
