#!/usr/bin/env python3
"""Freeze the reviewed candidate, then prepare a fresh pair from a clean commit."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from medical_posttrain.rl.common import ROOT,INDEX,read,record,sha256,immutable,durable,inherited,prepare,validate,now

CONFIG=ROOT/'configs/stages/s4_formal_shared.json'
PROTOCOL=INDEX/'formal_validation_protocol.json'


def candidate():
    assert not (INDEX/'formal_pair.json').exists()
    protocol=read(INDEX/'validation_protocol.json')
    protocol.update(generated_total_token_stride=1000000,
        token_axis='Physical training rollout prompt tokens once per prompt encounter + all generated output tokens, including rejects and overflow',
        token_exclusions=['validation','sync/control'],logical_prompt_tokens='physical prompt tokens * G, reported separately',
        token_schedule='All positive integer multiples of 1000000 until this run ends; same unbounded deterministic grid for both variants',
        trigger_rule='Evaluate the first committed window crossing each threshold. Save actual tokens/window; coalesce thresholds crossed in the same window with any group milestone. Never interpolate or substitute a later checkpoint.',
        comparison_rule='Compare shared reached thresholds; retain actual overshoot. Points above the common observed range are not equal-token comparisons.')
    immutable(PROTOCOL,protocol)
    cfg=inherited()
    cfg.update(mode='formal',target_training_groups=5000,stream_domain='stage4:formal:pair1',
        validation_protocol=record(PROTOCOL),checkpoint_interval_windows=1,
        health_protocol=dict(clip_threshold=.9,clip_consecutive_windows=8,
            length_block_windows=16,length_consecutive_blocks=3,length_cap_fraction=.9),
        pilot_receipts={v:record(INDEX/f'{v}_pilot_verification.json') for v in ('vanilla','dynamic')})
    immutable(CONFIG,cfg)
    print('Candidate and protocol recorded. Pair is NOT frozen or prepared yet.')


def freeze():
    assert not (INDEX/'formal_pair.json').exists(), 'Inspect existing formal pair; never overwrite'
    assert subprocess.check_output(['git','status','--porcelain'],text=True)=='', 'Commit reviewed implementation/protocol/evidence first'
    commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    frozen_at=now();cfg=read(CONFIG);initial=validate(cfg)
    assert cfg['mode']=='formal' and cfg['target_training_groups']==5000
    tests=read(INDEX/'formal_preflight_tests_v2.json');assert tests['result']=='PASS'
    for p,h in tests['runtime_hashes'].items():assert sha256(ROOT/p)==h, f'Tested runtime changed: {p}'
    pilot_shared=read(INDEX/'pilot_pair.json')['shared_config']
    for key in inherited():assert cfg[key]==pilot_shared[key], f'Candidate differs from reviewed pilot: {key}'
    adapter_config=read(Path(initial['adapter_path'])/'adapter_config.json')
    assert adapter_config['r']==32 and adapter_config['lora_alpha']==64 and adapter_config['lora_dropout']==0
    faults=read(INDEX/'recovery_fault_injections.json');assert faults['result']=='PASS' and len(faults['scenarios'])==3
    for result in faults['scenarios'].values():
        assert result['result']=='PASS' and result['real_process_termination']
        for ref in result['sources']:assert record(ref['path'])==ref
        m=read(INDEX/result['run_id']/'manifest.json')
        for p,h in m['source_hashes'].items():
            if p.startswith('src/medical_posttrain/rl/') or p=='scripts/run_stage4.py':assert sha256(ROOT/p)==h
    for variant in ('vanilla','dynamic'):
        r=read(INDEX/f'{variant}_pilot_verification.json')
        assert r['result']=='PASS' and r['state']['training_groups']==512
    base=ROOT/'experiments/stage0/s0_snapshot_20260908T132515_ec08e3/attempt_001/snapshot_manifest.json'
    base_manifest=read(base);reward=read(cfg['reward_manifest']['path'])
    for f in base_manifest['files']+initial['files']+reward['model_files']:
        assert sha256(f['path'])==f['sha256'],f['path']
    sample=Path(read(INDEX/'pilot_pair.json')['runs']['vanilla']['path'])/'windows/0063/checkpoint'
    checkpoint_bytes=sum(p.stat().st_size for p in sample.rglob('*') if p.is_file())
    free=shutil.disk_usage(sample).free;projection=checkpoint_bytes*1250*1.1+100*1024**3
    assert free>projection
    # Everything above is read-only: this is the exact clean source state at freeze.
    assert subprocess.check_output(['git','status','--porcelain'],text=True)==''
    clean=dict(commit=commit,porcelain='',timestamp=now(),scope='Entire tracked and untracked workspace clean immediately before creating operational run manifests')
    runs={}
    for variant in ('vanilla','dynamic'):
        out=prepare('formal_'+variant,'FORMAL',dict(cfg,sampling_mode=variant))
        runs[variant]=dict(run_id=out.name,path=str(out),config=record(out/'config.json'),manifest=record(out/'manifest.json'))
    manifests=[read(r['manifest']['path']) for r in runs.values()]
    execution=lambda m:{k:v for k,v in m['source_hashes'].items() if k.startswith('src/') or k in ('scripts/run_stage4.py','scripts/reload_stage4.py','scripts/audit_stage4_boundary.py','scripts/continue_stage4_formal.py')}
    assert execution(manifests[0])==execution(manifests[1])
    pair=dict(pair_id='stage4_formal_'+uuid.uuid4().hex[:12],mode='FORMAL',formal_frozen=True,
        frozen_at=frozen_at,code_commit=commit,clean_git_state=clean,
        operational_manifest_note='Preparing the first run creates untracked evidence visible in the second manifest dirty_state; source bytes and source commit are identical and were clean at freeze.',
        frozen_config=record(CONFIG),validation_protocol=record(PROTOCOL),runs=runs,execution_hashes=execution(manifests[0]),
        initialization=cfg['initialization'],sft_adapter_sha256=initial['adapter_sha256'],
        base_model_manifest=record(base),base_model_files=base_manifest['files'],candidate_pool=cfg['pool'],reward_manifest=cfg['reward_manifest'],
        parser=record(ROOT/'src/medical_posttrain/reward/parser.py'),semantic=record(ROOT/'src/medical_posttrain/reward/semantic.py'),
        recovery_gate=record(INDEX/'recovery_fault_injections.json'),decision=record(INDEX/'formal_freeze_decision.json'),
        stream=dict(seed=cfg['seed'],domain=cfg['stream_domain'],policy='Same deterministic candidate permutations and per-encounter request seed; Dynamic alone filters accuracy groups and refills; overflow expires'),
        resource_preflight=dict(free_bytes=free,checkpoint_bytes=checkpoint_bytes,formal_projection_with_reserve_bytes=projection),
        fresh_sft=True,fresh_optimizer=True,fresh_scheduler=True,fresh_stream=True,
        budget_per_variant=dict(training_groups=5000,trajectories=20000,policy_windows=625,optimizer_steps=1250),
        intended_difference='Accuracy-only Dynamic Sampling versus Vanilla; all principal model/reward/GSPO/sampling/validation settings shared',
        known_confounders=['Single training seed','Same seed/config does not guarantee byte-identical vLLM sampling across fresh processes; pilot/smoke variation retained',
            'Rollback restores exact saved input identities/RNG, but replayed floating-point optimizer output is not guaranteed bitwise identical; Fault A measurement retained'],
        stop_conditions=['Nonfinite or corrupt evidence fails closed','Persistent clip/length diagnostic pause under shared predeclared health protocol','Bounded refill starvation BLOCKED; no eligibility or budget relaxation'],
        selection='Retain all intermediate and final-budget checkpoints; no selection1024/test/Stage5 use',READY_FOR_STAGE5='NO')
    immutable(INDEX/'formal_pair.json',pair)
    for v,r in runs.items():durable(INDEX/f'active_formal_{v}.json',r)
    state=read(ROOT/'project_state.json');state['stages']['4']['run_ids'] += [r['run_id'] for r in runs.values()]
    durable(ROOT/'project_state.json',state)
    print({v:r['path'] for v,r in runs.items()})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['candidate','freeze']);a=p.parse_args()
    candidate() if a.action=='candidate' else freeze()
