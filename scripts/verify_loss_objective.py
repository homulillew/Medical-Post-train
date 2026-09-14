#!/usr/bin/env python3
"""Strict auxiliary-run replay; reuses unmodified formal raw/native-update gates."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,validate,now,encounter
from loss_objective_runtime import check


def verify_run(path,windows=64,require_final=True):
    from verify_stage4 import raw_batch,checkpoint,probe_delta
    from loss_objective_replay import update
    from medical_posttrain.rl.online import initial_state
    from medical_posttrain.rl.controller import select_groups,window_metrics,advance
    from medical_posttrain.sampling.dynamic import Stream
    from medical_posttrain.evidence.stage2 import jsonlines
    p=check();cfg=read(path/'config.json');arm=cfg['sampling_mode'];run_config=p['runs'][arm]['config'];assert record(path/'config.json')==run_config
    assert read(ROOT/'project_state.json')==p['project_state_snapshot']
    assert cfg['mode']=='auxiliary_loss_objective' and cfg['target_training_groups']==512 and cfg['sampling_mode'] in ('vanilla','dynamic')
    state=initial_state(validate(cfg));schedule=read(p['schedule']['path'])['windows']
    stream=Stream([r['prompt_id'] for r in jsonlines(cfg['pool']['path'])],cfg['seed'],cfg['stream_domain'])
    marker=None;previous_probe=None;sources=[];seen=set()
    attempts={read(a/'resume_start.json')['state']['policy_windows']:a for a in path.glob('attempt_*') if (a/'resume_start.json').exists()}
    for i in range(windows):
        if i in attempts:
            a=attempts[i];assert read(a/'resume_start.json')['state']==state
            sync=read(a/'initial_sync/receipt.json');assert sync['policy_version']==state['policy_version']
            previous_probe=read(a/'initial_sync/first.json')
            assert probe_delta(previous_probe,read(a/'initial_sync/repeat.json'))<=1e-4
        w=path/'windows'/f'{i:04d}';assert read(w/'state_before.json')==state
        groups=[];decisions=[];selected=[]
        for b in sorted((w/'batches').iterdir()):
            gg=raw_batch(b,cfg)
            for g in gg:
                expected=encounter(stream,state['cursor']+len(groups),path.name,state['policy_version'])
                assert all(g[k]==v for k,v in expected.items())
                assert g['group_id'] not in seen;seen.add(g['group_id'])
                for r in g['responses']:
                    assert r['run_id']==path.name and r['config_sha256']==run_config['sha256']
                    assert r['policy_version']==r['adapter_sha256']==state['policy_version']
                    assert r['reward_version']==cfg['reward_manifest']['sha256']
                groups.append(g)
            accepted,dd=select_groups(gg,state['policy_version'],arm,already=len(selected))
            assert read(b/'dispositions.json')==dd
            selected.extend(accepted);decisions.extend(dd)
            sources.extend(record(b/n) for n in ['raw.json','reservation.json','scored.json','semantic_vectors.npy','semantic_encoding.json'])
        if arm=='vanilla':assert len(groups)==8
        if arm=='vanilla':
            expected_ids=[(e['encounter_index'],e['prompt_id'],e['request_seed']) for e in schedule[i]['encounters']]
            assert [(g['encounter_index'],g['prompt_id'],g['request_seed']) for g in groups]==expected_ids
        else:
            assert all(d['eligible'] for d in decisions if d['disposition']=='selected')
        selection=read(w/'selection.json')
        assert len(selected)==8 and selection==dict(groups=selected,decisions=decisions,policy_version=state['policy_version'])
        assert window_metrics(groups,decisions)==read(w/'rollout_metrics.json')
        trained=update(w/'update',selected,cfg)
        assert trained['optimizer_steps_after']==(i+1)*2
        current=checkpoint(w/'checkpoint');assert current['optimizer_step']==(i+1)*2
        actual=read(w/'actor/actual_native_configs.json')
        assert actual['loss']['loss_agg_mode']=='seq-mean-token-mean' and actual['loss']['policy_loss']['loss_mode']=='vanilla'
        assert actual['loss']['clip_ratio_c']==3.0
        loaded=read(w/'actor/loaded_identity.json')
        assert loaded['trainable_digest']==trained['initial_trainable_digest']
        assert current['trainable_digest']==trained['final_trainable_digest']
        if marker:
            assert loaded['trainable_digest']==marker['trainable_digest']
            assert loaded['optimizer_digest']==marker['optimizer_digest']
            assert loaded['scheduler']==marker['scheduler'] and loaded['optimizer_steps']==[i*2]
            assert read(w/'actor/resume_receipt.json')['result']=='PASS'
        else:assert loaded['optimizer_steps']==[]
        marker=current
        policy=record(w/'checkpoint/adapter/adapter_model.safetensors')['sha256']
        sync=read(w/'sync/receipt.json')
        assert sync['policy_version']==sync['adapter_sha256']==policy and sync['previous_policy_delta']>1e-6
        first=read(w/'sync/first.json');repeat=read(w/'sync/repeat.json')
        assert first['token_ids']==repeat['token_ids']
        assert probe_delta(first,repeat)==sync['repeat_error']<=1e-4
        assert probe_delta(previous_probe,first)==sync['previous_policy_delta'];previous_probe=first
        assert read(w/'actor_exit.json')['exit_code']==0
        commit=read(w/'commit.json');assert commit['state_before']==state
        state=advance(state,groups,decisions,policy,2);assert state==commit['state_after']
        for ref in commit['artifacts']:assert record(ref['path'])==ref
        sources.extend(commit['artifacts']);sources.append(record(w/'commit.json'))
    assert state['training_groups']==windows*8 and state['optimizer_steps']==windows*2
    if arm=='vanilla':assert state['generated_groups']==windows*8
    if require_final:
        assert windows==64 and read(path/'summary.json')['state']==state==read(path/'checkpoint.json')['state']
        assert len(list((path/'windows').iterdir()))==64
        resume=read(path/'physical_resume_receipt.json');assert resume['result']=='PASS' and not resume['optimizer_replayed']
        assert resume['new_pid']!=resume['old_pid'] and read(path/'termination_observed.json')['dead']
        assert read(path/'pause_ready.json')['state']==read(path/'attempt_002/resume_start.json')['state']
        final=read(path/'final_reload/result.json');assert final['result']=='PASS' and final['finite_logprobs']
        assert final['loaded_identity']['optimizer_steps']==[128] and final['optimizer_steps_added']==0
        assert final['loaded_identity']['trainable_digest']==marker['trainable_digest']
        assert final['loaded_identity']['optimizer_digest']==marker['optimizer_digest']
        assert read(path/'final_reload/exit.json')['exit_code']==0
    assert not (path/'validation').exists(),'No monitor/selection/test inference inside training'
    return dict(result='PASS',timestamp=now(),scope='AUXILIARY_FULL_RAW' if require_final else 'REAL_32_GROUP_INTEGRATION',
        run_id=path.name,state={k:v for k,v in state.items() if not k.endswith('exposure')},
        frozen_protocol=record(ROOT/'experiments/stage4/grpo_objective_protocol_v1.json'),
        selection_replay='Vanilla exact frozen first64 schedule; Dynamic native correctness-only mixed/refill deterministic stream',
        scientific_gates=['raw token decode','parser/correctness','semantic vectors/hybrid reward','old logprobs frozen before optimizer',
                         'native GRPO advantages/native vanilla token loss','optimizer/scheduler/checkpoint lineage','vLLM policy synchronization'],
        sources=sources,verifier=record(Path(__file__)),native_gate_source=record(ROOT/'scripts/verify_stage4.py'),
        project_state_unchanged=True,selection1024_untouched=True,final_tests_untouched=True)
