"""Stage 3 acceptance gates recomputed from raw generation and transaction evidence."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import subprocess
import zipfile
import numpy as np
from medical_posttrain.evidence import now, sha256
from medical_posttrain.evidence.stage2 import read, jsonlines
from medical_posttrain.evidence.stage3 import INDEX
from medical_posttrain.data.exam import messages
from medical_posttrain.reward.parser import reasoning_text
from medical_posttrain.reward.hybrid import reward
from medical_posttrain.sampling.dynamic import Stream, initial_state, apply_batch, stop_reason
from medical_posttrain.sampling.stage3 import frozen
from medical_posttrain.sampling.analysis import summarize


def runtime_gate(path, mode):
    from transformers import AutoTokenizer
    from tokenizers.decoders import DecodeStream
    cfg=read(path/'config.json');manifest=read(path/'manifest.json');initial=frozen(cfg)
    assert cfg['mode']==mode and cfg['stage']==3 and cfg['optimizer_updates']==0
    assert cfg['sampling_metric']=='acc' and cfg['sampling']['n']==4
    assert cfg['target_accepted']==(256 if mode=='formal' else None)
    assert cfg['stream_domain']=='stage3:'+mode and cfg['seed']==42
    assert cfg['request_batch_prompts']==16 and 0<cfg['max_num_generation_batches']<=128
    assert manifest['config_sha256']==sha256(path/'config.json')
    if mode=='formal':assert not manifest['dirty_state'], 'Formal source must be committed before prepare'
    with zipfile.ZipFile(path/'source.zip') as archive:
        for name,digest in manifest['source_hashes'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest,name
        for name,digest in cfg['controller_hashes'].items():assert manifest['source_hashes'][name]==digest
    inherited=read('configs/stages/s2_formal_1024.json')
    for key in ('model','policy_version','initialization','pool','selection','reward_manifest','sampling','engine','seed','execution_hashes'):
        assert cfg[key]==inherited[key],key
    for key in ('prerequisite_stage2',):assert sha256(cfg[key]['path'])==cfg[key]['sha256']
    pool={r['prompt_id']:r for r in jsonlines(cfg['pool']['path'])};assert len(pool)==15000
    stream=Stream(list(pool),cfg['seed'],cfg['stream_domain']);state=initial_state(cfg)
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    seen=set();allrows=[];batchdirs=sorted((path/'batches').glob('*'))
    for i,batch in enumerate(batchdirs):
        assert batch.name==f'{i:04d}' and stop_reason(state,cfg) is None
        reservation=read(batch/'reservation.json');raw=read(batch/'raw.json');scored=read(batch/'scored.json');commit=read(batch/'commit.json')
        assert reservation['state_before']==state==commit['state_before']
        assert len(raw['groups'])==len(scored['groups'])==cfg['request_batch_prompts']
        expected=[stream.encounter(j,path.name,cfg['policy_version']) for j in range(state['encounter_index'],state['encounter_index']+16)]
        assert reservation['encounters']==expected
        assert raw['generation_seconds']>0
        encoding=read(batch/'semantic_encoding.json');vectors=np.load(batch/'semantic_vectors.npy')
        assert np.isfinite(vectors).all() and len(vectors)==len(encoding['texts'])
        rm=read(cfg['reward_manifest']['path'])
        assert encoding['encoder_id']==rm['encoder_id'] and encoding['revision']==rm['encoder_revision'] and encoding['chunk_policy']==rm['chunk_policy']
        for meta in encoding['metadata']:
            assert meta['truncated_tokens']==0
            if meta['reason']=='encoded':
                spans=meta['chunk_spans'];assert spans[0][0]==0 and spans[-1][1]==meta['encoder_tokens']
                assert all(0<b-a<=480 for a,b in spans) and all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
        for e,pids,g,sg in zip(expected,reservation['prompt_ids'],raw['groups'],scored['groups']):
            assert all(g[k]==v for k,v in e.items())
            source=pool[e['prompt_id']]
            expectedp=tok.apply_chat_template(messages(source),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
            assert pids==expectedp and g['prompt_tokens']==len(pids)
            assert {k:v for k,v in g.items() if k!='responses'}=={k:v for k,v in sg.items() if k!='responses'}
            assert len(g['responses'])==len(sg['responses'])==4
            for r,original in zip(sg['responses'],g['responses']):
                assert all(r[k]==v for k,v in original.items())
                assert r['run_id']==path.name
                assert r['trajectory_id'] not in seen;seen.add(r['trajectory_id']);allrows.append(r)
                assert r['trajectory_id']==e['group_id']+':'+str(r['member_index']) and r['member_index'] in range(4)
                assert r['prompt_id']==e['prompt_id'] and r['group_id']==e['group_id'] and r['request_seed']==e['request_seed']
                assert r['policy_version']==cfg['policy_version'] and r['adapter_sha256']==initial['adapter_sha256']
                assert r['config_sha256']==manifest['config_sha256'] and r['reward_version']==cfg['reward_manifest']['sha256']
                assert r['finish_reason'] in ('stop','length') and 0<len(r['token_ids'])==r['output_tokens']<=1024 and r['prompt_tokens']==len(pids)
                decoder=DecodeStream(ids=pids,skip_special_tokens=True)
                decoded=''.join(decoder.step(tok._tokenizer,t) or '' for t in r['token_ids'])
                assert decoded==r['raw_output'],(r['trajectory_id'],'raw_decode')
                assert r['question']==source['question'] and r['options']==source['options'] and r['ground_truth']==source['answer_set']
                reasoning=reasoning_text(decoded);ri=r['reasoning_embedding_index'];ei=r['reference_embedding_index']
                assert encoding['texts'][ri]==reasoning and encoding['texts'][ei]==source['reference_explanation']
                cosine=float(vectors[ri]@vectors[ei]);sem=float(np.clip(cosine,0,1)) if reasoning.strip() and source['reference_explanation'].strip() else 0.
                assert abs(r['semantic']-sem)<1e-6 and abs(r['semantic_cosine']-cosine)<1e-6
                rr=reward(decoded,source['answer_set'],sem,''.join(source['options']),r['finish_reason'])
                assert r['acc']==rr['acc'] and r['format']==rr['format']
                assert r['parsed']==json.loads(json.dumps(rr['parser'])) and r['parsed_answer']==rr['parser']['answer_set']
                assert r['format_valid']==rr['parser']['valid_format'] and r['ambiguous']==rr['parser']['ambiguous'] and r['parse_error']==rr['parser']['error_type']
                assert abs(r['total_reward']-rr['score'])<1e-6 and abs(r['semantic_contribution']-rr['semantic_contribution'])<1e-6
                span=rr['parser']['matched_span'];answer=decoded[span[0]:span[1]].strip() if span else ''
                assert r['reasoning_tokens']==len(tok.encode(reasoning,add_special_tokens=False)) and r['answer_tokens']==len(tok.encode(answer,add_special_tokens=False))
        state,decisions=apply_batch(state,scored['groups'],cfg,len(pool))
        assert state==commit['state_after'] and decisions==commit['decisions']
        for f in commit['artifacts']:assert sha256(f['path'])==f['sha256']
        assert all(d['valid'] for d in decisions), 'A real invalid group must retain FAILED status'
        for g,d in zip(scored['groups'],decisions):
            count=sum(r['acc'] for r in g['responses'])
            assert (d['disposition'] in ('accepted','overflow_eligible') and 0<count<4) or (d['disposition']=='rejected_all_wrong' and count==0) or (d['disposition']=='rejected_all_correct' and count==4)
    assert state==read(path/'checkpoint.json')
    summary=read(path/'summary.json');recomputed=summarize(path)
    for k,v in recomputed.items():assert summary[k]==v,k
    assert state['valid_generated_groups']==state['all_wrong_groups']+state['mixed_groups']+state['all_correct_groups']
    assert state['mixed_groups']==state['accepted_mixed_groups']+state['overflow_mixed_groups']
    assert state['completed_valid_trajectories']==4*state['valid_generated_groups']==len(allrows)
    assert sum(state['output_tokens_by_disposition'].values())==sum(r['output_tokens'] for r in allrows)
    assert summary['status']==stop_reason(state,cfg)==('FULL_PASS' if mode=='formal' else 'SMOKE_PASS')
    assert read(path/'status.json')['status']==summary['status']
    if mode=='formal':assert state['accepted_mixed_groups']==256 and state['batches']>1 and state['all_wrong_groups']+state['all_correct_groups']>0
    else:assert state['generated_groups']==32
    for attempt in sorted(path.glob('attempt_*')):
        identity=read(attempt/'identity_receipt.json')
        controls={k:read(attempt/f'identity_{k}.json') for k in ('base','sft','base_negative','sft_repeat','after_wake')}
        def delta(a,b):
            return max(abs(x[k]-y[k]) for x,y in zip(controls[a]['prompt_logprobs'][1:],controls[b]['prompt_logprobs'][1:]) for k in x.keys()&y.keys())
        for field,a,b in [('base_sft_delta','base','sft'),('base_negative_error','base','base_negative'),('repeat_error','sft','sft_repeat'),('wake_error','sft','after_wake')]:assert identity[field]==delta(a,b)
        assert controls['base']['token_ids']==controls['base_negative']['token_ids']
        assert controls['sft']['token_ids']==controls['sft_repeat']['token_ids']==controls['after_wake']['token_ids']
        assert identity['base_sft_delta']>1e-5 and all(identity[k]<=1e-4 for k in ('repeat_error','wake_error','base_negative_error'))
        assert identity['adapter_sha256']==initial['adapter_sha256'] and identity['optimizer_updates']==0
        env=read(attempt/'runtime_environment.json');assert env['VLLM_BATCH_INVARIANT']=='1' and env['VLLM_USE_V2_MODEL_RUNNER']=='0' and env['VLLM_USE_FLASHINFER_SAMPLER']=='0'
        assert read(attempt/'lora_shrink_config.json')['split_k']==1
        assert read(attempt/'vllm_args.json')==dict(model=cfg['model'],tokenizer=cfg['model'],**cfg['engine'])
        actual=read(attempt/'code.json')['source_hashes']
        assert all(actual[k]==v for k,v in cfg['controller_hashes'].items())
        events=jsonlines(attempt/'metrics.jsonl')
        starts=[e for e in events if e['event']=='generation_started'];ends=[e for e in events if e['event']=='generation_completed']
        assert [e['batch'] for e in starts]==[e['batch'] for e in ends], 'Unresolved generation attempt cost'
        for event in ends:
            raw=read(path/'batches'/f"{event['batch']:04d}"/'raw.json')
            assert event['seconds']==raw['generation_seconds'] and event['output_tokens']==sum(r['output_tokens'] for g in raw['groups'] for r in g['responses'])
    return summary,allrows


def resume_gate(smoke):
    pause=read(smoke/'pause_ready.json');termination=read(smoke/'termination.json')
    start=read(smoke/'attempt_002/resume_start.json');first=read(smoke/'attempt_001/resume_start.json')
    assert first['state']['generated_groups']==0 and start['state']['generated_groups']==16
    assert pause['state']==termination['state']==start['state']
    assert pause['next_encounters']==termination['next_encounters']==start['next_encounters']
    assert pause['pid']==termination['pid']==first['pid'] and start['pid']!=first['pid']
    assert termination['signal']=='SIGTERM' and read(smoke/'attempt_001/status.json')['status']=='INTERRUPTED'
    assert read(smoke/'batches/0000/commit.json')['state_after']==start['state']
    assert read(smoke/'batches/0001/reservation.json')['encounters']==start['next_encounters']
    assert read(smoke/'batches/0001/commit.json')['state_before']==start['state']
    death=read(smoke/'termination_observed.json');assert death['old_process_exited'] and death['gpu_memory_used_mib']<100
    assert start['checkpoint']['sha256']==pause['checkpoint']['sha256']


def verify(smoke_only=False):
    gates={};errors=[];counts={};selected=read(INDEX/'selected_runs.json')
    def gate(name,fn):
        try:
            value=fn();gates[name]=True;return value
        except Exception as error:
            import traceback
            gates[name]=False;errors.append(dict(gate=name,error=repr(error),traceback=traceback.format_exc()))
    def isolation():
        import jsonschema
        state=read('project_state.json');jsonschema.validate(state,read('schemas/project_state.schema.json'))
        assert state['stage0']['status']=='VERIFIED' and state['stages']['1']['status']==state['stages']['2']['status']=='DONE'
        for k in ('4','5','6'):
            assert state['stages'][k]['status']=='NOT_STARTED' and not state['stages'][k]['run_ids']
        assert sha256('contracts/stage_budgets.json')==state['contract_sha256']=='8ba60bd2e724fbbb74e26f4ef677eb6c6d51eaafcba6eb3bcb0b2eb588a51f24'
        for k in ('1','2'):
            ref=state['stages'][k]['verification_receipt'];assert sha256(ref['path'])==ref['sha256'] and read(ref['path'])['result']=='PASS'
        if state['stages']['3']['status'] in ('VERIFIED','DONE'):
            ref=state['stages']['3']['verification_receipt']
            assert sha256(ref['path'])==ref['sha256'] and read(ref['path'])['result']=='PASS'
        assert read(INDEX/'prerequisite-stage2.json')['result']=='PASS'
        base=read('experiments/stage0/s0_snapshot_20260908T132515_ec08e3/attempt_001/snapshot_manifest.json')
        for f in base['files']+read('experiments/stage1/initialization_manifest.json')['files']+read('experiments/stage2/reward_manifest.json')['model_files']:
            assert sha256(f['path'])==f['sha256'],f['path']
    gate('verified_predecessors_and_unchanged_contract_isolation',isolation)
    def tests():
        result=subprocess.run(['.venv-analysis/bin/python','-m','pytest','tests/test_dynamic_sampling.py','-q'],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
        import ast
        for source in ('src/medical_posttrain/sampling/stage3.py','src/medical_posttrain/sampling/dynamic.py','scripts/run_stage3.py'):
            calls=[n.func.attr for n in ast.walk(ast.parse(Path(source).read_text())) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
            assert not {'backward','step','zero_grad'}&set(calls),source
        receipt=read(INDEX/'tests-final.json');assert receipt['exit_code']==0
        assert sha256(receipt['log']['path'])==receipt['log']['sha256']
        for name,digest in receipt['test_hashes'].items():assert sha256(name)==digest,name
        return result.stdout
    gate('sixteen_patterns_reward_independence_validity_stream_overflow_resume_starvation',tests)
    path=lambda name:Path(read(INDEX/selected[name]/'manifest.json')['artifact_root'])
    smoke=path('smoke');gate('real_smoke_raw_pipeline_and_token_conservation',lambda:runtime_gate(smoke,'smoke'))
    gate('physical_termination_new_process_resume',lambda:resume_gate(smoke))
    if not smoke_only:
        formal=path('formal');result=gate('formal_new_generation_refill_256_raw_reward_and_conservation',lambda:runtime_gate(formal,'formal'))
        if result:
            summary,rows=result;counts.update(generated_groups=summary['generated_groups'],accepted_mixed_groups=summary['accepted_mixed_groups'],optimizer_updates=0)
        def separation():
            cfg=read(formal/'config.json');assert sha256(cfg['smoke_receipt']['path'])==cfg['smoke_receipt']['sha256']
            assert cfg['controller_hashes']==read(smoke/'config.json')['controller_hashes']
            assert read(cfg['smoke_receipt']['path'])['result']=='PASS'
            assert read(formal/'attempt_001/resume_start.json')['state']['generated_groups']==0
            assert formal.name!=smoke.name and read(formal/'config.json')['stream_domain']!=read(smoke/'config.json')['stream_domain']
            fsummary=read(formal/'summary.json');ssummary=read(smoke/'summary.json')
            overlap=len(set(fsummary['exposure']['prompt_exposure_count'])&set(ssummary['exposure']['prompt_exposure_count']))
            assert read(INDEX/'stream_overlap.json')['smoke_formal_prompt_overlap']==overlap
            assert read(INDEX/'stream_overlap.json')['stage2_formal_prompt_overlap']==fsummary['exposure']['stage2_formal_overlap']
        gate('fresh_smoke_formal_stream_separation_and_frozen_inputs',separation)
        def reports():
            from medical_posttrain.data.stage1 import text_hash
            rr=[r for b in sorted((formal/'batches').glob('*')) for g in read(b/'scored.json')['groups'] for r in g['responses']];byid={r['trajectory_id']:r for r in rr}
            review=read(INDEX/'manual_review.json');entries=review['entries']
            assert len(entries)>=50 and len({e['trajectory_id'] for e in entries})==len(entries)
            assert review['review_type']=='manual qualitative review' and not review['clinical_validation']
            for e in entries:
                assert len(e['observation'])>=15 and e['raw_output_sha256']==text_hash(byid[e['trajectory_id']]['raw_output'])
            coverage=read(INDEX/'case_coverage.json')
            required={'all_wrong','all_correct','one_of_four','two_of_four','three_of_four','mixed_unparseable_only','mixed_parsed_wrong','high_semantic_all_wrong','hybrid_variance_all_correct','long_accepted','long_rejected','short_accepted','multi_all_wrong','multi_mixed'}
            assert required<=set(coverage['categories'])
            allgroups={g['group_id']:g for b in sorted((formal/'batches').glob('*')) for g in read(b/'scored.json')['groups']}
            for case in coverage['cases']:
                assert case['group']==allgroups[case['group']['group_id']]
            assert 2<=len(review['frontier_group_ids'])<=5 and all(gid in allgroups for gid in review['frontier_group_ids'])
            decisions={d['group_id']:d for batch in sorted((formal/'batches').glob('*')) for d in read(batch/'commit.json')['decisions']}
            reviewed_groups={e['group_id'] for e in entries}
            assert {0,1,2,3,4}<={decisions[gid]['correct_count'] for gid in reviewed_groups}
            assert {'mixed_parsed_wrong','mixed_unparseable_only','mixed_both'}<={decisions[gid]['mixed_subtype'] for gid in reviewed_groups}
            assert any(len(allgroups[gid]['responses'][0]['ground_truth'])>1 for gid in reviewed_groups)
            for case in read(INDEX/'frontier_case_notes.json')['cases']:
                assert case['group']==allgroups[case['group_id']] and decisions[case['group_id']]['disposition']=='accepted'
                assert case['group_id'] in reviewed_groups
            report=Path('docs/stage_reports/03_dynamic_sampling.md').read_text()
            assert len(report)>4000
            for text in (formal.name,smoke.name,'30秒','2分钟','READY_FOR_STAGE4','限制','256'):assert text in report
            assert 'Stage 3' in Path('docs/implementation/COMPUTE_BUDGET.md').read_text()
            assert 'Stage 3' in Path('docs/RESUME_EVIDENCE.md').read_text()
            readiness=read(INDEX/'readiness.json');assert readiness['READY_FOR_STAGE4']=='YES' and readiness['optimizer_updates']==0
            calibration=read(INDEX/'compute_calibration.json');s=read(formal/'summary.json')
            expected=5000*s['sampling_amplification']*4*s['lengths']['mean']
            assert abs(calibration['stage4_dynamic_output_tokens']-expected)<1e-6
            assert abs(calibration['stage4_dynamic_generation_hours']-expected/s['costs']['output_tokens_per_second']/3600)<1e-9
        gate('real_manual_cases_report_interview_compute_readiness',reports)
        def seal():
            seal=read(INDEX/'final_artifacts.json')
            actual={str(p) for name in selected.values() for p in path_from_id(name).rglob('*') if p.is_file()}
            assert {r['path'] for r in seal}==actual
            for f in seal:assert sha256(f['path'])==f['sha256'],f['path']
        gate('complete_bulk_seal_and_failure_retention',seal)
        def failed_runs():
            failed=path_from_id(selected['failed_smoke']);assert read(failed/'status.json')['status']=='FAILED'
            assert 'run_id' in read(failed/'attempt_002/failure.json')['error']
            recovered=read(INDEX/failed.name/'recovered_cost_summary.json')
            actual=summarize(failed)
            for k,v in actual.items():assert recovered[k]==v,k
            assert actual['generated_groups']==32
            startup=path_from_id(selected['failed_startup']);assert read(startup/'status.json')['status']=='FAILED'
            assert not list((startup/'batches').glob('*/raw.json'))
            assert read(startup/'observations.json')['generated_group_attempts']==0
            for old in (failed,startup):
                manifest=read(old/'manifest.json');assert sha256(old/'config.json')==manifest['config_sha256']
                with zipfile.ZipFile(old/'source.zip') as archive:
                    for name,digest in manifest['source_hashes'].items():assert hashlib.sha256(archive.read(name)).hexdigest()==digest
        gate('negative_smoke_run_raw_costs_retained',failed_runs)
    return dict(stage=3,result='FAIL' if errors else 'PASS',scope='SMOKE_AND_RESUME' if smoke_only else 'FULL',timestamp=now(),gates=gates,errors=errors,counts=counts,
                verifier_sha256=sha256(__file__),contract_sha256=sha256('contracts/stage_budgets.json'),selected_runs=selected)


def path_from_id(run_id):
    return Path(read(INDEX/run_id/'manifest.json')['artifact_root'])
