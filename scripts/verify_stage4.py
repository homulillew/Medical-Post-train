"""Raw Stage4 evidence verifier. Missing full-run artifacts always fail closed."""
from pathlib import Path
import json
import numpy as np
from functools import lru_cache
from medical_posttrain.rl.common import ROOT,INDEX,read,record,sha256,validate,now


def checkpoint(path):
    marker = read(path/'COMMITTED.json')
    for ref in marker['files']:
        p = path/ref['path']
        assert p.is_file() and p.stat().st_size==ref['bytes'] and sha256(p)==ref['sha256']
    for name in ('model','optim','extra_state'):
        assert (path/'native'/f'{name}_world_size_1_rank_0.pt').exists()
    from safetensors import safe_open
    with safe_open(path/'adapter/adapter_model.safetensors',framework='numpy') as f:
        assert len(f.keys())==504
        for k in f.keys():
            assert np.isfinite(f.get_tensor(k)).all()
    return marker


@lru_cache(maxsize=2)
def tokenizer(model):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model,local_files_only=True)


@lru_cache(maxsize=2)
def candidate_pool(path):
    from medical_posttrain.evidence.stage2 import jsonlines
    return {r['prompt_id']:r for r in jsonlines(path)}


def probe_delta(a,b):
    assert len(a['prompt_logprobs'])==len(b['prompt_logprobs'])
    pairs=list(zip(a['prompt_logprobs'][1:],b['prompt_logprobs'][1:]))
    assert pairs and all(x.keys()==y.keys() for x,y in pairs)
    return max(abs(x[k]-y[k]) for x,y in pairs for k in x)


def raw_batch(path,cfg):
    from tokenizers.decoders import DecodeStream
    from medical_posttrain.evidence.stage2 import jsonlines
    from medical_posttrain.data.exam import messages
    from medical_posttrain.reward.hybrid import reward
    from medical_posttrain.reward.parser import reasoning_text
    raw = read(path/'raw.json')['groups']
    scored = read(path/'scored.json')['groups']
    reservation = read(path/'reservation.json')
    encoding = read(path/'semantic_encoding.json')
    vectors = np.load(path/'semantic_vectors.npy')
    assert np.isfinite(vectors).all()
    pool = candidate_pool(cfg['pool']['path'])
    tok = tokenizer(cfg['model'])
    assert len(raw)==len(scored)==len(reservation['encounters'])
    for g,sg,e,pids in zip(raw,scored,reservation['encounters'],reservation['prompt_ids']):
        assert all(g[k]==v for k,v in e.items())
        source = pool[g['prompt_id']]
        assert source['split']=='train'
        expected = tok.apply_chat_template(messages(source),tokenize=True,return_dict=False,
                    add_generation_prompt=True,enable_thinking=True)
        assert pids==expected and len(g['responses'])==len(sg['responses'])==4
        assert len({r['trajectory_id'] for r in g['responses']})==4
        for r,sr in zip(g['responses'],sg['responses']):
            assert all(sr[k]==v for k,v in r.items())
            assert r['question']==source['question'] and r['options']==source['options']
            assert r['ground_truth']==source['answer_set']
            decoder = DecodeStream(ids=pids,skip_special_tokens=True)
            text = ''.join(decoder.step(tok._tokenizer,t) or '' for t in r['token_ids'])
            assert text==r['raw_output'] and r['prompt_token_ids']==pids
            assert len(r['token_ids'])==r['output_tokens']==len(r['rollout_raw_logprobs'])
            assert np.isfinite(r['rollout_raw_logprobs']).all()
            assert r['finish_reason'] in ('stop','length')
            ri,ei = sr['reasoning_embedding_index'],sr['reference_embedding_index']
            assert encoding['texts'][ri]==reasoning_text(text)
            assert encoding['texts'][ei]==source['reference_explanation']
            sem = float(np.clip(vectors[ri]@vectors[ei],0,1)) if encoding['texts'][ri].strip() and encoding['texts'][ei].strip() else 0.
            assert abs(sr['semantic']-sem)<1e-6
            rr = reward(text,source['answer_set'],sem,''.join(source['options']),r['finish_reason'])
            assert sr['acc']==rr['acc'] and sr['format']==rr['format']
            assert sr['parsed_answer']==rr['parser']['answer_set']
            assert abs(sr['semantic_contribution']-rr['semantic_contribution'])<1e-6
            assert abs(sr['total_reward']-rr['score'])<1e-6
    return scored


def update(path,groups,cfg):
    import torch
    from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage,compute_policy_loss_gspo
    from verl.workers.config import ActorConfig
    old = np.load(path/'old.npz')
    frozen = read(path/'old_frozen.json')
    assert record(path/'old.npz')==frozen['artifact']
    rows = [r for g in groups for r in g['responses']]
    assert frozen['trajectory_ids']==[r['trajectory_id'] for r in rows]
    mask = old['mask']
    rewards = np.zeros_like(mask)
    for i,r in enumerate(rows):
        length = len(r['token_ids'])
        assert (mask[i,:length]==1).all() and (mask[i,length:]==0).all()
        rewards[i,length-1]=r['total_reward']
    assert np.array_equal(rewards,old['rewards'])
    adv,_ = compute_grpo_outcome_advantage(torch.tensor(rewards),torch.tensor(mask),np.repeat(np.arange(8),4),epsilon=1e-6)
    # A four-element FP32 reduction's rounding is amplified by division by a
    # small reward std. Check independently in FP64 with a forward-error bound,
    # rather than requiring CPU/GPU reduction trees to agree bitwise.
    for start in range(0,32,4):
        r=rewards[start:start+4].astype(np.float64).sum(1)
        std=r.std(ddof=1)
        reference=(r-r.mean())/(std+1e-6)
        observed=old['advantages'][start:start+4,0]
        if std==0:
            assert np.max(np.abs(observed))<=1e-6
        else:
            tolerance=max(1e-6,8*np.finfo(np.float32).eps*max(abs(r))/(std+1e-6))
            assert np.max(np.abs(reference-observed))<=tolerance
        assert np.allclose(old['advantages'][start:start+4],observed[:,None]*mask[start:start+4],atol=0,rtol=0)
    # Recompute the actual native objective using the now independently checked
    # GPU advantage artifact, not a numerically different CPU normalization.
    adv=torch.tensor(old['advantages'])
    config = ActorConfig(strategy='fsdp2',rollout_n=4,ppo_micro_batch_size_per_gpu=1,
        clip_ratio_low=cfg['clip_ratio_low'],clip_ratio_high=cfg['clip_ratio_high'],
        ppo_mini_batch_size=cfg['mini_prompts'],ppo_epochs=1)
    summary = read(path/'update.json')
    events = [json.loads(line) for line in (path/'events.jsonl').read_text().splitlines()]
    assert frozen['timestamp'] < events[0]['timestamp']
    for j,mini in enumerate(summary['minibatches']):
        evidence = np.load(path/f'mini_{j:02d}.npz')
        idx = evidence['indices']
        assert idx.tolist()==list(range(j*16,(j+1)*16))
        current = evidence['current_logprobs']
        ratios = np.exp(((current-old['old_logprobs'][idx])*mask[idx]).sum(1)/mask[idx].sum(1))
        assert np.allclose(ratios,evidence['ratios'],atol=1e-7,rtol=1e-6)
        losses,clips = [],[]
        for local,i in enumerate(idx):
            loss,metrics = compute_policy_loss_gspo(torch.tensor(old['old_logprobs'][i:i+1]),
                torch.tensor(current[local:local+1]),adv[i:i+1],torch.tensor(mask[i:i+1]),config=config)
            losses.append(loss.item())
            clips.append(metrics['actor/pg_clipfrac'])
        assert np.allclose(losses,evidence['losses'],atol=1e-6)
        assert np.allclose(clips,evidence['clips'],atol=1e-6)
        assert abs(np.mean(clips)-mini['clip_fraction'])<1e-6
        assert abs(np.mean(losses)-mini['policy_loss'])<1e-6
        assert np.isfinite(mini['grad_norm']) and mini['grad_norm']>=0
        if j==0:
            assert np.max(np.abs(ratios-1))<=1e-6
    assert summary['parameter_delta_l2']>0 and summary['initial_trainable_digest']!=summary['final_trainable_digest']
    return summary


def diagnostic(path):
    cfg = read(path/'config.json')
    validate(cfg)
    assert read(path/'manifest.json')['run_class']=='DIAGNOSTIC'
    source = Path(cfg['trajectory_source']['path']).parent if cfg.get('trajectory_source') else path/'trajectories'
    groups = raw_batch(source,cfg)
    assert len(groups)==8
    baseline = None
    summaries = []
    for lr in cfg['diagnostic_lrs']:
        directory = path/f'lr_{lr:g}'
        c = read(directory/'config.json')
        s = update(directory/'update',groups,c)
        a = np.load(directory/'update/old.npz')
        if baseline is None:
            baseline = {k:a[k] for k in a.files}
        assert all(np.array_equal(a[k],v) for k,v in baseline.items())
        assert read(directory/'update/parity.json')['result']=='PASS'
        assert read(directory/'exit.json')['exit_code']==0
        assert checkpoint(directory/'checkpoint')['optimizer_step']==2
        summaries.append(dict(lr=lr,summary=s))
    assert read(path/'summary.json')['status']=='DIAGNOSTIC_PASS'
    return dict(result='PASS',scope='DIAGNOSTIC_ONLY',timestamp=now(),run_id=path.name,
        real_train_groups=8,shared_trajectories=32,conditions=summaries,
        formal_training_groups=0,verifier_sha256=sha256(__file__))


def online_run(path,expected_mode='smoke'):
    from medical_posttrain.rl.online import initial_state
    from medical_posttrain.rl.controller import select_groups,window_metrics,advance
    from medical_posttrain.rl.common import encounter
    from medical_posttrain.sampling.dynamic import Stream
    from medical_posttrain.evidence.stage2 import jsonlines
    cfg=read(path/'config.json')
    initial=validate(cfg)
    assert cfg['mode']==expected_mode
    target={'smoke':32,'pilot':512,'formal':5000}[expected_mode]
    assert cfg['target_training_groups']==target and cfg['groups_per_window']==8
    assert cfg['mini_prompts']==4 and cfg['ppo_epochs']==1
    assert read(path/'manifest.json')['run_class']==expected_mode.upper()
    state=initial_state(initial)
    pool=jsonlines(cfg['pool']['path'])
    stream=Stream([r['prompt_id'] for r in pool],cfg['seed'],cfg['stream_domain'])
    windows=sorted((path/'windows').glob('*'))
    assert len(windows)==target//8
    all_ratios=[]
    all_clips=[]
    previous_marker=None
    previous_probe=None
    attempts={read(a/'resume_start.json')['state']['policy_windows']:a
              for a in sorted(path.glob('attempt_*')) if (a/'resume_start.json').exists()}
    for i,w in enumerate(windows):
        if i in attempts:
            a=attempts[i]
            assert read(a/'resume_start.json')['state']==state
            receipt=read(a/'initial_sync/receipt.json')
            assert receipt['adapter_sha256']==receipt['policy_version']==state['policy_version']
            previous_probe=read(a/'initial_sync/first.json')
            assert probe_delta(previous_probe,read(a/'initial_sync/repeat.json'))<=1e-4
        assert w.name==f'{i:04d}' and read(w/'state_before.json')==state
        groups,decisions,selected=[],[],[]
        for b in sorted((w/'batches').glob('*')):
            batch=raw_batch(b,cfg)
            for g in batch:
                expected=encounter(stream,state['cursor']+len(groups),path.name,state['policy_version'])
                assert all(g[k]==v for k,v in expected.items())
                assert sorted(r['member_index'] for r in g['responses'])==[0,1,2,3]
                for r in g['responses']:
                    assert r['run_id']==path.name and r['config_sha256']==sha256(path/'config.json')
                    assert r['reward_version']==cfg['reward_manifest']['sha256']
                    assert r['adapter_sha256']==r['policy_version']==state['policy_version']
                groups.append(g)
            accepted,dd=select_groups(batch,state['policy_version'],cfg['sampling_mode'],already=len(selected))
            assert read(b/'dispositions.json')==dd
            decisions.extend(dd)
            selected.extend(accepted)
        selection=read(w/'selection.json')
        assert selection['groups']==selected and selection['decisions']==decisions and len(selected)==8
        assert window_metrics(groups,decisions)==read(w/'rollout_metrics.json')
        trained=update(w/'update',selected,cfg)
        assert trained['optimizer_steps_after']==(i+1)*2
        marker=checkpoint(w/'checkpoint')
        assert marker['optimizer_step']==(i+1)*2
        loaded=read(w/'actor/loaded_identity.json')
        assert loaded['trainable_digest']==trained['initial_trainable_digest']
        assert marker['trainable_digest']==trained['final_trainable_digest']
        if previous_marker:
            assert loaded['trainable_digest']==previous_marker['trainable_digest']
            assert loaded['optimizer_digest']==previous_marker['optimizer_digest']
            assert loaded['optimizer_steps']==[i*2]
            assert loaded['scheduler']==previous_marker['scheduler']
            assert read(w/'actor/resume_receipt.json')['result']=='PASS'
        else:
            assert loaded['optimizer_steps']==[]
        previous_marker=marker
        policy=sha256(w/'checkpoint/adapter/adapter_model.safetensors')
        sync=read(w/'sync/receipt.json')
        assert sync['policy_version']==sync['adapter_sha256']==policy
        assert sync['previous_policy_delta']>1e-6 and sync['repeat_error']<=1e-4
        first,repeat=read(w/'sync/first.json'),read(w/'sync/repeat.json')
        assert first['token_ids']==repeat['token_ids']
        assert probe_delta(first,repeat)==sync['repeat_error']
        assert probe_delta(previous_probe,first)==sync['previous_policy_delta']
        previous_probe=first
        assert read(w/'actor_exit.json')['exit_code']==0
        commit=read(w/'commit.json')
        assert commit['state_before']==state
        after=advance(state,groups,decisions,policy,2)
        assert commit['state_after']==after
        for ref in commit['artifacts']:
            assert record(ref['path'])==ref
        state=after
        all_ratios.extend(r for mini in trained['minibatches'] for r in mini['ratios'])
        all_clips.extend(mini['clip_fraction'] for mini in trained['minibatches'])
    assert read(path/'summary.json')['state']==state
    assert read(path/'checkpoint.json')['state']==state
    assert state['training_groups']==target and state['policy_windows']==target//8
    assert max(all_clips)>0 and max(abs(r-1) for r in all_ratios)>1e-6
    real_resume=(path/'physical_resume_receipt.json').exists()
    if expected_mode in ('smoke','pilot') or real_resume:
        resume=read(path/'physical_resume_receipt.json')
        assert resume['result']=='PASS' and resume['new_pid']!=resume['old_pid']
        assert read(path/'termination_observed.json')['dead']
        assert read(path/'pause_ready.json')['state']==read(path/'attempt_002/resume_start.json')['state']
    validations=validation_run(path,cfg) if expected_mode!='smoke' else []
    return dict(result='PASS',scope=expected_mode.upper(),run_id=path.name,sampling_mode=cfg['sampling_mode'],
        timestamp=now(),state=state,real_resume=real_resume,clip_active=True,validation=validations,
        sequence_ratio_min=min(all_ratios),sequence_ratio_max=max(all_ratios),
        average_mini_clip=float(np.mean(all_clips)),verifier_sha256=sha256(__file__))


def validation_run(path,cfg):
    from medical_posttrain.rl.validation import monitor_rows
    from medical_posttrain.rl.actor import array_stats
    from medical_posttrain.reward.parser import parse
    from medical_posttrain.rl.online import initial_state
    from tokenizers.decoders import DecodeStream
    ref=cfg['validation_protocol']
    assert record(ref['path'])==ref
    protocol=read(ref['path'])
    sources=monitor_rows(protocol)
    tok=tokenizer(cfg['model'])
    from medical_posttrain.data.exam import messages
    final=cfg['target_training_groups']//8
    schedule=sorted(set([n for n in protocol['checkpoint_windows'] if n<=final]+[final]))
    assert [p.name for p in sorted((path/'validation').glob('*'))]==[f'{n:04d}' for n in schedule]
    summaries=[]
    for n in schedule:
        directory=path/'validation'/f'{n:04d}'
        state=read(path/'windows'/f'{n-1:04d}'/'commit.json')['state_after'] if n else initial_state(validate(cfg))
        rows=[]
        seconds=0.
        for b in range(32):
            saved=read(directory/f'batch_{b:03d}.json')
            reserved=read(directory/f'reservation_{b:03d}.json')
            ss=sources[b*16:b*16+16]
            assert reserved['prompt_ids']==[r['prompt_id'] for r in ss]
            assert reserved['policy_version']==state['policy_version']
            assert len(saved['predictions'])==16
            for source,r,pids in zip(ss,saved['predictions'],reserved['prompt_token_ids']):
                expected=tok.apply_chat_template(messages(source),tokenize=True,return_dict=False,
                    add_generation_prompt=True,enable_thinking=True)
                assert pids==expected==r['prompt_token_ids']
                assert r['prompt_id']==source['prompt_id'] and r['answer_set']==source['answer_set']
                assert r['policy_version']==state['policy_version']
                decoder=DecodeStream(ids=pids,skip_special_tokens=True)
                text=''.join(decoder.step(tok._tokenizer,t) or '' for t in r['response_ids'])
                assert text==r['output'] and 0<len(r['response_ids'])<=1024
                parsed=parse(text,''.join(source['options']),r['finish_reason'])
                assert r['parsed_answer']==parsed.answer_set and r['strict_format']==parsed.valid_format
                assert r['acc']==int(parsed.answer_set==source['answer_set'])
                assert r['answer_closure']==('<answer>' in text and '</answer>' in text)
                assert r['think_closure']==('<think>' in text and '</think>' in text)
            rows.extend(saved['predictions'])
            seconds+=saved['seconds']
        s=read(directory/'summary.json')
        assert s['n']==len(rows)==512 and not s['test_used'] and not s['selection_used']
        assert s['protocol']==ref and s['policy_version']==state['policy_version']
        for k in ('policy_windows','optimizer_steps','training_groups'):
            assert s[k]==state[k]
        for metric,key in [('accuracy','acc'),('strict_format','strict_format'),('answer_closure','answer_closure'),('think_closure','think_closure')]:
            assert s[metric]==sum(r[key] for r in rows)/512
        assert s['response_length']==array_stats([len(r['response_ids']) for r in rows])
        assert s['validation_output_tokens']==sum(len(r['response_ids']) for r in rows)
        assert s['validation_prompt_tokens']==sum(len(r['prompt_token_ids']) for r in rows)
        assert s['cumulative_generated_output_tokens']==state['output_tokens']
        assert s['cumulative_generated_prompt_tokens']==state['prompt_tokens']
        assert s['cumulative_generated_total_tokens']==state['prompt_tokens']+state['output_tokens']
        assert s['seconds']==seconds
        summaries.append(s)
    return summaries


def verify():
    import traceback
    errors,gates,counts=[],{},{}
    def gate(name,fn):
        try:
            value=fn()
            gates[name]=dict(result='PASS',evidence=value)
            return value
        except Exception as exc:
            gates[name]=dict(result='FAIL',error=repr(exc),traceback=traceback.format_exc())
            errors.append(name+': '+repr(exc))
            return None
    def isolation():
        s=read(ROOT/'project_state.json')
        assert all(s['stages'][str(i)]['status']=='DONE' for i in (1,2,3))
        assert all(s['stages'][str(i)]['status']=='NOT_STARTED' and not s['stages'][str(i)]['run_ids'] for i in (5,6))
        assert read(INDEX/'prerequisite-handoff.json')['result']=='PASS'
        assert sha256(ROOT/'contracts/stage_budgets.json')==s['contract_sha256']
        return dict(predecessors='DONE',stage5_6='NOT_STARTED',contract=s['contract_sha256'])
    gate('isolation_and_contract',isolation)
    def transaction_faults():
        evidence=read(INDEX/'recovery_fault_injections.json')
        required={'before_checkpoint_rename','after_rename_before_pointer','after_optimizer_before_checkpoint'}
        assert set(evidence['scenarios'])==required
        for scenario in evidence['scenarios'].values():
            assert scenario['result']=='PASS' and scenario['real_process_termination']
            assert scenario['old_pid']!=scenario['new_pid']
            assert scenario['committed_budget_continuous'] and scenario['cost_not_rolled_back']
            for ref in scenario['sources']:
                assert record(ref['path'])==ref
        return evidence
    gate('checkpoint_transaction_fault_injections',transaction_faults)
    gate('optimization_diagnostic',lambda:diagnostic(Path(read(INDEX/'active_diagnostic.json')['path'])))
    for v in ('vanilla','dynamic'):
        gate(v+'_smoke',lambda v=v:online_run(Path(read(INDEX/f'active_smoke_{v}.json')['path']),'smoke'))
        gate(v+'_pilot',lambda v=v:online_run(Path(read(INDEX/f'active_pilot_{v}.json')['path']),'pilot'))
    def shared():
        p=read(INDEX/'formal_pair.json')
        ref=p['frozen_config']
        assert record(ref['path'])==ref
        frozen=read(ref['path'])
        assert frozen['mode']=='formal' and frozen['target_training_groups']==5000
        for v in ('vanilla','dynamic'):
            run=Path(p['runs'][v]['path'])
            cfg=read(run/'config.json')
            assert cfg==dict(frozen,sampling_mode=v)
            m=read(run/'manifest.json')
            assert m['run_class']=='FORMAL' and m['git_commit']==p['code_commit']
            assert m['config_sha256']==sha256(run/'config.json')
            assert m['created_at']>p['frozen_at']
            assert m['created_at']>read(INDEX/f'{v}_pilot_verification.json')['timestamp']
        a,b=[read(Path(p['runs'][v]['path'])/'manifest.json') for v in ('vanilla','dynamic')]
        execution=lambda m:{k:v for k,v in m['source_hashes'].items() if k.startswith('src/') or k=='scripts/run_stage4.py'}
        assert execution(a)==execution(b)==p['execution_hashes']
        assert all(sha256(ROOT/k)==v for k,v in p['execution_hashes'].items())
        first=[]
        for v in ('vanilla','dynamic'):
            run=Path(p['runs'][v]['path'])
            first.append(read(run/'windows/0000/actor/loaded_identity.json'))
        assert first[0]['trainable_digest']==first[1]['trainable_digest']
        assert first[0]['optimizer_steps']==first[1]['optimizer_steps']==[]
        return p
    pair=gate('frozen_shared_formal_pair',shared)
    if pair:
        for v in ('vanilla','dynamic'):
            r=gate(v+'_formal_raw',lambda v=v:online_run(Path(pair['runs'][v]['path']),'formal'))
            if r:
                counts[v]=dict(training_groups=r['state']['training_groups'],training_trajectories=4*r['state']['training_groups'],
                    policy_windows=r['state']['policy_windows'],optimizer_steps=r['state']['optimizer_steps'],
                    generated_groups=r['state']['generated_groups'],output_tokens=r['state']['output_tokens'],
                    prompt_tokens=r['state']['prompt_tokens'])
            def final_reload(v=v):
                run=Path(pair['runs'][v]['path'])
                receipt=read(run/'final_reload/result.json')
                marker=checkpoint(run/'windows/0624/checkpoint')
                assert receipt['result']=='PASS'
                assert receipt['loaded_identity']['trainable_digest']==marker['trainable_digest']
                assert receipt['loaded_identity']['optimizer_digest']==marker['optimizer_digest']
                assert receipt['loaded_identity']['optimizer_steps']==[1250]
                assert read(run/'final_reload/exit.json')['exit_code']==0
                assert receipt['finite_logprobs'] and receipt['response_tokens']>0
                return record(run/'final_reload/result.json')
            gate(v+'_final_native_reload',final_reload)
    def deliverables():
        from analyze_stage4 import analyze
        manifest=read(INDEX/'deliverables.json')
        assert manifest['READY_FOR_STAGE5']=='YES'
        required=('stage_report','interview_story','cases','resume_evidence','plots','checkpoint_index','pilot_review')
        for key in required:
            refs=manifest[key] if isinstance(manifest[key],list) else [manifest[key]]
            assert refs
            for ref in refs:
                assert record(ref['path'])==ref
        report=Path(manifest['stage_report']['path']).read_text()
        assert all(pair['runs'][v]['run_id'] in report for v in ('vanilla','dynamic'))
        reviewed=read(manifest['cases']['path'])
        assert len(reviewed['manual_reviews'])>=2
        for case in reviewed['manual_reviews']:
            assert case['read_in_full'] and case['observation']
            for ref in case['sources']:
                assert record(ref['path'])==ref
        index=read(manifest['checkpoint_index']['path'])
        for v in ('vanilla','dynamic'):
            result=analyze(Path(pair['runs'][v]['path']))
            assert len(result['windows'])==625 and result['formal_completed']
            assert index[v]['final']['policy_windows']==625
            for item in index[v]['validation_checkpoints']:
                assert record(item['adapter']['path'])==item['adapter']
        return manifest
    if pair:
        gate('reports_cases_curves_readiness',deliverables)
    else:
        errors.append('Reports/readiness cannot pass without a frozen formal pair')
    return dict(stage=4,result='FAIL' if errors else 'PASS',scope='FULL',timestamp=now(),errors=errors,
        gates=gates,counts=counts,READY_FOR_STAGE5='NO' if errors else 'YES',
        contract_sha256=sha256(ROOT/'contracts/stage_budgets.json'),verifier_sha256=sha256(__file__))
