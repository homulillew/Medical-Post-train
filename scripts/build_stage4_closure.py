#!/usr/bin/env python3
"""Build a review/navigation pack and a frozen selection protocol from saved evidence.

CPU/file operations only. No training, generation, test-content reads or manual review.
All prospective artifacts are built in memory before any exclusive output creation.
"""
import ast
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from checkpoint_selection import GROUPS, reference, read, require_stage4_gate
from medical_posttrain.reward.parser import parse
from medical_posttrain.sampling.dynamic import Stream

S4 = ROOT / 'experiments/stage4'
S5 = ROOT / 'experiments/stage5'
OUTPUTS = {}
CACHE = {}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode()


def add(path, value):
    path = Path(path).resolve()
    assert not path.exists(), f'Immutable output already exists: {path}'
    assert path not in OUTPUTS
    OUTPUTS[path] = value.encode() if isinstance(value, str) else encoded(value)
    return ref(path)


def ref(path):
    path = Path(path).resolve()
    if path in OUTPUTS:
        b = OUTPUTS[path]
        return dict(path=str(path), sha256=hashlib.sha256(b).hexdigest(), bytes=len(b))
    if path not in CACHE:
        CACHE[path] = reference(path)
    return CACHE[path]


def checked(r):
    assert ref(r['path']) == r, r['path']
    return r


def response(row, question, source, identity, advantage=None):
    text = row.get('raw_output', row.get('output'))
    parsed = parse(text, ''.join(question['options']), row['finish_reason'])
    truth = question.get('answer_set', question.get('ground_truth'))
    correct = int(parsed.answer_set == truth)
    if 'acc' in row:
        assert correct == row['acc']
    # Keep every emitted character, including malformed/control regions. These
    # are saved local model outputs, not a claim about proprietary hidden thought.
    return dict(visible_response=text, raw_response_verbatim=True,
                parsed=parsed.to_dict(), correctness=correct,
                output_length_tokens=len(row.get('token_ids', row.get('response_ids'))),
                finish_reason=row['finish_reason'], member_index=row.get('member_index'),
                trajectory_id=row.get('trajectory_id'), raw_source=ref(source),
                identity=identity, reward_components={k: row[k] for k in
                    ['acc', 'semantic', 'semantic_cosine', 'semantic_contribution', 'format', 'total_reward'] if k in row},
                advantage=advantage,
                advantage_status='RETAINED_NATIVE_SCALAR' if advantage is not None else 'NOT_APPLICABLE_OR_NOT_RETAINED')


def case_base(number, category, question, split):
    pid = question['prompt_id']
    assert pid.startswith(('cmexam_val:', 'cmexam_train:'))
    return dict(case_id=f'S4-MR-{number:02d}', category=category, prompt_id=pid,
                source_split=split, question=question['question'], options=question['options'],
                ground_truth=question.get('answer_set', question.get('ground_truth')),
                models={}, human_reviewed=False, read_in_full=False,
                reviewer=None, manual_observation=None, review_status='PENDING_HUMAN_REVIEW')


def packets(pair, aux):
    out = Path(aux['artifact_path'])
    questions = {r['prompt_id']: r for r in read(out / 'ablation_eval_dataset.json')}
    predictions = {}
    policies = read(out / 'ablation_eval/actual_policies.json')
    for model in ['sft', 'vanilla', 'dynamic', 'random3x']:
        predictions[model] = {}
        for f in sorted((out / 'ablation_eval' / model).glob('batch_*.json')):
            if 'reservation' not in f.name:
                for r in read(f)['predictions']:
                    predictions[model][r['prompt_id']] = (r, f)
    rules = [
        ('sft_wrong_vanilla_correct_dynamic_correct', lambda s,v,d,r: not s and v and d),
        ('sft_wrong_vanilla_wrong_dynamic_correct', lambda s,v,d,r: not s and not v and d),
        ('vanilla_correct_dynamic_wrong', lambda s,v,d,r: v and not d),
        ('dynamic_correct_random3x_wrong', lambda s,v,d,r: d and not r),
        ('random3x_correct_dynamic_wrong', lambda s,v,d,r: r and not d),
    ]
    cases = []
    used = set()
    for category, predicate in rules:
        ids = [pid for pid in sorted(questions) if pid not in used and predicate(
            *[predictions[m][pid][0]['acc'] for m in ['sft','vanilla','dynamic','random3x']])]
        assert ids, category
        pid = ids[0]; used.add(pid)
        case = case_base(len(cases)+1, category, questions[pid], 'cmexam_val/ablation_eval_512_v1')
        for model in predictions:
            row, source = predictions[model][pid]
            identity = dict(**policies[model], training_groups=0 if model=='sft' else 512,
                            run_id=aux['run_id'] if model=='random3x' else
                            read(ROOT/'experiments/stage1/initialization_manifest.json')['run_id'] if model=='sft' else pair['runs'][model]['run_id'])
            case['models'][model] = dict(available=True, group_classification='N1_NOT_G4',
                responses=[response(row, questions[pid], source, identity)])
        case['question_source'] = ref(out / 'ablation_eval_dataset.json')
        cases.append(case)
    front = S4 / 'frontier_diagnostic_v1'
    fp = read(front / 'preregistration.json')
    manifest = read(front / 'manifest.json'); checked(manifest['dataset'])
    rows = read(manifest['dataset']['path'])
    order = Stream([r['prompt_id'] for r in rows], fp['seed'], fp['stream_domain']).order(0)
    metrics = {m: {r['prompt_id']:r for r in read(front/f'{m}_prompt_metrics.json')}
               for m in ['sft','vanilla','dynamic']}
    rules = [
        ('sft_g4_all_wrong_hard_tail_proxy_dynamic_gain', lambda s,v,d: s['all_wrong'] and d['correct_count']-v['correct_count']>=2),
        ('sft_mixed_dynamic_not_better', lambda s,v,d: s['mixed'] and d['correct_count']<=v['correct_count']),
        ('parsed_wrong_mixed', lambda s,v,d: s['classification']=='mixed_parsed_wrong'),
        ('unparseable_only_mixed', lambda s,v,d: s['classification']=='mixed_unparseable_only'),
    ]
    for category, predicate in rules:
        ids = [pid for pid in sorted(order) if pid not in used and predicate(*[metrics[m][pid] for m in metrics])]
        assert ids, category
        pid = ids[0]; used.add(pid); batch = order.index(pid)//fp['batch_groups']
        model_data = {}
        for model in metrics:
            f = Path(fp['artifact_path'])/model/'batches'/f'{batch:04d}'/'raw.json'
            group = next(g for g in read(f)['groups'] if g['prompt_id']==pid)
            assert len(group['responses'])==4
            q = dict(group['responses'][0], prompt_id=pid)
            identity = dict(**fp['policies'][model], evaluation_run_id=fp['run_id'],
                training_groups=0 if model=='sft' else 5000)
            model_data[model] = dict(available=True, group_classification=metrics[model][pid]['classification'],
                group_id=group['group_id'], correct_count=metrics[model][pid]['correct_count'],
                responses=[response(r,q,f,identity) for r in group['responses']])
        model_data['random3x'] = dict(available=False, reason='Random3x was not evaluated on Frontier; no new inference.')
        case = case_base(len(cases)+1, category, q, 'cmexam_val/Frontier1000')
        case['models'] = model_data; case['question_source'] = manifest['dataset']
        cases.append(case)
    root = Path(pair['runs']['dynamic']['path'])
    wanted = {'rejected_all_correct','rejected_all_wrong','overflow_eligible','selected'}
    found = {}
    for w in sorted((root/'windows').iterdir()):
        for b in sorted((w/'batches').iterdir()):
            for decision in read(b/'dispositions.json'):
                disposition = decision['disposition']
                if disposition not in wanted or disposition in found:
                    continue
                group = next(g for g in read(b/'scored.json')['groups'] if g['group_id']==decision['group_id'])
                # A concrete high-cost example: at least 1024 output tokens
                # discarded in this group. First qualifying group, not a peak claim.
                if disposition=='rejected_all_correct' and sum(r['output_tokens'] for r in group['responses'])<1024:
                    continue
                found[disposition] = (w,b,decision,group)
        if set(found)==wanted:
            break
    assert set(found)==wanted
    for disposition in ['rejected_all_correct','rejected_all_wrong','overflow_eligible','selected']:
        w,b,decision,group = found[disposition]
        q = dict(group['responses'][0], prompt_id=group['prompt_id'])
        case = case_base(len(cases)+1, disposition, q, 'cmexam_train/formal_RL_pool')
        state = read(w/'state_before.json')
        count = state['training_groups']
        adapter = Path(aux['evaluation_policies']['sft']['adapter_path'])/'adapter_model.safetensors' if not count else root/'windows'/f'{int(w.name)-1:04d}'/'checkpoint/adapter/adapter_model.safetensors'
        identity = dict(run_id=root.name, window=w.name, training_groups_before=count,
                        policy_version=state['policy_version'], adapter=ref(adapter))
        assert identity['adapter']['sha256']==state['policy_version']
        advantages = [None]*4
        extra = [ref(b/'scored.json'), ref(b/'dispositions.json'), ref(w/'state_before.json')]
        if disposition=='selected':
            selection = read(w/'selection.json')['groups']
            offset = next(i for i,g in enumerate(selection) if g['group_id']==group['group_id'])*4
            with np.load(w/'update/old.npz') as arrays:
                advantages = arrays['advantages'][offset:offset+4,0].astype(float).tolist()
            extra += [ref(w/'selection.json'),ref(w/'update/old.npz')]
        case['models']['dynamic'] = dict(available=True, disposition=disposition,
            group_classification=decision['classification'], mixed_subtype=decision['mixed_subtype'],
            group_id=group['group_id'], output_tokens=sum(r['output_tokens'] for r in group['responses']),
            responses=[response(r,q,b/'raw.json',identity,a) for r,a in zip(group['responses'],advantages)])
        for m in ['sft','vanilla','random3x']:
            case['models'][m] = dict(available=False, reason='No matched response bundle for this training encounter; no generation or substitution.')
        case['training_disposition'] = disposition; case['sources'] = extra
        case['question_source'] = ref(b/'raw.json')
        cases.append(case)
    return dict(status='WAITING_FOR_HUMAN_NARRATIVE_AND_MANUAL_REVIEW', human_reviewed=False,
        selection_rule='First unused lexicographic prompt ID satisfying each eval/Frontier predicate; first chronological qualifying training group. All-correct discard >=1024 output tokens. Illustrative cases, not prevalence estimates.',
        count=len(cases), cases=cases,
        review_instructions=['Read every response for chosen cases; record reviewer identity, date, full-read confirmation, observation and exact source refs in a separate genuine manual review artifact.',
            'Machine categories and correctness are parser-derived observations, not clinical validation or human endorsement.',
            'Frontier cases include all four responses for each of three policies; auxiliary cases include all four checkpoints at n=1.',
            'Training cases expose only retained matched encounter evidence. Missing model responses are explicitly unavailable.'],
        recommended_case_ids=['S4-MR-04','S4-MR-05','S4-MR-06','S4-MR-07','S4-MR-10','S4-MR-13'])


def render_packet(packet):
    lines = ['# Stage4 manual review packet — raw evidence', '',
             'Status: WAITING_FOR_HUMAN_NARRATIVE_AND_MANUAL_REVIEW. No human review has been recorded.', '',
             'Full emitted responses are reproduced verbatim in text blocks. Missing matched outputs remain unavailable.', '']
    for c in packet['cases']:
        lines += [f"## {c['case_id']} — {c['category']}", '', f"Prompt: `{c['prompt_id']}`; split: `{c['source_split']}`", '',
                  c['question'], '', '```json', json.dumps(c['options'],ensure_ascii=False,indent=2),'```','',f"Ground truth: `{c['ground_truth']}`",'']
        for model,data in c['models'].items():
            lines += [f'### {model}', '']
            if not data['available']:
                lines += [data['reason'], '']; continue
            lines += [f"Group: `{data['group_classification']}`; disposition: `{data.get('disposition','not a training rollout')}`",'']
            for i,r in enumerate(data['responses']):
                lines += [f"Response {i+1}: correct={r['correctness']}; tokens={r['output_length_tokens']}; parser={json.dumps(r['parsed'],ensure_ascii=False)}",'',
                    '```text',r['visible_response'],'```','',
                    'Identity: '+json.dumps(r['identity'],ensure_ascii=False),'',
                    'Reward: '+json.dumps(r['reward_components'],ensure_ascii=False)+f"; advantage: {r['advantage']}",'',
                    'Raw source: '+json.dumps(r['raw_source'],ensure_ascii=False),'']
    return '\n'.join(lines).rstrip()+'\n'


def deliverables_schema():
    source = ROOT/'scripts/verify_stage4.py'
    text = source.read_text(); tree = ast.parse(text)
    function = next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='deliverables')
    required = ast.literal_eval(next(n.value for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='required' for t in n.targets)))
    old = read(S4/'stage4_acceptance_readiness_v1.json')
    assert list(required)==old['unmet_gate']['required_manifest_keys']
    return dict(status='SCHEMA_ONLY_NOT_DELIVERABLES', verifier=ref(source),
        source_lines=[function.lineno,function.end_lineno], exact_code=ast.get_source_segment(text,function),
        required_path=str(S4/'deliverables.json'), required_keys=['READY_FOR_STAGE5',*required],
        reference_format={'path':'existing file path; absolute paths match repository record()', 'sha256':'64 lowercase hex SHA256 of complete bytes', 'bytes':'integer byte length'},
        reference_validation='record(ref.path) == ref; extra reference keys fail equality. Nonempty list or single reference permitted in initial loop.',
        effective_types={'stage_report':'single ref (subsequently indexed directly)', 'cases':'single JSON ref', 'checkpoint_index':'single JSON ref',
                         'interview_story':'single ref or nonempty ref list', 'resume_evidence':'single ref or nonempty ref list',
                         'plots':'single ref or nonempty ref list', 'pilot_review':'single ref or nonempty ref list'},
        manual_review_structure={'manual_reviews':[{'read_in_full':'truthy; genuine full-response reading required by owner',
            'observation':'truthy; actual reviewer observation, not machine-authored completion',
            'sources':'list of exact record() references, each checked'}]}, minimum_manual_reviews=2,
        machine_check_limits=['Current code does not require a nonempty sources list or reviewer identity/date; owner requires genuine attributable review, so include these.',
            'Current code hash-checks interview_story, plots, resume_evidence and pilot_review here; it does not semantically assess their narrative content. Contract requirements still apply.'],
        report_constraints={'fixed_report_path':False,'must_contain':'Both formal run IDs as substrings','suggested_path':str(ROOT/'docs/stage_reports/04_gspo.md')},
        checkpoint_index_constraints={'variants':['vanilla','dynamic'],'final.policy_windows':625,
            'validation_checkpoints':'Each item.adapter exact reference verified',
            'additional_computation':'analyze(run) must return 625 windows and formal_completed true for both runs'},
        existing_reference_candidates={k:ref(S4/f) for k,f in {
            'plots':'formal_plots/manifest.json','checkpoint_index':'formal_checkpoint_index.json',
            'pilot_review':'pilot_review.json','resume_evidence':'vanilla_formal_resume_verified_002.json'}.items()},
        readiness_semantics='Manifest READY_FOR_STAGE5 must equal YES; full verifier emits YES only with no failed gates. Pack/schema preparation is not a full PASS.',
        consistent_with_existing_readiness=True, full_verifier_rerun=False,
        blockers=['Final stage report','Final interview story','At least two genuine full-response manual reviews','Completed deliverables.json'])


def recovery(pair):
    incident = read(S4/'vanilla_formal_incident_002.json')
    adopted = read(S4/'vanilla_formal_resume_verified_002.json')
    fresh = read(S4/'vanilla_formal_fresh_guard_verified_002.json')
    root = Path(pair['runs']['vanilla']['path']); pending = root/'windows/0402'
    update = read(pending/'update/update.json'); commit = read(pending/'commit.json')
    assert adopted['optimizer_steps_reexecuted']==0 and adopted['result']==fresh['result']=='PASS'
    keep = lambda s:{k:v for k,v in s.items() if not isinstance(v,dict)}
    return dict(case_id='S4-SYS-01', human_reviewed=False, category='GPU_release_checkpoint_adoption',
        run_id=root.name, failure_window='0402', failure=read(incident['original_failure']['path']),
        failure_boundary='Actor exited successfully after optimizer/checkpoint completion; GPU-release assertion preceded controller synchronization/commit.',
        optimizer_work_completed=True, optimizer_steps_completed=[m['optimizer_step'] for m in update['minibatches']],
        checkpoint_complete=True, controller_commit_complete_at_failure=False,
        before=keep(commit['state_before']), after_adoption=keep(commit['state_after']),
        replay_risk='Repeating optimizer steps 805/806 would duplicate already durable updates and violate the original update budget/lineage.',
        adoption_basis='Complete checkpoint marker and native optimizer/scheduler/RNG identities; reload added zero steps; sync then controller commit.',
        root_cause_limit=incident['root_cause_assessment'], adoption=adopted, next_fresh_window=fresh,
        sources=[ref(S4/f) for f in ['vanilla_formal_incident_002.json','vanilla_formal_recovery_002.json',
            'vanilla_formal_resume_verified_002.json','vanilla_formal_fresh_guard_verified_002.json',
            'recovery_fault_injections.json','fault_a_replay_numerics.json']]+[
                checked(incident['original_failure']),ref(pending/'checkpoint/COMMITTED.json'),ref(pending/'update/update.json'),
                ref(pending/'actor_exit.json'),ref(ROOT/'docs/implementation/STAGE4_GPU_RELEASE_RECOVERY.md')],
        independent_second_case={'case_id':'S4-SYS-02','category':'Random3x_real_fresh_process_resume',
            'receipt':ref(S5/'aux_random3x_resume_v1.json'),'facts':read(S5/'aux_random3x_resume_v1.json')},
        limit='Successful recovery does not establish bitwise GPU backward replay or eliminate all possible future failures.')


def selection_protocol(pair, aux, env):
    vp = read(S4/'formal_validation_protocol.json'); checked(vp['partition'])
    part = read(vp['partition']['path']); ids = part['selection']
    assert len(ids)==len(set(ids))==1024 and not set(ids)&set(part['monitor'])
    assert all(pid.startswith('cmexam_val:') for pid in ids)
    clusters = [part['cluster_ids'][pid] for pid in ids]; assert len(set(clusters))==1024
    index = read(S4/'formal_checkpoint_index.json')
    candidates = {}
    for variant in ['vanilla','dynamic']:
        by_count = {x['training_groups']:x for x in index[variant]['validation_checkpoints']}
        candidates[variant] = []
        for count in GROUPS:
            item = by_count[count]; checked(item['adapter'])
            candidates[variant].append(dict(checkpoint_id=f'{variant}:{count:04d}',
                run_id=pair['runs'][variant]['run_id'], training_groups=count,
                policy_windows=item['policy_windows'], optimizer_steps=item['optimizer_steps'],
                adapter=item['adapter'], index_source=ref(S4/'formal_checkpoint_index.json')))
    init = read(ROOT/'experiments/stage1/initialization_manifest.json')
    model = Path(aux['config']['model'])
    template = model/'chat_template.jinja'
    if not template.exists(): template = model/'tokenizer_config.json'
    try: require_stage4_gate()
    except PermissionError as exc: blocked = str(exc)
    else: raise AssertionError('Expected Stage4 gate to remain closed')
    return dict(status='FROZEN_BEFORE_SELECTION_GENERATION', scope='PREPARATION_ONLY',
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        execution_allowed_only_if_stage4_ready_for_stage5='YES',
        execution_gate={'stage4_status':'DONE','full_verifier_result':'PASS','full_verifier_scope':'FULL',
                        'READY_FOR_STAGE5':'YES','enforcement_source':ref(ROOT/'scripts/checkpoint_selection.py')},
        current_gate={'allowed':False,'reason':blocked}, generation_performed=False,
        dataset={'split':'cmexam_val_selection1024','n':1024,'prompt_ids':ids,'cluster_ids':clusters,
                 'partition':vp['partition'],'source_metadata':vp['source'],
                 'content_access_this_task':'IDs/cluster metadata only; no selection question/answer content or inference'},
        fixed_sft={'run_id':init['run_id'],'selection':False,'adapter':ref(Path(init['adapter_path'])/'adapter_model.safetensors')},
        candidates=candidates, candidate_training_groups=GROUPS,
        decoding={'n':1,'temperature':0.0,'top_p':1.0,'top_k':-1,'max_tokens':1024,'seed':42},
        prompt={'messages_source':ref(ROOT/'src/medical_posttrain/data/exam.py'),
                'chat_template':ref(template),'rollout_source':ref(ROOT/'src/medical_posttrain/rl/rollout.py'),
                'add_generation_prompt':True,'enable_thinking':True},
        parser=ref(ROOT/'src/medical_posttrain/reward/parser.py'),
        runtime={'versions':{k:env['packages'][k] for k in ['torch','transformers','peft','verl','vllm','tokenizers']},
                 'engine':aux['config']['engine'], 'environment':{'VLLM_BATCH_INVARIANT':'1','VLLM_USE_FLASHINFER_SAMPLER':'0',
                 'VLLM_USE_V2_MODEL_RUNNER':'0','VLLM_WORKER_MULTIPROC_METHOD':'spawn'},'batch_size_prompts':16},
        selection_rule=['highest exact/canonical correct count / 1024','lowest unparseable count / 1024',
                        'lowest truncation count / 1024','fewest training groups','lexicographically smallest checkpoint_id'],
        variant_selection='Independent ranking within Vanilla and Dynamic; all 10 candidates must finish before ranking.',
        invalid_answer_policy='Valid wrong answers and parser failures count as outcomes, never resampled. Unparseable counts as incorrect.',
        technical_failure_policy='Zero automatic retries. Retain reservation/error/cost; fail closed on missing/incomplete output. No candidate may be selected from a partial run.',
        excluded_selection_inputs=['response_length','test_scores','CMB_scores','Frontier_scores','Random3x_results','external_DeepSeek_scores'],
        external_tools=False, search=False, paid_API=False,
        scientific_endpoints={v:candidates[v][-1] for v in candidates},
        future_reporting=['SFT + validation-selected Vanilla/Dynamic deployment-style primary comparison',
                          'Retain Vanilla5000 vs Dynamic5000 equal-update scientific endpoint comparison even if selection chooses earlier checkpoints'],
        no_results=True, freeze_policy='Exclusive creation; commit before any selection generation. Any amendment requires a new version and explicit record.',
        prerequisite_note='Protocol is not an inference launcher. Future launchers must invoke require_stage4_gate and validate_selection_input before loading selection contents or generating.')


def main():
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    remote = subprocess.check_output(['git','rev-parse','origin/main'],cwd=ROOT,text=True).strip()
    assert head==remote, 'Synchronize remote main before closure'
    dirty = subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
    tracked = subprocess.check_output(['git','ls-files','experiments','src','configs','contracts','project_state.json','README.md'],cwd=ROOT,text=True).splitlines()
    protected = [ref(ROOT/f) for f in tracked if (ROOT/f).is_file()]
    pair = read(S4/'formal_pair.json'); aux = read(S5/'aux_random3x_protocol_v2.json')
    state = read(ROOT/'project_state.json'); assert state==aux['project_state_snapshot']
    assert state['stages']['4']['status']=='FULL_PASS'
    passed = {}
    for name, path in {
        'vanilla_raw':S4/'vanilla_formal_verification.json','dynamic_raw':S4/'dynamic_formal_verification.json',
        'frontier':S4/'frontier_diagnostic_v1/verification.json','signal':S4/'signal_density_verification_v1.json',
        'random3x_raw':S5/'aux_random3x_verification_v1.json','random3x_resume':S5/'aux_random3x_resume_v1.json',
        'random3x_eval':S5/'aux_random3x_eval_verification_v1.json',
        **{v+'_reload':Path(r['path'])/'final_reload/result.json' for v,r in pair['runs'].items()}
    }.items():
        assert read(path)['result']=='PASS', name
        passed[name] = ref(path)
    baseline = dict(timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),local_HEAD=head,remote_main=remote,
        git_status_at_build=dirty,project_state=ref(ROOT/'project_state.json'),stage_states={k:v['status'] for k,v in state['stages'].items()},
        passed_receipts=passed,random3x_status=read(S5/'aux_random3x_status.json'),
        expected_state_differences=[],preexisting_uncommitted_operational_files=[
            'experiments/stage5/aux_random3x_status.json','experiments/stage5/aux_random3x_publication.json',
            'experiments/stage4/frontier_diagnostic_v1/publication_status.json'],
        protected_existing_files=protected,scope='Offline evidence packaging; no model or API execution')
    audit_ref = add(S4/'closure_start_audit_v1.json',baseline)
    packet = packets(pair,aux)
    packet_ref = add(S4/'manual_review_packet_v1.json',packet)
    add(S4/'manual_review_packet_v1.md',render_packet(packet))
    recovery_ref = add(S4/'recovery_interview_case_v1.json',recovery(pair))
    schema_ref = add(S4/'stage4_deliverables_schema_v1.json',deliverables_schema())
    env_path = Path(pair['runs']['dynamic']['path'])/'attempt_001/environment.json'
    env = read(env_path)
    protocol = selection_protocol(pair,aux,env)
    protocol_ref = add(S5/'checkpoint_selection_protocol_v1.json',protocol)
    post = read(S4/'postformal_analysis_v1.json'); signal = read(S4/'signal_density_analysis_v1.json')
    frontier = read(S4/'frontier_diagnostic_v1/analysis.json'); init = read(ROOT/'experiments/stage1/initialization_manifest.json')
    cfg = aux['config']
    vc = read(Path(pair['runs']['vanilla']['path'])/'config.json'); dc = read(Path(pair['runs']['dynamic']['path'])/'config.json')
    diffs = {k:{'vanilla':vc.get(k),'dynamic':dc.get(k)} for k in set(vc)|set(dc) if vc.get(k)!=dc.get(k)}
    assert set(diffs)=={'sampling_mode'}, diffs
    handoff = dict(status='WAITING_FOR_HUMAN_NARRATIVE_AND_MANUAL_REVIEW',source_HEAD=head,
        stage4_state='FULL_PASS',stage5_state='NOT_STARTED',READY_FOR_STAGE5='NO',start_audit=audit_ref,
        runtime_model={'model':'Qwen/Qwen3-8B','base_revision':init['base_revision'],
            'tokenizer_revision':init['tokenizer_revision'],'GPU':env['gpu'],'runtime_versions':protocol['runtime']['versions'],
            'environment_source':ref(env_path),'LoRA':{k:init[k] for k in ['lora_rank','lora_alpha','lora_targets','dtype']}},
        sft_initialization={'manifest':ref(ROOT/'experiments/stage1/initialization_manifest.json'),'run_id':init['run_id'],'adapter_sha256':init['adapter_sha256']},
        GSPO={'advantage_estimator':'Native GRPO-style outcome advantage; per-group centering/sample-std normalization',
            'policy_loss':'Native GSPO sequence-level importance ratio and clipping','old_logprobs':'Actor-computed and frozen for all 32 trajectories before either optimizer minibatch',
            'critic':False,'permanent_reference_worker':False,'KL':False,'G':4,'groups_per_window':8,'minibatches_per_window':2,
            'learning_rate':cfg['learning_rate'],'clip_low':cfg['clip_ratio_low'],'clip_high':cfg['clip_ratio_high'],
            'grad_clip':cfg['grad_clip'],'ppo_epochs':cfg['ppo_epochs'],'entropy_coefficient':cfg['entropy_coefficient'],
            'source':ref(ROOT/'src/medical_posttrain/rl/actor.py')},
        formal_pair={'manifest':ref(S4/'formal_pair.json'),'runs':pair['runs'],'config_differences':diffs,
            'config_sources':{v:ref(Path(r['path'])/'config.json') for v,r in pair['runs'].items()},
            'intended_difference':pair['intended_difference'],'known_confounders':pair['known_confounders'],
            'budget':pair['budget_per_variant'],'final_metrics':post['accuracy_vs_update_budget'][-1],
            'equal_update_curve':post['accuracy_vs_update_budget'],'equal_token_curve':post['accuracy_vs_shared_rollout_tokens'],
            'observations':post['observations'],'cost_accounting':post['cost_decomposition'],
            'run_statistics':post['runs'],'source':ref(S4/'postformal_analysis_v1.json'),'receipts':passed},
        signal_density={'source':ref(S4/'signal_density_analysis_v1.json'),
            'runs':{k:{field:v[field] for field in ['densities','by_type']} for k,v in signal['runs'].items()},
            'effective_signal_amplification':signal['effective_signal_amplification'],'limits':signal['limits']},
        frontier={'source':ref(S4/'frontier_diagnostic_v1/analysis.json'),
            'results':frontier,'H1_status':'NOT_SUPPORTED','H2_status':'No established increase or noninferiority; signed mixed-to-all-wrong CI crosses zero.',
            'H3_status':'Mixed subtypes retained separately; no general resolution superiority established.',
            'H4_status':'Both RL policies shift SFT mixed toward all-correct; extra Dynamic point-estimate gains concentrate in SFT-G4-all-wrong hard-tail proxy.',
            'hard_tail_proxy_label':'SFT-G4-all-wrong hard-tail proxy','absolute_difficulty_claim':False},
        random3x={'preregistration':ref(S5/'aux_random3x_protocol_v2.json'),'schedule':aux['schedule'],
            'selection_rule':aux['selection_rule'],'schedule_matched':True,'exact_token_matched':False,
            'raw_verifier':passed['random3x_raw'],'resume':passed['random3x_resume'],
            'training_comparison':read(S5/'aux_random3x_training_comparison_v1.json'),
            'training_source':ref(S5/'aux_random3x_training_comparison_v1.json'),
            'evaluation':read(S5/'aux_random3x_eval_analysis_v1.json'),'evaluation_source':ref(S5/'aux_random3x_eval_analysis_v1.json'),
            'mechanism_status':'INCONCLUSIVE'},
        engineering_cases={'batch_invariance':{'sources':[ref(ROOT/'docs/stage_reports/00_runtime_compatibility.md'),ref(ROOT/'src/medical_posttrain/rl/rollout.py')],
            'facts':'Native sampling, VLLM_BATCH_INVARIANT=1 and LoRA shrink split_k=1 retained; cross-process sampled responses are not guaranteed bitwise identical.'},
            'clipping_LR_diagnostic':{'sources':[ref(S4/'diagnostic_verification.json'),ref(S4/'diagnostic_manual_case.json'),ref(ROOT/'docs/implementation/STAGE4_DECISIONS.md')],
            'facts':'Same retained batch tested at 1e-5/3e-6/1e-6; adopted common LR1e-6. Objective clipping differs from any-bound exceedance.'},
            'GPU_release_and_native_resume':recovery_ref},
        supported_claims=['Dynamic leads all preregistered positive equal-update monitor milestones in this single formal pair.',
            'Correctness-discriminative selected-group density is 0.4102 Vanilla and 1.0 Dynamic; amplification is about 2.4378, not gradient quality.',
            'Vanilla nonmixed groups can carry semantic/format shaping advantages.',
            'Random3x point estimate is consistent with correctness-aware selection rather than exposure-only explanation, but paired intervals leave the mechanism inconclusive.',
            'Formal, Frontier and auxiliary estimates are retained independently; all positive/negative comparisons are preserved.'],
        unsupported_claims=['Dynamic final superiority statistically established','Dynamic globally compute-efficient',
            'Dynamic saves rollout compute','Random3x proves causal selection effect','GSPO experimentally superior to GRPO',
            'Frontier H1 supported','clinical validity','final-test superiority','Nonmixed equals zero gradient','Nonsignificance establishes equivalence'],
        manual_review_packet=packet_ref,recovery_case=recovery_ref,deliverables_schema=schema_ref,
        selection_protocol=protocol_ref,selection1024_untouched=True,final_tests_untouched=True,paid_API_new_calls=0,
        README_status='README_STATUS_STALE',README_source=ref(ROOT/'README.md'),
        README_facts_to_update=['Random3x completed 512 groups/64 windows/128 steps and generated1360 groups.',
            'Real resume, full raw verifier and independent four-policy evaluation PASS.',
            'SFT/Vanilla512/Random3x512/Dynamic512 accuracy = 59.9609375/63.8671875/62.5/64.6484375 percent.',
            'Dynamic-Random3x +2.1484375pp, CI crosses zero; mechanism INCONCLUSIVE.',
            'New manual-review packet and exact deliverable schema available; Stage4 remains FULL_PASS and Stage5 NOT_STARTED.'],
        next_action='ChatGPT/user completes Stage4 narrative + at least two genuine full-response manual reviews; then rerun full Stage4 verifier. Do not execute selection1024 before READY_FOR_STAGE5=YES.')
    add(ROOT/'experiments/handoffs/stage4_final_docs_to_chatgpt_v1.json',handoff)
    # Existing content must remain unchanged, including parser, reward, overlap
    # manifests, README, project state and every previously tracked experiment.
    for r in protected:
        assert reference(r['path'])==r, r['path']
    for path in OUTPUTS:
        assert not path.exists(), path
    for path, data in OUTPUTS.items():
        path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
    print(json.dumps({'created':[str(p.relative_to(ROOT)) for p in OUTPUTS],
                      'cases':len(packet['cases']),'protected_files':len(protected),'selection_protocol':protocol_ref},indent=2))


if __name__=='__main__':
    main()
