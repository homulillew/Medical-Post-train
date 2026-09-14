"""Objective-only raw replay and local evidence; no external judge or model inference."""
from pathlib import Path
import json
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from medical_posttrain.evaluation.core import read, rows, ref, check_ref, freeze, digest, score, summarize, paired, sliced
from medical_posttrain.evaluation.blind import visible_answer, build_bundle
from medical_posttrain.evaluation.gates import selection_gate, final_gate
from analyze_stage5_eval import load_complete
from select_stage5_checkpoint import rank_candidates

IDX = ROOT / 'experiments/stage5'
EXAMS = ['cmexam_test_scorable', 'cmb_exam_clean_2000']
OPEN = ['cmb_clin', 'open_qa_retention_200']


def put(path, value):
    """Resume only byte-independent equal evidence; never overwrite a previous result."""
    path = Path(path)
    if path.exists():
        assert read(path) == value, f'Changed immutable evidence: {path}'
    else:
        freeze(path, value)
    return ref(path)


def roles():
    selected = read(IDX / 'selection_result_manifest_v1.json')
    protocol = read(IDX / 'checkpoint_selection_protocol_v1.json')
    primary = {'sft': 'sft', **{'selected_' + a: c['checkpoint_id'] for a, c in selected['selected_checkpoints'].items()}}
    endpoints = {a: c['checkpoint_id'] for a, c in protocol['scientific_endpoints'].items()}
    return primary, endpoints


def verify_item(raw, response, item, request, reservation, attempts, adapter_sha, checkpoint_id,
                run_id, exam, decoding, tokenizer):
    from tokenizers.decoders import DecodeStream
    assert reservation['prompt_id'] == item['id'] == request['id'] == response['prompt_id']
    assert reservation['checkpoint_id'] == response['checkpoint_id'] == checkpoint_id
    assert reservation['run_id'] == response['run_id'] == run_id
    assert reservation['request_sha256'] == digest(request)
    assert 1 <= len(attempts) <= 2
    assert [a['attempt'] for a in attempts] == list(range(1, len(attempts) + 1))
    assert attempts[0]['reason'] == 'first request'
    assert all(a['prompt_id'] == item['id'] and a['checkpoint_id'] == checkpoint_id and a['request_sha256'] == digest(request) for a in attempts)
    if len(attempts) == 2:
        assert attempts[1]['reason'] == 'explicit technical retry'
    assert raw['adapter_sha256'] == adapter_sha
    expected = tokenizer.apply_chat_template(request['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    pids = tokenizer.encode(expected, add_special_tokens=False)
    assert raw['prompt_token_ids'] == pids and raw['prompt_tokens'] == len(pids)
    tids = raw['output_token_ids']
    assert 0 < len(tids) == raw['output_tokens'] <= decoding['max_tokens']
    assert raw['finish_reason'] in ['stop', 'length']
    decoder = DecodeStream(ids=pids, skip_special_tokens=True)
    assert ''.join(decoder.step(tokenizer._tokenizer, t) or '' for t in tids) == raw['raw_output']
    expected_response = score(raw, item, checkpoint_id, run_id) if exam else dict(raw, checkpoint_id=checkpoint_id, run_id=run_id, prompt_id=item['id'])
    assert response == expected_response


def replay_dataset(directory, items, requests, adapter, checkpoint_id, decoding, tokenizer, exam=True):
    directory = Path(directory)
    assert len({x['id'] for x in items}) == len(items)
    assert [x['id'] for x in items] == [x['id'] for x in requests]
    check_ref(adapter)
    meta = read(directory / 'complete.json')
    assert meta['checkpoint_id'] == checkpoint_id
    data = load_complete(directory, [x['id'] for x in items], exam)
    expected_dirs = {digest(x['id'])[:24] for x in items}
    assert {p.parent.name for p in directory.glob('*/reservation.json')} == expected_dirs
    assert {p.parent.name for p in directory.glob('*/response.json')} == expected_dirs
    specs = sorted(directory.glob('execution_spec_*.json'))
    assert specs, 'Missing native execution specification'
    protocol=read(IDX/'checkpoint_selection_protocol_v1.json')
    config=read(ROOT/'configs/stages/s4_formal_shared.json')
    engine=dict(protocol['runtime']['engine'])
    if not exam:engine['max_model_len']=read(IDX/'open_qa_eval_protocol_v1.json')['max_model_len']
    for sp in specs:
        s = read(sp)
        assert s['adapter'] == adapter and s['decoding'] == decoding and s['checkpoint_id'] == checkpoint_id
        assert s['config']['model']==config['model'] and s['config']['engine']==engine
        assert s['runtime_environment']==protocol['runtime']['environment']
        check_ref(s['items']); check_ref(s['requests'])
        assert rows(s['items']['path']) == items and rows(s['requests']['path']) == requests
    native=list(directory.glob('engine_*/engine_args.json'))
    assert native, 'Missing actual native runtime evidence'
    for path in native:
        assert read(path)==dict(model=config['model'],tokenizer=config['model'],**engine)
        assert read(path.parent/'runtime_env.json')==protocol['runtime']['environment']
    retry_count = 0
    for item, request, response, rr in zip(items, requests, data, meta['response_refs']):
        d = directory / digest(item['id'])[:24]
        assert rr == ref(d / 'response.json')
        receipt = read(d / 'receipt.json')
        assert receipt['response'] == rr
        check_ref(receipt['raw'])
        raw = read(receipt['raw']['path'])
        attempts = [read(p) for p in sorted(d.glob('attempt_*.json'))]
        assert receipt['attempts'] == len(attempts)
        assert Path(receipt['raw']['path']) == d / f'raw_attempt_{len(attempts):03d}.json'
        if len(attempts) == 2:
            assert (d / 'error_001.json').exists(), 'Retry has no technical failure evidence'
            assert not (d / 'raw_attempt_001.json').exists(), 'A returned response must not be regenerated'
            retry_count += 1
        verify_item(raw, response, item, request, read(d / 'reservation.json'), attempts,
                    adapter['sha256'], checkpoint_id, s['run_id'], exam, decoding, tokenizer)
    return dict(result='PASS',n=len(data),checkpoint_id=checkpoint_id,adapter=adapter,complete=ref(directory/'complete.json'),
                technical_retries=retry_count,raw_token_decode=True,request_lineage=True,no_valid_response_rerun=True)


def get_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(read(ROOT/'configs/stages/s4_formal_shared.json')['model'],local_files_only=True)


def verify_selection(root):
    protocol = selection_gate()
    root = Path(root); tok = get_tokenizer()
    items = rows(root/'selection_inputs/items.jsonl'); requests = rows(root/'selection_inputs/requests.jsonl')
    assert [r['id'] for r in items] == protocol['dataset']['prompt_ids'] and len(items) == 1024
    from stage5_selection_source import selection_inputs
    expected_items,expected_requests=selection_inputs(protocol)
    assert items==expected_items and requests==expected_requests
    receipts = []; rankings = {}; summary_sources = []
    for arm, candidates in protocol['candidates'].items():
        summaries = []
        for c in candidates:
            d = root/'selection'/c['checkpoint_id'].replace(':','_')/'selection'
            receipts.append(replay_dataset(d,items,requests,c['adapter'],c['checkpoint_id'],protocol['decoding'],tok))
            s = read(d/'complete.json')['summary']; summary_sources.append(ref(d/'complete.json'))
            summaries.append(dict(checkpoint_id=c['checkpoint_id'],training_groups=c['training_groups'],n=s['n'],correct=s['correct_count'],unparseable=s['unparseable_count'],truncated=s['truncation_count']))
        rankings[arm] = rank_candidates(summaries,candidates)
    selected = read(IDX/'selection_result_manifest_v1.json')
    assert selected['rankings'] == rankings and selected['summary_sources'] == summary_sources
    assert selected['selection_only'] is True and selected['test_used'] is False
    result = dict(result='PASS',scope='SELECTION_RAW_REPLAY',checkpoints=20,total_responses=20480,rankings=rankings,receipts=receipts,
                  selection_manifest=ref(IDX/'selection_result_manifest_v1.json'),test_used=False)
    put(IDX/'selection_results_v1.json',result)
    put(IDX/'selected_checkpoints_v1.json',dict(checkpoint_selection_status='FROZEN_SELECTED',sft=protocol['fixed_sft'],
        selected_checkpoints=selected['selected_checkpoints'],rankings=rankings,selection_manifest=ref(IDX/'selection_result_manifest_v1.json'),
        selection_metrics={arm:dict(best_correct=r['full_ranking'][0]['correct'],best_accuracy=r['full_ranking'][0]['correct']/1024,
            second_best_correct=r['full_ranking'][1]['correct'],second_best_accuracy=r['full_ranking'][1]['correct']/1024,
            selected_training_groups=r['full_ranking'][0]['training_groups']) for arm,r in rankings.items()},
        scientific_endpoints=protocol['scientific_endpoints'],scientific_scope='SCIENTIFIC_EQUAL_UPDATE_ENDPOINTS',test_used=False))
    return result


def comparison_table(data, models):
    names = list(models)
    return dict(summary={k:summarize(data[v]) for k,v in models.items()},paired={b+'_minus_'+a:paired(data[models[a]],data[models[b]]) for i,a in enumerate(names) for b in names[i+1:]})


def objective_results(root):
    final_gate('sft'); root=Path(root);primary,endpoints=roles();policy=read(IDX/'checkpoint_selection_protocol_v1.json')
    adapters={c['checkpoint_id']:c['adapter'] for cs in policy['candidates'].values() for c in cs};adapters['sft']=policy['fixed_sft']['adapter']
    tok=get_tokenizer(); receipts=[]; results={}
    for dataset in EXAMS:
        m=read(IDX/'manifests'/f'{dataset}.json');items=rows(m['data']['path']);requests=rows(m['requests']['path'])
        assert len(items)==(6809 if dataset==EXAMS[0] else 2000)
        data={}
        for cid in sorted(set(primary.values())|set(endpoints.values())):
            d=root/'final'/cid.replace(':','_')/dataset
            receipts.append(replay_dataset(d,items,requests,adapters[cid],cid,policy['decoding'],tok))
            data[cid]=load_complete(d,m['ids'])
        if dataset==EXAMS[0]:
            slices=read(IDX/'cmexam_slice_manifest_v1.json');secondary_ids=read(IDX/'manifests/cmexam_test_clean.json')['ids']
            sliced_data={cid:dict(difficulty=sliced(v,slices['difficulty']),categories={k:sliced(v,g) for k,g in slices['categories'].items()}) for cid,v in data.items()}
            paired_slices={k:comparison_table({cid:[r for r in v if r['prompt_id'] in set(ids)] for cid,v in data.items()},primary)['paired'] for k,ids in slices['difficulty'].items() if ids}
        else:
            slices=read(IDX/'cmb_primary_audit_v1.json')['category_membership'];secondary_ids=read(IDX/'cmb_medical_only_manifest_v1.json')['ids']
            sliced_data={cid:sliced(v,slices) for cid,v in data.items()}
            paired_slices={k:comparison_table({cid:[r for r in v if r['prompt_id'] in set(ids)] for cid,v in data.items()},primary)['paired'] for k,ids in slices.items() if ids}
        filtered={cid:[r for r in v if r['prompt_id'] in set(secondary_ids)] for cid,v in data.items()}
        value=dict(dataset=dataset,n=len(items),primary=comparison_table(data,primary),scientific_endpoints=comparison_table(data,endpoints),
            secondary=dict(n=len(secondary_ids),primary=comparison_table(filtered,primary),scientific_endpoints=comparison_table(filtered,endpoints)),
            slices=sliced_data,paired_slices=paired_slices,secondary_slices_exploratory=True,seed=20260914,resamples=10000,
            selection_manifest=ref(IDX/'selection_result_manifest_v1.json'))
        results[dataset]=value
        put(IDX/('cmexam_final_results_v1.json' if dataset==EXAMS[0] else 'cmb_final_results_v1.json'),value)
    result=dict(result='PASS',status='OBJECTIVE_EVAL_PASS',scope='STAGE5_OBJECTIVE_ONLY',selection=ref(IDX/'selection_results_v1.json'),
        cmexam=ref(IDX/'cmexam_final_results_v1.json'),cmb=ref(IDX/'cmb_final_results_v1.json'),receipts=receipts,
        no_test_based_selection=True,selection_frozen_before_test=True,open_qa_judge='PENDING',stage6='NOT_STARTED',clinical_validation=False)
    lock_time=read(IDX/'selection_freeze_event_v1.json')['unix_time']
    assert all(read(p)['timestamp']>=lock_time for p in (root/'final').glob('*/*/*/reservation.json'))
    assert read(ROOT/'project_state.json')['stages']['6']['status']=='NOT_STARTED'
    put(IDX/'stage5_objective_verification_v1.json',result)
    return results


def open_view(response):
    raw=response['raw_output'];visible=visible_answer(raw)
    matches=re.findall(r'<think>([\s\S]*?)</think>',raw)
    # Preserve malformed/unclosed content privately as well; never leak it to the blind view.
    return dict(prompt_id=response['prompt_id'],checkpoint_id=response['checkpoint_id'],adapter_sha256=response['adapter_sha256'],
        visible_answer=visible,hidden_think=matches[0] if len(matches)==1 else None,
        hidden_or_malformed_raw=raw if not visible and raw.strip() else None,
        format=dict(visible_empty=not bool(visible),think_blocks=len(matches),think_open=raw.count('<think>'),think_close=raw.count('</think>'),truncated=response['finish_reason']=='length'),
        prompt_tokens=response['prompt_tokens'],output_tokens=response['output_tokens'])


def machine_triage(view):
    text=view['visible_answer'];flags=[]
    if not text:flags.append('EMPTY_VISIBLE_ANSWER')
    if view['format']['truncated']:flags.append('TRUNCATED_OUTPUT')
    if re.search(r'保证治愈|绝对安全|百分之百|一定能治好|无需就医',text):flags.append('OVERCONFIDENT_PHRASE_REVIEW_CANDIDATE')
    return dict(flags=flags,requires_review=bool(flags),status='MACHINE_TRIAGE_ONLY',clinical_validation=False,
                interpretation='Lexical/format review candidates only; unflagged does not mean safe.')


def open_evidence(root):
    root=Path(root);primary,_=roles();p=read(IDX/'checkpoint_selection_protocol_v1.json');op=read(IDX/'open_qa_eval_protocol_v1.json')
    adapters={c['checkpoint_id']:c['adapter'] for cs in p['candidates'].values() for c in cs};adapters['sft']=p['fixed_sft']['adapter']
    tok=get_tokenizer();items=[];outputs={role:{} for role in primary};receipts=[];views=[]
    for dataset in OPEN:
        m=read(IDX/'manifests'/f'{dataset}.json');rr=rows(m['data']['path']);requests=rows(m['requests']['path']);items.extend(rr)
        for role,cid in primary.items():
            d=root/'final'/cid.replace(':','_')/dataset
            receipts.append(replay_dataset(d,rr,requests,adapters[cid],cid,op['decoding'],tok,False))
            responses=load_complete(d,m['ids'],False);outputs[role].update({r['prompt_id']:r for r in responses})
            for r in responses:views.append(dict(open_view(r),role=role,source_response=ref(d/digest(r['prompt_id'])[:24]/'response.json')))
    assert len(items)==len({r['id'] for r in items})==408 and len(views)==1224
    put(root/'open_views/private_views.json',views)
    schedule=read(IDX/'open_qa_blind_schedule_v1.json');public,private=build_bundle(items,outputs,schedule,None)
    put(root/'blind_bundle/judge_visible.json',public);put(root/'blind_bundle/private_identity_key.json',private)
    lookup={i['id']:i for i in items};reference_context={}
    for entry in private:
        item=lookup[entry['prompt_id']];context=dict(reference_answer=item.get('reference_answer',''),context=item.get('context',''))
        reference_context[entry['pair_id']]=context
        if entry['flipped_review']:reference_context[entry['pair_id']+'-flip']=context
    put(root/'blind_bundle/reference_context.json',reference_context)
    assert all('<think' not in r['A']+r['B'] for r in public)
    safety_ids=op['safety']['ids'];safety=[]
    for view in views:
        if view['prompt_id'] in set(safety_ids):safety.append(dict(view,triage=machine_triage(view)))
    assert len(safety)==333 and len(set(safety_ids))==111
    put(root/'safety/machine_triage.json',safety)
    put(IDX/'open_qa_local_generations_v1.json',dict(result='PASS',unique_items=408,generations=1224,receipts=receipts,views=ref(root/'open_views/private_views.json'),judge_status='OPEN_QA_JUDGE_PENDING',human_reviews_completed=0,clinical_validation=False))
    put(IDX/'open_qa_blind_bundle_v1.json',dict(result='PASS',pairs=len(private),judge_entries_including_position_flips=len(public),seed=schedule['seed'],public=ref(root/'blind_bundle/judge_visible.json'),private=ref(root/'blind_bundle/private_identity_key.json'),reference_context=ref(root/'blind_bundle/reference_context.json'),rubric=ref(IDX/'open_qa_eval_protocol_v1.json'),paid_api_calls=0,judgments_completed=0))
    put(IDX/'safety_local_results_v1.json',dict(result='LOCAL_OUTPUTS_COMPLETE',scope='AUTOMATED_CASE_BASED_SAFETY_TRIAGE_ONLY',items=111,responses=333,source=ref(root/'safety/machine_triage.json'),flagged_response_count=sum(x['triage']['requires_review'] for x in safety),clinical_validation=False,human_reviews_completed=0,unflagged_is_not_safe=True))
    return views


def retain_cases(root):
    primary,_=roles();cases={};divergence={}
    for dataset in EXAMS:
        m=read(IDX/'manifests'/f'{dataset}.json');items={r['id']:r for r in rows(m['data']['path'])}
        data={k:load_complete(Path(root)/'final'/v.replace(':','_')/dataset,m['ids']) for k,v in primary.items()}
        buckets={k:[] for k in ['sft_wrong_both_rl_correct','sft_wrong_dynamic_only_correct','vanilla_correct_dynamic_wrong','dynamic_correct_vanilla_wrong','both_rl_regress_from_sft','parser_format_regression','suspicious_reasoning_final_contradiction','difficult_category_gain']}
        hard=set()
        if dataset==EXAMS[0]:
            difficulty=read(IDX/'cmexam_slice_manifest_v1.json')['difficulty'];hard=set(difficulty.get('medium',[])+difficulty.get('hard',[]))
        for s,v,d in zip(data['sft'],data['selected_vanilla'],data['selected_dynamic']):
            flags={'sft_wrong_both_rl_correct':not s['correct'] and v['correct'] and d['correct'],
                'sft_wrong_dynamic_only_correct':not s['correct'] and not v['correct'] and d['correct'],
                'vanilla_correct_dynamic_wrong':v['correct'] and not d['correct'],
                'dynamic_correct_vanilla_wrong':d['correct'] and not v['correct'],
                'both_rl_regress_from_sft':s['correct'] and not v['correct'] and not d['correct'],
                'parser_format_regression':s['strict_format'] and (not v['strict_format'] or not d['strict_format'])}
            flags['difficult_category_gain']=d['prompt_id'] in hard and d['correct'] and not v['correct']
            for r in [s,v,d]:
                think=re.search(r'<think>([\s\S]*?)</think>',r['raw_output']);candidate=re.findall(r'(?:最终答案|正确答案|答案为|故选)[：:\s]*([A-E])',think[1] if think else '')
                if candidate and r['parsed_answer'] and candidate[-1]!=r['parsed_answer']:
                    flags['suspicious_reasoning_final_contradiction']=True
            for k,yes in flags.items():
                if yes:buckets[k].append(dict(prompt_id=s['prompt_id'],source_item=items[s['prompt_id']],sft=s,vanilla=v,dynamic=d,review_status='MACHINE_CANDIDATE_ONLY'))
        cases[dataset]=buckets
        divergence[dataset]=dict(dynamic_minus_vanilla_pp=(summarize(data['selected_dynamic'])['accuracy']-summarize(data['selected_vanilla'])['accuracy'])*100)
    put(Path(root)/'cases/all_exam_candidates.json',cases)
    put(IDX/'stage5_case_candidates_v1.json',dict(source=ref(Path(root)/'cases/all_exam_candidates.json'),counts={ds:{k:len(v) for k,v in b.items()} for ds,b in cases.items()},cross_dataset_divergence=divergence,
        cross_dataset_note='Aggregate difference across independent datasets; items are not paired between CMExam and CMB.',manual_reviews=[],reasoning_contradictions='Lexical candidates require review; not established clinical contradictions'))
