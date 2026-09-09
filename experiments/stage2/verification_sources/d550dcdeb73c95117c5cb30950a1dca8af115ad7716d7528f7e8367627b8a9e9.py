"""Stage 2 acceptance from raw artifacts, never from summary counts alone."""
from collections import Counter
import csv
import json
import math
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from medical_posttrain.evidence import now,sha256,write_json
from medical_posttrain.evidence.stage2 import read,jsonlines,selected_path,INDEX,record
from medical_posttrain.data.exam import options_schema,messages
from medical_posttrain.data.stage1 import normalize_question,text_hash
from medical_posttrain.reward.parser import canonical_answer,reasoning_text
from medical_posttrain.reward.hybrid import reward
from medical_posttrain.rollout.profile import validate_frozen,request_seed
from medical_posttrain.rollout.statistics import group_summary,summarize


def verify():
    gates={};errors=[];counts={}
    def gate(name,fn):
        try:fn();gates[name]=True
        except Exception as error:gates[name]=False;errors.append(dict(gate=name,error=repr(error)))
    def isolation():
        state=read('project_state.json');assert state['stage0']['status']=='VERIFIED' and state['stages']['1']['status']=='DONE'
        for k in ('3','4','5','6'):
            assert state['stages'][k]['status']=='NOT_STARTED' and not state['stages'][k]['run_ids']
            assert all(v==0 or v==[] for v in state['stages'][k]['progress'].values())
        assert sha256('contracts/stage_budgets.json')==state['contract_sha256']
        ref=state['stages']['1']['verification_receipt'];assert sha256(ref['path'])==ref['sha256'] and read(ref['path'])['result']=='PASS'
    gate('prior_stages_and_unchanged_contract_isolation',isolation)
    data=selected_path('data');formal=selected_path('formal');smoke=selected_path('smoke');semantic=selected_path('semantic')
    pool=jsonlines(data/'candidate_pool.jsonl');lookup={r['prompt_id']:r for r in pool}
    def data_gate():
        cfg=read(data/'config.json');raw=read(cfg['raw_manifest']);clusters=jsonlines(cfg['clusters'])
        sealed=read('experiments/stage1/final_artifacts.json');source=next(r for r in sealed if r['path']==cfg['clusters'])
        assert sha256(cfg['clusters'])==source['sha256']==cfg['clusters_sha256']
        membership={sid:c['cluster_id'] for c in clusters for sid in c['members']}
        heldout={c['cluster_id'] for c in clusters if any(s.startswith(('cmexam_val:','cmexam_test:','cmb_test:')) for s in c['members'])}
        meta=next(r for r in raw['sources'] if r['source']=='cmexam_train');assert sha256(meta['path'])==meta['sha256']
        with open(meta['path']) as f:
            reader=csv.DictReader(f);assert reader.fieldnames==['Question','Options','Answer','Explanation'];source=list(reader)
        assert len(source)==54497
        candidates=[]
        for i,r in enumerate(source):
            sid=f'cmexam_train:{meta["revision"]}:{i}';cid=membership[sid]
            try:
                opts=options_schema(r['Options']);canonical_answer(r['Answer'],''.join(opts))
                if not normalize_question(r['Question']) or cid in heldout:continue
                if any(t in r['Question']+'\n'+r['Options'] for t in ('<think','<answer','<|im_','<|endoftext|>')):continue
            except ValueError:continue
            candidates.append((sid,cid))
        seen=set();expected=[]
        for sid,cid in sorted(candidates,key=lambda x:text_hash('42:pool:'+x[0])):
            if cid not in seen:seen.add(cid);expected.append(sid)
        assert [r['prompt_id'] for r in pool]==expected[:15000]
        assert len(pool)==len(lookup)==len({r['cluster_id'] for r in pool})==15000
        for r in pool:
            original=source[r['source_row']]
            assert r['question']==original['Question'] and r['options']==options_schema(original['Options'])
            assert r['answer_set']==canonical_answer(original['Answer'],''.join(r['options']))
            assert r['reference_explanation']==original['Explanation'] and r['cluster_id']==membership[r['prompt_id']]
            assert r['cluster_id'] not in heldout
        order=sorted(lookup,key=lambda sid:text_hash('42:profiling:'+sid))
        assert read(data/'profiling_prompt_ids.json')==order[:1000]
        assert read(data/'smoke_prompt_ids.json')==order[1000:1050]
        val=read(data/'validation_partitions.json');a,b,c=[set(val[k]) for k in ('monitor','selection','diagnostic_reserve')]
        assert len(a)==512 and len(b)==1024 and len(a|b|c)==6811 and not (a&b or a&c or b&c)
        assert not ({val['cluster_ids'][sid] for sid in a}&{val['cluster_ids'][sid] for sid in b})
        assert not val['generation_performed']
        counts['candidate_pool']=len(pool)
    gate('full_source_validated_clean_deterministic_15k_and_reserved_validation',data_gate)
    def diagnostic_gate():
        summary=read(semantic/'summary.json');pairs=jsonlines(semantic/'controlled_pairs.jsonl')
        assert summary['status']=='PASS' and len(pairs)>=100 and len({p['pair_id'] for p in pairs})==len(pairs)
        assert {r['model_id'] for r in summary['results']}=={'BAAI/bge-m3','abhinand/MedEmbed-small-v0.1'}
        required={'synonym','unrelated','negation','dose_number','age','sex','disease_entity','drug_entity','treatment_direction','duration','causality','repeated_fluff','question_only','correct_conclusion_wrong_explanation','wrong_conclusion_high_overlap'}
        assert required<={p['category'] for p in pairs}
        for model in summary['results']:
            prefix=model['model_id'].split('/')[-1]
            rows=jsonlines(semantic/f'{prefix}_scores.jsonl');assert len(rows)==len(pairs)
            encoding=read(semantic/f'{prefix}_encoding.json');vectors=np.load(semantic/f'{prefix}_vectors.npy')
            index={t:i for i,t in enumerate(encoding['texts'])}
            for row in rows:
                cosine=float(vectors[index[row['anchor']]]@vectors[index[row['candidate']]])
                assert abs(cosine-row['cosine'])<1e-6
            assert model['native_short_text_max_error']<1e-5 and model['truncated_tokens']==0
            for f in model['model_files']:assert sha256(f['path'])==f['sha256']
        frozen=read(INDEX/'reward_manifest.json')
        assert frozen['weights']=={'acc':.8,'sem':.15,'format':.05} and frozen['sampling_metric']=='acc' and frozen['correctness_gating']
        for k in ('parser','reward','semantic'):assert sha256(frozen[k]['path'])==frozen[k]['sha256']
        counts['semantic_pairs']=len(pairs)
    gate('controlled_semantic_comparison_and_frozen_gated_reward',diagnostic_gate)
    def runtime_gate(path,expected_prompts):
        cfg=read(path/'config.json');initial=validate_frozen(cfg)
        assert cfg['sampling']['n']==4 and cfg['planned_prompts']==expected_prompts and cfg['optimizer_updates']==0
        from transformers import AutoTokenizer
        from tokenizers.decoders import DecodeStream
        tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
        selection=read(cfg['selection']['path']);prompts=jsonlines(path/'prompts.jsonl')
        assert [r['prompt_id'] for r in prompts]==selection and len(selection)==expected_prompts
        for prompt in prompts:
            assert prompt['messages']==messages(lookup[prompt['prompt_id']])
            assert tok.apply_chat_template(prompt['messages'],tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)==prompt['prompt_ids']
        pidmap={p['prompt_id']:p['prompt_ids'] for p in prompts}
        raw=[read(p) for p in sorted((path/'raw_groups').glob('*.json'))]
        assert [g['prompt_id'] for g in raw]==selection
        rows=jsonlines(path/'trajectories.jsonl');assert len(rows)==expected_prompts*4
        assert len({r['trajectory_id'] for r in rows})==len(rows)
        assert len({r['policy_version'] for r in rows})==1
        encoding=read(path/'semantic_encoding.json');vectors=np.load(path/'semantic_vectors.npy')
        assert np.isfinite(vectors).all()
        for meta in encoding['metadata']:
            assert meta['truncated_tokens']==0
            if meta['reason']=='encoded':
                spans=meta['chunk_spans'];assert spans[0][0]==0 and spans[-1][1]==meta['encoder_tokens']
                assert all(0<b-a<=480 for a,b in spans) and all(a[1]==b[0] for a,b in zip(spans,spans[1:]))
        rawrows=[r for g in raw for r in g['responses']]
        for r,original in zip(rows,rawrows):
            for k,v in original.items():assert r[k]==v,(r['trajectory_id'],k)
            assert r['finish_reason'] in ('stop','length') and 0<len(r['token_ids'])==r['output_tokens']<=cfg['sampling']['max_tokens']
            assert r['config_sha256']==sha256(path/'config.json') and r['reward_version']==cfg['reward_manifest']['sha256']
            assert r['adapter_sha256']==initial['adapter_sha256'] and r['policy_version']==cfg['policy_version']
            assert r['request_seed']==request_seed(cfg['seed'],r['prompt_id'])
            prompt_ids=pidmap[r['prompt_id']];assert r['prompt_tokens']==len(prompt_ids)
            stream=DecodeStream(ids=prompt_ids,skip_special_tokens=True)
            decoded=''.join(stream.step(tok._tokenizer,t) or '' for t in r['token_ids'])
            assert decoded==r['raw_output'],(r['trajectory_id'],'decode')
            source=lookup[r['prompt_id']]
            assert r['ground_truth']==source['answer_set'] and r['question']==source['question'] and r['options']==source['options']
            reasoning=reasoning_text(decoded);ri=r['reasoning_embedding_index'];ei=r['reference_embedding_index']
            assert encoding['texts'][ri]==reasoning and encoding['texts'][ei]==source['reference_explanation']
            cosine=float(vectors[ri]@vectors[ei]);sem=float(np.clip(cosine,0,1)) if reasoning.strip() and source['reference_explanation'].strip() else 0.
            assert abs(r['semantic']-sem)<1e-6 and abs(r['semantic_cosine']-cosine)<1e-6
            recomputed=reward(decoded,r['ground_truth'],sem,''.join(r['options']),r['finish_reason'])
            assert r['acc']==recomputed['acc'] and r['format']==recomputed['format']
            assert r['parsed']==json.loads(json.dumps(recomputed['parser']))
            assert abs(r['total_reward']-recomputed['score'])<1e-6 and abs(r['semantic_contribution']-recomputed['semantic_contribution'])<1e-6
            assert (r['acc']==0 and r['total_reward']<=.05 and r['semantic_contribution']==0) or (r['acc']==1 and r['total_reward']>=.8)
            span=recomputed['parser']['matched_span'];answer=decoded[span[0]:span[1]].strip() if span else ''
            assert r['reasoning_tokens']==len(tok.encode(reasoning,add_special_tokens=False)) and r['answer_tokens']==len(tok.encode(answer,add_special_tokens=False))
            assert r['answer_closed']==('<answer>' in decoded and '</answer>' in decoded)
        groups=jsonlines(path/'groups.jsonl');assert groups==[group_summary(rows[i:i+4]) for i in range(0,len(rows),4)]
        summary=read(path/'summary.json');recomputed=summarize(rows,groups,path)
        for key,value in recomputed.items():assert summary[key]==value,key
        assert summary['status']=='PASS' and sum(summary['group_counts'].values())==expected_prompts
        assert not summary['costs']['incomplete_requests'], 'Unknown failed tails require explicit accounting resolution'
        generation_attempts=[]
        for attempt in path.glob('attempt_*'):
            command=read(attempt/'command.json')
            if command[command.index('--action')+1]!='rollout':continue
            generation_attempts.append(attempt)
            identity=read(attempt/'identity_receipt.json')
            controls={k:read(attempt/f'identity_{k}.json') for k in ('base','sft','base_negative','sft_repeat','after_wake')}
            def delta(a,b):
                return max(abs(x[k]-y[k]) for x,y in zip(controls[a]['prompt_logprobs'][1:],controls[b]['prompt_logprobs'][1:]) for k in x.keys()&y.keys())
            for field,a,b in [('base_sft_delta','base','sft'),('base_negative_error','base','base_negative'),('repeat_error','sft','sft_repeat'),('wake_error','sft','after_wake')]:
                assert identity[field]==delta(a,b)
            assert controls['base']['token_ids']==controls['base_negative']['token_ids']
            assert controls['sft']['token_ids']==controls['sft_repeat']['token_ids']==controls['after_wake']['token_ids']
            assert identity['adapter_sha256']==initial['adapter_sha256']
            assert identity['base_sft_delta']>1e-5 and identity['repeat_error']<=1e-4 and identity['wake_error']<=1e-4
            assert identity['base_negative_error']<=1e-4 and identity['optimizer_updates']==0
            assert read(attempt/'lora_shrink_config.json')['split_k']==1
            env=read(attempt/'runtime_environment.json');assert env['VLLM_BATCH_INVARIANT']=='1' and env['VLLM_USE_V2_MODEL_RUNNER']=='0' and env['VLLM_USE_FLASHINFER_SAMPLER']=='0'
            assert read(attempt/'vllm_args.json')==dict(model=cfg['model'],tokenizer=cfg['model'],**cfg['engine'])
        assert generation_attempts, 'Missing real rollout identity controls'
        assert read(path/'status.json')['status']=='PASS'
        return rows,groups
    gate('real_50x4_smoke_raw_pipeline',lambda:runtime_gate(smoke,50))
    def formal_gate():
        rows,groups=runtime_gate(formal,1000);counts.update(formal_prompts=len(groups),completed_responses=len(rows),optimizer_updates=0)
        assert set(read(formal/'config.json')['execution_hashes'])
        assert read(formal/'manifest.json')['run_class']=='FORMAL'
        cfg=read(formal/'config.json');smoke_cfg=read(smoke/'config.json')
        for key in ('policy_version','initialization','pool','reward_manifest','sampling','engine','seed','request_batch_prompts','execution_hashes'):
            assert cfg[key]==smoke_cfg[key],key
        assert sha256(cfg['smoke_receipt']['path'])==cfg['smoke_receipt']['sha256']
        assert sha256(formal/'freeze_decision_at_launch.md')==cfg['freeze_decision']['sha256']
    gate('formal_1000x4_fixed_policy_raw_parser_reward_lengths_and_groups',formal_gate)
    def tests_gate():
        receipt=read(INDEX/'tests-final.json');assert receipt['exit_code']==0
        for path,digest in receipt['test_hashes'].items():assert sha256(path)==digest
        assert sha256(receipt['log']['path'])==receipt['log']['sha256']
        result=subprocess.run(['.venv-analysis/bin/python','-m','pytest','tests/test_stage2_reward.py','tests/test_stage2_data_groups.py','-q'],capture_output=True,text=True)
        assert result.returncode==0,result.stdout+result.stderr
    gate('parser_and_reward_invariants_tests_pass',tests_gate)
    def reports_gate():
        review=read(INDEX/'manual_review.json');rows=jsonlines(formal/'trajectories.jsonl');byid={r['trajectory_id']:r for r in rows}
        entries=review['entries'];assert 50<=len(entries)<=100 and len({r['trajectory_id'] for r in entries})==len(entries)
        assert review['review_type']=='manual qualitative review' and not review['clinical_validation']
        for r in entries:
            assert r['trajectory_id'] in byid and len(r['observation'])>=15 and r['parser_review'] in ('PASS','ISSUE_RECORDED')
            assert r['raw_output_sha256']==text_hash(byid[r['trajectory_id']]['raw_output'])
        cases=read(INDEX/'case_coverage.json')
        case_byid={c['case_id']:c for c in cases['cases']}
        assert len(case_byid)==len(cases['cases'])
        for c in cases['cases']:
            raw=byid[c['trajectory_id']]
            assert c['prompt_id']==raw['prompt_id'] and c['group_id']==raw['group_id']
            assert c['raw_output_sha256']==text_hash(raw['raw_output'])
            assert raw['raw_output'].startswith(c['raw_output_excerpt'])
            assert c['ground_truth']==raw['ground_truth'] and c['parsed_answer']==raw['parsed_answer']
            assert c['reward_components']=={k:raw[k] for k in c['reward_components']}
        required={'all_wrong','one_of_four','two_of_four','three_of_four','all_correct','high_semantic_wrong','correct_low_semantic','parser_ambiguous','fallback_correct','format_failure','truncation','long_reasoning','short_reasoning','multi_select'}
        assert required<=set(cases['categories'])
        for category in required:
            item=cases['categories'][category];assert item['status'] in ('OBSERVED','NOT_OBSERVED')
            if item['status']=='OBSERVED':assert item['case_ids'] and all(cid in case_byid for cid in item['case_ids'])
            else:assert not item['case_ids']
        selection=read(INDEX/'manual_review_selection.json')
        assert set(selection['trajectory_ids'])=={r['trajectory_id'] for r in entries}
        assert sha256(selection['packet']['path'])==selection['packet']['sha256']
        report=Path('docs/stage_reports/02_reward_rollout.md').read_text()
        assert len(report)>5000
        for text in ('READY_FOR_STAGE3','30秒','2分钟','Q1','Q10',formal.name,'NOT_ASSESSED','sampling_metric'):assert text in report,text
        assert 'Stage 2' in Path('docs/RESUME_EVIDENCE.md').read_text()
        assert (INDEX/'compute_calibration.json').exists()
    gate('real_cases_50plus_review_report_interview_and_compute',reports_gate)
    def seal_gate():
        seal=read(INDEX/'final_artifacts.json')
        sealed_paths={f['path'] for f in seal};assert len(sealed_paths)==len(seal)
        for f in seal:assert sha256(f['path'])==f['sha256'],f['path']
        for run in (data,semantic,smoke,formal):
            for p in run.rglob('*'):
                if p.is_file():assert str(p) in sealed_paths, f'Unsealed evidence: {p}'
        assert any('raw_groups' in f['path'] for f in seal) and any('FAILED' in Path(f['path']).read_text() for f in seal if f['path'].endswith('status.json'))
    gate('artifact_seal_and_failed_run_retention',seal_gate)
    return dict(stage=2,result='FAIL' if errors else 'PASS',timestamp=now(),gates=gates,errors=errors,counts=counts,
                verifier_sha256=sha256(__file__),contract_sha256=sha256('contracts/stage_budgets.json'))
