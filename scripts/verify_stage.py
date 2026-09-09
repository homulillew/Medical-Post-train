#!/usr/bin/env python3
"""Raw-evidence verification for implemented Stage 1 and Stage 2 gates."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import subprocess
import sys

from medical_posttrain.evidence import sha256,write_json

ROOT=Path(__file__).resolve().parents[1]

def verify_training_records(config,manifest,rows,metrics,coverage,summary):
    """Check actual per-update sample/token lineage against the frozen dataset."""
    assert config['run_class']==manifest['run_class']=='FORMAL', 'Smoke/pilot cannot satisfy formal training'
    assert not manifest['dirty_state'], 'Formal source was not clean'
    assert config['planned_examples']==len(rows)==20000 and config['planned_epochs']==1
    assert Counter(r['source'] for r in rows)==dict(medical_o1=10000,huatuo=10000)
    assert len({r['sample_id'] for r in rows})==20000
    expected=list(rows);random.Random(config['seed']).shuffle(expected)
    ids=[r['sample_id'] for r in expected]
    assert coverage['planned_ids']==ids
    flat=[sid for row in metrics for sid in row['sample_ids']]
    assert flat==coverage['covered_ids']==ids[:len(flat)], 'Duplicate, skipped, reordered or fabricated effective coverage'
    assert len(set(flat))==len(flat) and len(flat)>=19800
    assert coverage['missing_ids']==ids[len(flat):]
    assert coverage['fraction']==len(flat)/20000>=.99
    assert summary['coverage_fraction']==coverage['fraction'] and summary['unique_examples']==len(flat)
    lookup={r['sample_id']:r for r in rows}
    total=supervised=0;cursor=0
    for step,row in enumerate(metrics,1):
        assert row['global_step']==step and row['event']=='update'
        assert row['examples']==len(row['sample_ids'])
        cursor+=row['examples'];assert row['sample_cursor']==cursor
        for key in ('loss','grad_norm','learning_rate','next_learning_rate','update_seconds','tokens_per_second'):
            assert math.isfinite(row[key]) and row[key]>=0, f'Invalid {key}'
        assert row['loss']>0 and row['grad_norm']>0 and row['update_seconds']>0
        assert row['processed_tokens']==sum(lookup[s]['total_tokens'] for s in row['sample_ids'])
        assert row['supervised_tokens']==sum(sum(x!=-100 for x in lookup[s]['labels'][1:]) for s in row['sample_ids'])
        total+=row['processed_tokens'];supervised+=row['supervised_tokens']
    assert summary['global_step']==len(metrics)
    assert total==summary['processed_tokens']==coverage['processed_tokens']
    assert supervised==summary['supervised_tokens']==coverage['supervised_tokens']
    assert supervised/sum(sum(x!=-100 for x in r['labels'][1:]) for r in rows)>=.99
    assert summary['initial_trainable_digest']!=summary['final_trainable_digest']
    return dict(unique_examples=len(flat),coverage=len(flat)/20000,updates=len(metrics),processed_tokens=total,supervised_tokens=supervised)

def verify(root=ROOT):
    from verify_stage1_data import verify as verify_data
    errors=[];gates={};counts={}
    def gate(name,fn):
        try:
            value=fn();gates[name]=True;return value
        except (AssertionError,KeyError,FileNotFoundError,ValueError,TypeError) as e:
            gates[name]=False;errors.append(f'{name}: {type(e).__name__}: {e}');return None
    def state_gate():
        s=json.loads((root/'project_state.json').read_text())
        assert s['stage0']['status']=='VERIFIED'
        assert sha256(root/'contracts/stage_budgets.json')==s['contract_sha256']
        for stage in '23456':assert s['stages'][stage]['status']=='NOT_STARTED' and not s['stages'][stage]['run_ids']
        budget=json.loads((root/'contracts/stage_budgets.json').read_text())
        for file,h in budget['source_documents'].items():assert sha256(root/file)==h
        receipt=json.loads((root/s['stage0']['verification_receipt']).read_text());assert receipt['result']=='PASS'
        return s
    state=gate('research_contract_and_stage_isolation',state_gate)
    selected=gate('explicit_selected_runs',lambda:json.loads((root/'experiments/stage1/selected_runs.json').read_text()))
    if not selected:return dict(result='FAIL',errors=errors,gates=gates,counts=counts)
    gate('frozen_balanced_data_and_decontamination',lambda:verify_data(root/'experiments/stage1'/selected['data']))
    def run_path(key):
        assert selected[key],f'Missing {key} run'
        manifest=json.loads((root/'experiments/stage1'/selected[key]/'manifest.json').read_text())
        return Path(manifest['artifact_root'])
    def small_gate(key,examples,updates):
        path=run_path(key);cfg=json.loads((path/'config.json').read_text());summary=json.loads((path/'summary.json').read_text())
        assert cfg['run_class']==key.upper() and summary['examples']==examples and summary['global_step']==updates
        receipt=json.loads((path/'reload_generation/receipt.json').read_text());assert receipt['status']=='PASS'
        assert sha256(receipt['generations_path'])==receipt['generations_sha256'] and receipt['generation_count']>=4
        assert receipt['identity_logit_max_delta']>1e-5
        if key=='pilot':
            resume=json.loads((path/'resume_receipt.json').read_text());assert resume['parameter_max_abs_error']<=1e-5 and resume['loss_abs_error']<1e-5 and resume['same_next_sample_ids'] and resume['scheduler_equal']
            assert sha256(path/'resume_reference.pt')==resume['reference_parameters_sha256']
            attempts=sorted(path.glob('attempt_*'));assert len(attempts)>=2
            events=[json.loads(line) for a in attempts for line in (a/'metrics.jsonl').open()]
            restored=[r for r in events if r['event']=='resume_restore'];assert restored and restored[-1]['sample_cursor']==512 and restored[-1]['global_step']==32
            launched=[json.loads(p.read_text()) for p in path.glob('launch_*.json')];assert len({r['pid'] for r in launched})>=2
        return True
    gate('real_smoke_and_reload',lambda:small_gate('smoke',128,8))
    gate('real_pilot_and_process_resume',lambda:small_gate('pilot',1024,64))
    formal=gate('formal_run_exists',lambda:run_path('formal'))
    if formal:
        def formal_gate():
            config=json.loads((formal/'config.json').read_text());manifest=json.loads((formal/'manifest.json').read_text());summary=json.loads((formal/'summary.json').read_text())
            assert sha256(formal/'config.json')==manifest['config_sha256']
            assert config['model_revision']=='b968826d9c46dd6066d109eabc6255188de91218' and config['model_id']=='Qwen/Qwen3-8B' and config['dtype']=='bfloat16'
            assert config['lora_rank']==32 and config['lora_alpha'] in (32,64) and not config['packing']
            assert set(config['lora_targets'])=={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}
            for launch in formal.glob('launch_*.json'):
                assert json.loads(launch.read_text())['pytorch_alloc_conf']==config['pytorch_alloc_conf']
            assert sha256(config['train_path'])==config['train_sha256'] and sha256(config['validation_path'])==config['validation_sha256']
            rows=[json.loads(l) for l in open(config['train_path'])];metrics=[json.loads(l) for l in (formal/'canonical_metrics.jsonl').open()];coverage=json.loads((formal/'coverage.json').read_text())
            result=verify_training_records(config,manifest,rows,metrics,coverage,summary)
            # Reconcile canonical metrics with the retained physical attempt stream.
            physical={}
            for attempt in formal.glob('attempt_*'):
                provenance=json.loads((attempt/'provenance.json').read_text());assert provenance['source_hashes']==manifest['source_hashes']
                for line in (attempt/'metrics.jsonl').open():
                    row=json.loads(line)
                    if row['event']=='update':physical[(row['attempt'],row['global_step'])]=row
            assert all(row==physical[(row['attempt'],row['global_step'])] for row in metrics)
            assert (formal/'source.zip').is_file() and (formal/'environment.json').is_file()
            assert summary['validation'][0]['global_step']==0 and summary['validation'][-1]['global_step']==len(metrics)
            for v in summary['validation']:
                assert math.isfinite(v['loss']) and v['loss']>0
            assert summary['validation'][0]['examples']==summary['validation'][-1]['examples']==1000
            return result
        result=gate('full_epoch_raw_metrics_and_coverage',formal_gate)
        if result:counts.update(result)
        def checkpoints_gate():
            from medical_posttrain.training.stage1 import check_checkpoint
            summary=json.loads((formal/'summary.json').read_text())
            paths=[p for p in (formal/'checkpoints').iterdir() if (p/'COMPLETE.json').exists()]
            assert len(paths)>=2
            for path in paths:check_checkpoint(path)
            final=check_checkpoint(summary['final_checkpoint'])
            assert final['sample_cursor']>=19800 and final['trainable_digest']==summary['final_trainable_digest']
            import torch
            from medical_posttrain.training.stage1 import tensor_state_digest
            state=torch.load(Path(summary['final_checkpoint'])/'state.pt',map_location='cpu',weights_only=False)
            metrics=[json.loads(l) for l in (formal/'canonical_metrics.jsonl').open()]
            coverage=json.loads((formal/'coverage.json').read_text())
            assert state['canonical_updates']==metrics and state['covered_ids']==coverage['covered_ids']
            assert state['global_step']==summary['global_step'] and state['scheduler']['last_epoch']==state['global_step']
            assert state['optimizer']['state'] and tensor_state_digest(state['optimizer'])==final['optimizer_digest']
            assert tensor_state_digest(state['rng'])==final['rng_digest']
            adapter=Path(summary['final_adapter']);assert sha256(adapter/'adapter_model.safetensors')==summary['adapter_sha256']
            ac=json.loads((adapter/'adapter_config.json').read_text());config=json.loads((formal/'config.json').read_text())
            assert ac['r']==32 and ac['lora_alpha']==config['lora_alpha']
            reload=json.loads((formal/'reload_generation/receipt.json').read_text());assert reload['status']=='PASS' and reload['adapter_sha256']==summary['adapter_sha256'] and reload['trainable_digest']==summary['final_trainable_digest']
            assert sha256(reload['generations_path'])==reload['generations_sha256']
        gate('checkpoints_and_final_fresh_process_reload',checkpoints_gate)
    def eval_gate():
        path=run_path('evaluation');record=root/'experiments/stage1'/selected['evaluation'];summary=json.loads((record/'summary.json').read_text());manifest=json.loads((record/'manifest.json').read_text())
        assert summary['status']=='PASS' and 50<=summary['prompt_count']<=100
        for artifact in json.loads((record/'artifacts_manifest.json').read_text()):assert sha256(artifact['path'])==artifact['sha256']
        prompts=json.loads((path/'prompts.json').read_text());ids=[r['sample_id'] for r in prompts];assert len(set(ids))==summary['prompt_count']
        assert Counter(r['source'] for r in prompts)==dict(medical_o1=25,huatuo=25)
        formal_config=json.loads((formal/'config.json').read_text());val=[json.loads(l) for l in open(formal_config['validation_path'])]
        assert ids==[r['sample_id'] for r in val[:25]+val[500:525]]
        from evaluate_stage1 import aggregate
        for cap in summary['limits']:
            paired=[]
            for model in ('base','sft'):
                rows=[json.loads(l) for l in (path/f'{model}_{cap}.jsonl').open()]
                assert [r['sample_id'] for r in rows]==ids
                assert aggregate(rows)==summary['results'][f'{model}_{cap}']
                assert all(len(r['output_ids'])==r['total_tokens']<=cap for r in rows)
                if model=='sft':assert all(r['adapter_sha256']==manifest['parent_adapter_sha256'] for r in rows)
                paired.append(rows)
            assert all(a['prompt_ids']==b['prompt_ids'] for a,b in zip(*paired))
        assert summary['identity']['base_sft_prompt_logprob_delta']>1e-5 and summary['identity']['repeat_sft_prompt_logprob_error']<=1e-4
    gate('paired_heldout_generation_and_length_analysis',eval_gate)
    def reports_gate():
        for file in ('docs/stage_reports/01_medical_sft.md','docs/implementation/STAGE1_DECISIONS.md','docs/implementation/COMPUTE_BUDGET.md','docs/RESUME_EVIDENCE.md','experiments/stage1/case_coverage.json','experiments/stage1/final_artifacts.json'):
            assert (root/file).is_file(),file
        report=(root/'docs/stage_reports/01_medical_sft.md').read_text()
        for text in (selected['formal'],selected['evaluation'],'30秒','2分钟','READY_FOR_STAGE2','限制'):assert text in report
        assert len(report)>3000
        cases=json.loads((root/'experiments/stage1/case_coverage.json').read_text());assert cases['cases'] and cases['not_observed']
        for artifact in json.loads((root/'experiments/stage1/final_artifacts.json').read_text()):assert sha256(artifact['path'])==artifact['sha256']
    gate('real_report_cases_interview_and_artifact_seal',reports_gate)
    return dict(result='FAIL' if errors else 'PASS',stage=1,errors=errors,gates=gates,counts=counts,verifier_sha256=sha256(__file__),contract_sha256=sha256(root/'contracts/stage_budgets.json'),selected_runs=selected)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True);parser.add_argument('--output');args=parser.parse_args()
    if args.stage==1:
        result=verify()
    elif args.stage==2:
        from verify_stage2 import verify as verify_stage2
        result=verify_stage2()
    elif args.stage==3:
        from verify_stage3 import verify as verify_stage3
        result=verify_stage3()
    else:raise SystemExit('Stage 4–6 verifiers are not implemented or authorized in this stage.')
    if args.output:write_json(args.output,result)
    print(json.dumps(result,indent=2));raise SystemExit(result['result']!='PASS')
