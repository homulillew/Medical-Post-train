#!/usr/bin/env python3
"""Verify observed Stage 0 evidence. Missing/failed gates always fail closed."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
GATES={'runtime','snapshot','template','bf16_load','lora_backward','lora_save','adapter_reload','checkpoint_resume','vllm_load','vllm_lora_identity','actor_rollout_switch','verl_import_compose','verl_fsdp2','reward_dict','gspo_numeric','minibatch_diagnostic','response_length','semantic_diagnostic'}

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def verify(root=ROOT,bulk_hashes=False):
    errors=[]; gates={}; runs=[]
    state=json.loads((root/'project_state.json').read_text())
    def check(condition,message):
        if not condition:errors.append(message)
    check(set(state['stages'])==set('123456'),'Formal stage set changed')
    for stage,s in state['stages'].items():
        check(s['status']=='NOT_STARTED' and not s['run_ids'],f'Stage {stage} changed')
        check(all(v in (0,[]) for v in s['progress'].values()),f'Stage {stage} progress changed')
    contract=root/'contracts/stage_budgets.json'
    check(sha(contract)==state['contract_sha256'],'Formal budget hash drift')
    for p,h in json.loads(contract.read_text())['source_documents'].items():check(sha(root/p)==h,f'Contract source changed: {p}')
    for p in ('env/train.lock','env/analysis.lock','env/environment_manifest.json','env/README.md','docs/stage_reports/00_runtime_compatibility.md'):
        check((root/p).is_file(),f'Missing {p}')
    env_path=root/'env/environment_manifest.json'
    if env_path.is_file():
        frozen=json.loads(env_path.read_text())
        for name,h in frozen.get('locks',{}).items():check((root/name).exists() and sha(root/name)==h,f'Frozen dependency lock drift: {name}')
    selected_path=root/'experiments/stage0/selected_runs.json'
    if not selected_path.exists():return dict(result='FAIL',errors=errors+['No explicit selected runs'],missing_gates=sorted(GATES),gates={},runs=[])
    selected=json.loads(selected_path.read_text())
    log_path=root/'experiments/stage0/logs_receipt.json'
    if log_path.is_file():
        sealed={f['path']:f for f in json.loads(log_path.read_text())['files']}
    else:
        sealed={};errors.append('Final log receipt missing')
    for run_id in selected['run_ids']:
        try:
            rd=root/'experiments/stage0'/run_id
            manifest=json.loads((rd/'manifest.json').read_text())
            check(manifest['run_id']==run_id and manifest['stage']==0 and manifest['run_class'] in ('DIAGNOSTIC','SMOKE'),f'{run_id}: invalid identity/class')
            attempts=sorted(rd.glob('attempt_*')); attempt=attempts[-1]
            status=json.loads((attempt/'status.json').read_text())
            check(status['status']=='PASS',f'{run_id}: not PASS')
            for name in ('resolved_config.json','environment.json','code.json','command.json','stdout.log','stderr.log','metrics.jsonl','artifacts.jsonl','artifacts_manifest.json','checkpoint_refs.json','observations.md'):
                check((attempt/name).is_file(),f'{run_id}: missing {name}')
            for name in ('stdout.log','stderr.log'):
                p=attempt/name;key=str(p.relative_to(root))
                check(key in sealed and sha(p)==sealed[key]['sha256'],f'{run_id}: unsealed/changed log {name}')
            hashes=json.loads((attempt/'artifacts_manifest.json').read_text())
            for f in hashes:
                p=attempt/f['path']; check(p.is_file() and sha(p)==f['sha256'],f'{run_id}: evidence hash mismatch {p}')
            for line in (attempt/'artifacts.jsonl').read_text().splitlines():
                f=json.loads(line); p=Path(f['path'])
                check(p.is_file() and p.stat().st_size==f['size'],f'{run_id}: missing/size mismatch bulk {p}')
                if bulk_hashes and p.is_file():check(sha(p)==f['sha256'],f'{run_id}: bulk hash mismatch {p}')
            metrics=[json.loads(l) for l in (attempt/'metrics.jsonl').read_text().splitlines()]
            check(bool(metrics) and all(m['run_id']==run_id for m in metrics),f'{run_id}: metrics identity/missing')
            summary=json.loads((attempt/'summary.json').read_text())
            for gate,ok in summary.get('gates',{}).items():
                if ok is True and status['status']=='PASS':gates[gate]=run_id
            if 'lora_backward' in summary.get('gates',{}):
                sweep=[m for m in metrics if m['event']=='memory_sweep']
                check({m['sequence_length'] for m in sweep}=={512,1024,2048},'Incomplete memory sweep')
                check(all(m['parameter_delta_l2']>0 and m['nvml_peak_bytes']>16_000_000_000 for m in sweep),'No real 8B training evidence')
            if 'snapshot' in summary.get('gates',{}):
                check(summary['revision']=='b968826d9c46dd6066d109eabc6255188de91218' and summary['source']=='Qwen/Qwen3-8B','Wrong model snapshot')
                for f in summary['files']:
                    p=Path(f['path']); check(p.is_file() and p.stat().st_size==f['size'],'Snapshot missing/size mismatch')
                    if bulk_hashes and p.is_file():check(sha(p)==f['sha256'],'Snapshot hash mismatch')
            if 'vllm_lora_identity' in summary.get('gates',{}):
                check(summary['max_prompt_logprob_delta']>1e-5,'Adapter positive control absent')
                check(summary.get('runner')=='v1' and summary.get('verified_sleep_cycles')==3,'Explicit V1 three-cycle mitigation missing')
                check(summary.get('batch_invariant') is True,'Batch-invariant LoRA mitigation missing')
                check(json.loads((attempt/'lora_shrink_config.json').read_text())['split_k']==1,'LoRA shrink is not deterministic configuration')
                repeats=[m for m in metrics if m['event']=='repeat_sleep_wake']
                check({m['cycle'] for m in repeats}=={2,3} and all(m['token_ids_equal'] and m['max_matched_prompt_logprob_error']<=1e-4 for m in repeats),'Repeated sleep identity evidence missing')
            if 'checkpoint_resume' in summary.get('gates',{}):check(summary['global_step']==4 and summary['continued_parameter_max_error']<=1e-5,'Resume continuation mismatch')
            if 'response_length' in summary.get('gates',{}):check({(m['limit'],m['responses']) for m in metrics if m['event']=='output_length'}=={(512,128),(1024,128)},'Incomplete length comparison')
            if 'gspo_numeric' in summary.get('gates',{}):check({m['device'] for m in metrics if m['event']=='gspo_numeric'}=={'cpu','cuda'},'Numeric device coverage incomplete')
            if 'runtime' in summary.get('gates',{}):
                runtime=summary['runtime']; check('RTX 5880 Ada' in runtime['gpu']['stdout'],'GPU differs'); check(runtime['pip_check']['returncode']==0,'Dependencies inconsistent')
            runs.append(dict(run_id=run_id,status=status['status'],summary_sha256=sha(attempt/'summary.json')))
        except (OSError,ValueError,KeyError,IndexError) as exc:errors.append(f'{run_id}: {exc}')
    missing=GATES-set(gates)
    if missing:errors.append('Missing gates: '+', '.join(sorted(missing)))
    return dict(result='PASS' if not errors else 'FAIL',errors=errors,missing_gates=sorted(missing),gates=gates,runs=runs,bulk_hashes_checked=bulk_hashes,verifier_sha256=sha(Path(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bulk-hashes',action='store_true');p.add_argument('--output');a=p.parse_args()
    result=verify(bulk_hashes=a.bulk_hashes)
    text=json.dumps(result,indent=2,ensure_ascii=False)+'\n'
    if a.output:
        path=Path(a.output)
        with path.open('x') as f:f.write(text)
    print(text)
    sys.exit(0 if result['result']=='PASS' else 1)
