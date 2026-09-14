"""Fail-closed prerequisites; CPU inspection never initializes a CUDA runtime."""
import os,subprocess
from pathlib import Path
from .core import read,ref,check_ref
ROOT=Path(__file__).resolve().parents[3]

def stage4_gate(root=ROOT):
    # Reuse the frozen validator without changing its source/protocol hash.
    import importlib.util
    spec=importlib.util.spec_from_file_location('frozen_selection_gate',Path(root)/'scripts/checkpoint_selection.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    try:return m.require_stage4_gate(root)
    except (OSError,KeyError,ValueError) as e:raise PermissionError('Stage4 gate evidence missing/invalid') from e

def gpu_snapshot():
    result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True)
    if result.returncode:raise PermissionError('GPU ownership cannot be established')
    cuda=[s.strip() for s in result.stdout.splitlines() if s.strip()]
    workers=[]
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=path.read_bytes().split(b'\0');words=[s.decode(errors='replace') for s in argv if s]
        except (OSError,PermissionError):continue
        # Inspect argv elements, not shell command text which may mention a path in a test.
        if any(Path(s).name in ['loss_objective_runtime.py','run_stage4.py','random3x_runtime.py'] for s in words) and any(s in ['worker','supervise','actor-window','actor-diagnostic','reload'] for s in words):
            workers.append(dict(pid=int(path.parent.name),argv=words))
    return dict(cuda_processes=cuda,training_workers=workers)

def gpu_guard(snapshot=None):
    s=gpu_snapshot() if snapshot is None else snapshot
    if s['cuda_processes'] or s['training_workers']:raise PermissionError('GPU is owned by an active process; Stage5 inference refused')
    return s

def verify_seal(root=ROOT):
    base=Path(root)/'experiments/stage5';seal=read(base/'preflight_seal_v1.json')
    for r in seal['artifacts']+seal['sources']:check_ref(r)
    return seal

def selection_gate(root=ROOT):
    try:
        stage4_gate(root);verify_seal(root)
        p=read(Path(root)/'experiments/stage5/checkpoint_selection_protocol_v1.json')
        for r in [p['dataset']['partition'],p['dataset']['source_metadata'],p['parser'],p['prompt']['messages_source'],p['prompt']['chat_template']]:check_ref(r)
        return p
    except (OSError,KeyError,ValueError) as e:raise PermissionError('Selection source/seal evidence missing or invalid') from e

def final_gate(checkpoint_id,root=ROOT):
    root=Path(root);selection_gate(root);base=root/'experiments/stage5'
    try:
        lock=read(base/'selection_result_lock_v1.json');check_ref(lock['selection_result'])
        selected=read(lock['selection_result']['path'])
        if selected['status']!='FROZEN_SELECTED':raise PermissionError('Selection must be FROZEN_SELECTED before ANY final test')
        check_ref(selected['protocol']);p=read(base/'checkpoint_selection_protocol_v1.json')
        if selected['protocol']!=ref(base/'checkpoint_selection_protocol_v1.json'):raise PermissionError('Selection protocol changed')
        for r in selected['summary_sources']:check_ref(r)
        for key in ['vanilla','dynamic']:
            c=selected['selected_checkpoints'][key];expected=next(x for x in p['candidates'][key] if x['checkpoint_id']==c['checkpoint_id'])
            if c!=expected:raise PermissionError('Selected checkpoint identity/hash mismatch')
            check_ref(c['adapter'])
        allowed={'sft'}|{c['checkpoint_id'] for c in selected['selected_checkpoints'].values()}|{c['checkpoint_id'] for c in p['scientific_endpoints'].values()}
        if checkpoint_id not in allowed:raise PermissionError('Checkpoint not frozen for final evaluation')
        for r in lock['final_protocols']:check_ref(r)
        expected=[ref(base/n) for n in ['cmexam_final_eval_protocol_v1.json','cmb_primary_audit_v1.json','open_qa_eval_protocol_v1.json']]
        if lock['final_protocols']!=expected:raise PermissionError('Final protocol binding differs')
        for n in ['cmexam_test_scorable','cmb_exam_clean_2000','cmb_clin','open_qa_retention_200']:
            m=read(base/'manifests'/f'{n}.json');check_ref(m['data']);check_ref(m['requests'])
        return selected
    except (OSError,KeyError,StopIteration,ValueError) as e:raise PermissionError('Final selection/source evidence missing or invalid') from e

def reservation_guard(directory,resume=False,technical_retry=False):
    directory=Path(directory)
    completed=directory/'response.json';pending=directory/'reservation.json'
    if completed.exists():
        if not resume:raise PermissionError('Existing final generation; explicit resume required')
        receipt=read(directory/'receipt.json');check_ref(receipt['response'])
        return 'REUSE_COMPLETE'
    if pending.exists() and not (resume and technical_retry):raise PermissionError('Unresolved technical attempt: explicit resume and technical retry required')
    return 'TECHNICAL_RETRY' if pending.exists() else 'NEW'

def worker_spec_gate(spec,root=ROOT):
    """A directly invoked worker cannot bypass frozen checkpoint/data/decoding identity."""
    root=Path(root);base=root/'experiments/stage5';p=selection_gate(root)
    mode=spec['mode'];cid=spec['checkpoint_id'];dataset=spec['dataset']
    out=Path(read(base/'selection_execution_plan_v1.json')['artifact_root'])
    if mode=='selection':
        allowed={c['checkpoint_id']:c['adapter'] for arm in p['candidates'].values() for c in arm}
        if dataset!='selection':raise PermissionError('Non-selection dataset in selection mode')
        expected_items=ref(out/'selection_inputs/items.jsonl');expected_requests=ref(out/'selection_inputs/requests.jsonl');decoding=p['decoding']
        from .core import rows
        if [r['id'] for r in rows(expected_items['path'])]!=p['dataset']['prompt_ids']:raise PermissionError('Selection ID/order mismatch')
    elif mode=='final':
        selected=final_gate(cid,root);cfg=read(root/'configs/stages/s4_formal_shared.json');check_ref(cfg['initialization']);initial=read(cfg['initialization']['path'])
        allowed={'sft':ref(Path(initial['adapter_path'])/'adapter_model.safetensors')}
        allowed.update({c['checkpoint_id']:c['adapter'] for c in list(selected['selected_checkpoints'].values())+list(p['scientific_endpoints'].values())})
        if dataset not in ['cmexam_test_scorable','cmb_exam_clean_2000','cmb_clin','open_qa_retention_200']:raise PermissionError('Unfrozen final dataset')
        m=read(base/'manifests'/f'{dataset}.json');expected_items=m['data'];expected_requests=m['requests']
        decoding=p['decoding'] if dataset in ['cmexam_test_scorable','cmb_exam_clean_2000'] else read(base/'open_qa_eval_protocol_v1.json')['decoding']
    else:raise PermissionError('Unknown evaluation mode')
    if cid not in allowed or spec['adapter']!=allowed[cid]:raise PermissionError('Worker adapter not a frozen candidate')
    if spec['items']!=expected_items or spec['requests']!=expected_requests or spec['decoding']!=decoding:raise PermissionError('Worker data/decoding differs from frozen protocol')
    if Path(spec['directory'])!=out/mode/cid.replace(':','_')/dataset:raise PermissionError('Worker artifact namespace mismatch')
    base_cfg=read(root/'configs/stages/s4_formal_shared.json');engine=dict(p['runtime']['engine'])
    if dataset in ['cmb_clin','open_qa_retention_200']:engine['max_model_len']=read(base/'open_qa_eval_protocol_v1.json')['max_model_len']
    if spec['config']['model']!=base_cfg['model'] or spec['config']['engine']!=engine or spec['runtime_environment']!=p['runtime']['environment']:raise PermissionError('Worker runtime drift')
    directory=Path(spec['directory'])
    existing=list(directory.glob('*/reservation.json'))
    if existing and not spec['resume']:raise PermissionError('Prior generation attempt exists; explicit resume required before model load')
    if any(not (r.parent/'response.json').exists() for r in existing) and not spec['technical_retry']:raise PermissionError('Unresolved attempt requires explicit technical retry before model load')
    check_ref(spec['adapter']);check_ref(expected_items);check_ref(expected_requests)
    return True
