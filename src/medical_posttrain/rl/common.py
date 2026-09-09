"""Durable Stage4 artifacts, inherited input guards and run identity."""
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import zipfile

from medical_posttrain.evidence import now, sha256, write_json
from medical_posttrain.evidence.stage2 import read, record, jsonlines
from medical_posttrain.evidence.stage3 import source_hashes

ROOT = Path(__file__).resolve().parents[3]
BULK = Path('/data/WSH/medical-post-train-artifacts/runs')
INDEX = ROOT/'experiments/stage4'


def durable(path, value):
    path = Path(path)
    write_json(path, value)
    with path.open('rb') as f:
        os.fsync(f.fileno())
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    durable(path, value)


def event(out, name, **fields):
    row = dict(event=name, timestamp=now(), **fields)
    path = Path(out)/'events.jsonl'
    with path.open('a') as f:
        f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)
    return row


def inherited():
    src = read(ROOT/'configs/stages/s3_formal.json')
    keys = ('model','initialization','pool','reward_manifest','sampling','engine','execution_hashes','seed')
    cfg = {k: src[k] for k in keys}
    cfg.update(stage=4, groups_per_window=8, mini_prompts=4, ppo_epochs=1,
               sampling_metric='acc', max_generation_batches=32, request_batch_prompts=8,
               clip_ratio_low=.0003, clip_ratio_high=.0004, grad_clip=1.,
               learning_rate=1e-6, weight_decay=0., betas=[.9,.999], eps=1e-8,
               lr_scheduler='constant', warmup_steps=0, entropy_coefficient=0.,
               kl_reward=False, kl_loss=False, reference_worker=False, critic=False,
               parity_limits=dict(token_mean_abs=.03, token_p99_abs=.25, sequence_mean_abs_max=.02))
    return cfg


def validate(cfg):
    for key in ('initialization','pool','reward_manifest'):
        assert sha256(cfg[key]['path']) == cfg[key]['sha256'], key
    for path, digest in cfg['execution_hashes'].items():
        assert sha256(ROOT/path) == digest, path
    initial = read(cfg['initialization']['path'])
    assert sha256(Path(initial['adapter_path'])/'adapter_model.safetensors') == initial['adapter_sha256']
    assert initial['lora_rank'] == 32 and initial['lora_alpha'] == 64
    assert cfg['sampling']['n'] == 4 and cfg['sampling']['max_tokens'] == 1024
    assert not any(cfg[k] for k in ('kl_reward','kl_loss','reference_worker','critic','entropy_coefficient'))
    return initial


def prepare(purpose, run_class, cfg):
    validate(cfg)
    state = read(ROOT/'project_state.json')
    assert all(state['stages'][str(i)]['status'] == 'DONE' for i in (1,2,3))
    assert all(state['stages'][str(i)]['status'] == 'NOT_STARTED' for i in (5,6))
    assert read(INDEX/'prerequisite-stage3.json')['result'] == 'PASS'
    name = 's4_'+purpose+'_'+time.strftime('%Y%m%dT%H%M%S', time.gmtime())+'_'+uuid.uuid4().hex[:6]
    out = BULK/name
    out.mkdir()
    git = INDEX/name
    git.mkdir()
    immutable(out/'config.json', cfg)
    sources = source_hashes()
    with zipfile.ZipFile(out/'source.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sources:
            z.write(ROOT/p, p)
    manifest = dict(run_id=name, stage=4, run_class=run_class, purpose=purpose,
                    artifact_root=str(out), config_sha256=sha256(out/'config.json'),
                    source_hashes=sources, source_archive=record(out/'source.zip'),
                    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    dirty_state=subprocess.check_output(['git','status','--porcelain'],text=True),
                    created_at=now(), config=cfg)
    immutable(out/'manifest.json',manifest)
    immutable(git/'manifest.json',manifest)
    durable(out/'status.json',dict(status='PREPARED',timestamp=now()))
    (out/'observations.md').write_text('# Observations\n\nNo experimental results yet.\n')
    return out


def check_sources(out):
    manifest = read(Path(out)/'manifest.json')
    assert sha256(Path(out)/'config.json') == manifest['config_sha256']
    for p, digest in manifest['source_hashes'].items():
        # Runtime modules are frozen; reports/other stage launchers may be added independently.
        if p.startswith('src/medical_posttrain/rl/') or p=='scripts/run_stage4.py':
            assert sha256(ROOT/p) == digest, p


def seal(out, summary):
    out = Path(out)
    durable(out/'summary.json',summary)
    durable(out/'status.json',dict(status=summary['status'],timestamp=now()))
    durable(INDEX/out.name/'summary.json',summary)
    durable(INDEX/out.name/'artifacts_manifest.json',
            [record(p) for p in sorted(out.rglob('*')) if p.is_file()])


def encounter(stream, index, run_id, policy_version):
    # Identical request randomness for shared encounters across the pair, even
    # after policy divergence; stream.order is inherited and never blacklists.
    import hashlib
    e = stream.encounter(index,run_id,policy_version)
    e['request_seed'] = int.from_bytes(hashlib.sha256(
        f'{stream.seed}:{stream.domain}:{index}:{e["prompt_id"]}'.encode()).digest()[:4], 'big')
    return e
