#!/usr/bin/env python3
"""Freeze and execute a disjoint validation diagnostic; never train or select a checkpoint."""
import argparse
from collections import Counter
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, durable, now
from medical_posttrain.data.exam import options_schema, canonical_answer, messages
from medical_posttrain.data.stage1 import text_hash

INDEX = ROOT / 'experiments/stage4/frontier_diagnostic_v1'
SEED = 20260913


def lines(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def gates():
    pair = read(ROOT / 'experiments/stage4/formal_pair.json')
    for rel, digest in pair['execution_hashes'].items():
        assert record(ROOT / rel)['sha256'] == digest, rel
    assert record(pair['frozen_config']['path']) == pair['frozen_config']
    for name, run in pair['runs'].items():
        receipt = read(ROOT / f'experiments/stage4/{name}_formal_verification.json')
        assert receipt['result'] == 'PASS' and receipt['state']['training_groups'] == 5000
        r = Path(run['path'])
        assert read(r / 'final_reload/result.json')['result'] == 'PASS'
        assert read(r / 'final_reload/exit.json')['exit_code'] == 0
    return pair


def freeze():
    pair = gates()
    assert not (INDEX / 'preregistration.json').exists(), 'Existing preregistration is immutable'
    cfg = read(pair['frozen_config']['path'])
    protocol = read(cfg['validation_protocol']['path'])
    assert record(protocol['partition']['path']) == protocol['partition']
    assert record(protocol['source']['path'])['sha256'] == protocol['source']['sha256']
    partition = read(protocol['partition']['path'])
    data_cfg = read(Path(cfg['pool']['path']).parent / 'config.json')
    cluster_path = Path(data_cfg['clusters'])
    assert record(cluster_path)['sha256'] == data_cfg['clusters_sha256']
    membership = {}
    test_clusters = set()
    for c in lines(cluster_path):
        membership.update({m: c['cluster_id'] for m in c['members']})
        if any(m.startswith(('cmexam_test:', 'cmb_test:')) for m in c['members']):
            test_clusters.add(c['cluster_id'])
    pool = list(lines(cfg['pool']['path']))
    assert record(cfg['pool']['path']) == cfg['pool']
    sft_paths = [cluster_path.parent / n for n in ('train.jsonl', 'validation.jsonl')]
    sft_cfg = read(ROOT / 'configs/stages/s1_formal.json')
    for file, kind in zip(sft_paths, ('train', 'validation')):
        assert str(file) == sft_cfg[kind+'_path']
        assert record(file)['sha256'] == sft_cfg[kind+'_sha256']
    forbidden = set(test_clusters)
    exclusion_sets = {
        'monitor512': {membership[p] for p in partition['monitor']},
        'selection1024': {membership[p] for p in partition['selection']},
        'rl_pool': {r['cluster_id'] for r in pool},
        'sft_train_validation': {r['cluster_id'] for p in sft_paths for r in lines(p)},
        'sealed_test_clusters': test_clusters,
    }
    for ids in exclusion_sets.values():
        forbidden.update(ids)
    raw = list(csv.DictReader(open(protocol['source']['path'])))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model'], local_files_only=True)
    candidates, rejected, seen = [], Counter(), set()
    for pid in sorted(partition['diagnostic_reserve'], key=lambda p: text_hash(f'{SEED}:frontier:{p}')):
        cid = membership[pid]
        if cid in forbidden:
            rejected['excluded_cluster'] += 1
            continue
        if cid in seen:
            rejected['duplicate_reserve_cluster'] += 1
            continue
        row = raw[int(pid.rsplit(':', 1)[1])]
        try:
            options = options_schema(row['Options'])
            answer = canonical_answer(row['Answer'], ''.join(options))
            assert row['Question'].strip()
            if any(t in row['Question'] + row['Options'] for t in ('<think', '<answer', '<|im_', '<|endoftext|>')):
                raise ValueError('reserved_control_token')
            r = dict(prompt_id=pid, question=row['Question'], options=options, answer_set=answer,
                     cluster_id=cid, split='val_diagnostic', source_revision=protocol['source']['revision'])
            prompt = tok.apply_chat_template(messages(r), tokenize=False, add_generation_prompt=True, enable_thinking=True)
            if len(tok.encode(prompt, add_special_tokens=False)) + 1024 > cfg['engine']['max_model_len']:
                raise ValueError('context_overflow')
        except (ValueError, AssertionError) as exc:
            rejected[str(exc) or 'empty_question'] += 1
            continue
        seen.add(cid)
        candidates.append(r)
    assert len(candidates) >= 1000, 'Insufficient clean reserve: never relax exclusions'
    rows = candidates[:1000]
    chosen = {r['cluster_id'] for r in rows}
    assert len(chosen) == len({r['prompt_id'] for r in rows}) == 1000
    overlaps = {k: len(chosen & s) for k, s in exclusion_sets.items()}
    assert not any(overlaps.values())
    run_id = 's4_frontier_diagnostic_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = Path(pair['runs']['dynamic']['path']).parent / run_id
    out.mkdir()
    INDEX.mkdir(parents=True, exist_ok=True)
    immutable(out / 'dataset.json', rows)
    ids = [r['prompt_id'] for r in rows]
    immutable(INDEX / 'prompt_ids.json', ids)
    manifest = dict(name='frontier_diagnostic_1000', seed=SEED, count=1000,
                    source=protocol['source'], partition=protocol['partition'], clusters=record(cluster_path),
                    sft_exclusion_sources=[record(p) for p in sft_paths], pool=cfg['pool'],
                    dataset=record(out / 'dataset.json'), prompt_ids=record(INDEX / 'prompt_ids.json'),
                    eligible_unique_clusters=len(candidates), exclusions=dict(rejected), overlap_counts=overlaps,
                    selection_rule='Existing text_hash helper: SHA256(seed:frontier:ID); first clean representative per cluster, first 1000',
                    stratification='Unavailable: frozen official CSV has no difficulty/category annotations; no inferred labels or model-based selection',
                    test_content_read=False, optimizer_updates=0)
    immutable(INDEX / 'manifest.json', manifest)
    init = read(cfg['initialization']['path'])
    policies = {'sft': dict(adapter_path=init['adapter_path'], adapter_sha256=init['adapter_sha256'])}
    for name, r in pair['runs'].items():
        adapter = Path(r['path']) / 'windows/0624/checkpoint/adapter'
        policies[name] = dict(adapter_path=str(adapter), adapter_sha256=record(adapter / 'adapter_model.safetensors')['sha256'])
    prereg = dict(timestamp=now(), run_id=run_id, artifact_path=str(out), status='FROZEN_BEFORE_GENERATION',
                 scope='validation diagnostic / exploratory mechanism evidence',
                 manifest=record(INDEX / 'manifest.json'), policies=policies, policy_order=['sft', 'vanilla', 'dynamic'],
                 sampling=cfg['sampling'], engine=cfg['engine'], seed=SEED, stream_domain='stage4:frontier:v1',
                 config=cfg, batch_groups=8, required_groups_per_policy=1000, required_responses=12000, optimizer_updates=0,
                 seed_note='Same encounter request seeds via frozen rl.common.encounter; no bitwise common-random-number claim across policies/processes',
                 hypotheses={
                     'H1': 'Among SFT mixed prompts, Dynamic5000 has higher all-correct resolution than Vanilla5000.',
                     'H2': 'Dynamic should not increase SFT mixed-to-all-wrong regression; report signed paired difference, no invented noninferiority margin.',
                     'H3': 'Report resolution separately for mixed_parsed_wrong, mixed_unparseable_only and mixed_both; compare subgroup effects without redefining eligibility.',
                     'H4': 'Final mixed fraction may decrease while all-correct increases; separately retain hard SFT all-wrong transitions.'},
                 bootstrap=dict(seed=SEED, resamples=10000, unit='prompt', paired=True, confidence=.95,
                                interval='percentile', subgroup='Resample prompt pairs within fixed SFT-defined subgroup; empty subgroup => null',
                                multiple_comparisons='Exploratory unadjusted intervals; not held-out test or training-seed inference'),
                 secondary_rules=dict(majority='Unique canonical answer plurality; ties and absent parsed answers count wrong',
                                      consistency='All four canonical answers identical and parsed',
                                      entropy='Shannon bits over canonical answers, unparseable represented by a separate category'),
                 failure_policy='Fail closed on missing/invalid response, hash mismatch, stale reservation or source mutation; valid wrong answers never retried',
                 scientific_sources={k: record(ROOT / k) for k in pair['execution_hashes']},
                 runner_sources={str(p.relative_to(ROOT)): record(p) for p in [Path(__file__), ROOT / 'scripts/analyze_frontier.py']})
    immutable(INDEX / 'preregistration.json', prereg)
    immutable(out / 'preregistration.json', prereg)
    durable(INDEX / 'status.json', dict(status='PREPARED', run_id=run_id, timestamp=now(), optimizer_updates=0))
    print(json.dumps(dict(run_id=run_id, eligible=len(candidates), overlaps=overlaps, manifest=record(INDEX / 'manifest.json'))))


def check_prereg():
    p = read(INDEX / 'preregistration.json')
    assert p == read(Path(p['artifact_path']) / 'preregistration.json')
    assert record(p['manifest']['path']) == p['manifest']
    manifest = read(p['manifest']['path'])
    for ref in [manifest['dataset'], manifest['prompt_ids'], manifest['partition']]:
        assert record(ref['path']) == ref
    for ref in list(p['scientific_sources'].values()) + list(p['runner_sources'].values()):
        assert record(ref['path']) == ref
    return p


def policy_worker(name):
    p = check_prereg()
    from medical_posttrain.rl.rollout import Rollout
    from medical_posttrain.sampling.dynamic import Stream
    from medical_posttrain.runtime import environment
    rows = read(read(p['manifest']['path'])['dataset']['path'])
    pool = {r['prompt_id']: r for r in rows}
    stream = Stream(list(pool), p['seed'], p['stream_domain'])
    out = Path(p['artifact_path']) / name
    out.mkdir(exist_ok=False)
    immutable(out / 'environment.json', environment())
    immutable(out / 'command.json', sys.argv)
    cfg = p['config']
    engine = Rollout(out, cfg, p['policies'][name])
    try:
        for start in range(0, 1000, p['batch_groups']):
            engine.generate(out / 'batches' / f'{start // p["batch_groups"]:04d}', stream, pool,
                            start, min(p['batch_groups'], 1000-start), p['run_id']+'_'+name,
                            record(INDEX / 'preregistration.json')['sha256'])
            durable(INDEX / 'status.json', dict(status='RUNNING', policy=name, completed_policy_groups=start+8,
                    timestamp=now(), worker_pid=os.getpid(), run_id=p['run_id'], optimizer_updates=0))
    finally:
        engine.close()
    immutable(out / 'generation_complete.json', dict(status='GENERATED', groups=1000, responses=4000, timestamp=now()))


def supervise():
    p = check_prereg()
    gates()
    out = Path(p['artifact_path'])
    with (ROOT.parent / 'stage4-gpu.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            for name in p['policy_order']:
                log = out / (name + '_process')
                log.mkdir()
                cmd = [sys.executable, str(Path(__file__)), 'policy', '--policy', name]
                with (log/'stdout.log').open('x') as so, (log/'stderr.log').open('x') as se:
                    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=so, stderr=se)
                    immutable(log/'launch.json', dict(command=cmd, pid=proc.pid, timestamp=now()))
                    while proc.poll() is None:
                        durable(INDEX/'heartbeat.json', dict(pid=os.getpid(), policy=name, child_pid=proc.pid, timestamp=now()))
                        time.sleep(10)
                    immutable(log/'exit.json', dict(exit_code=proc.returncode, timestamp=now()))
                    assert proc.returncode == 0, f'Policy {name} failed; no automatic regeneration'
                deadline = time.monotonic()+60
                while int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()) >= 100:
                    assert time.monotonic() < deadline, 'GPU teardown timeout'
                    time.sleep(.2)
            subprocess.run([sys.executable, str(ROOT/'scripts/analyze_frontier.py')], check=True, cwd=ROOT)
            durable(INDEX/'status.json', dict(status='DIAGNOSTIC_VERIFIED', timestamp=now(), run_id=p['run_id'],
                    optimizer_updates=0, stage4_done=False, stage5_ready=False))
        except BaseException as exc:
            immutable(out/f'failure_{time.time_ns()}.json', dict(error=repr(exc), traceback=traceback.format_exc(), timestamp=now()))
            durable(INDEX/'status.json', dict(status='FAILED', timestamp=now(), error=repr(exc), optimizer_updates=0))
            raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['freeze','supervise','policy'])
    ap.add_argument('--policy', choices=['sft','vanilla','dynamic'])
    a = ap.parse_args()
    if a.action == 'freeze': freeze()
    elif a.action == 'supervise': supervise()
    else: policy_worker(a.policy)


if __name__ == '__main__':
    main()
