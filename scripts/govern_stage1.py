"""Create an auditable, source-balanced Stage 1 dataset before any training."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from medical_posttrain.data.medical import medical_o1, huatuo
from medical_posttrain.data.stage1 import DuplicateGraph, encode_conversation, normalize_question, quality_reason, text_hash
from medical_posttrain.evidence import write_json, sha256
from medical_posttrain.runtime import environment

BULK = Path('/data/WSH/medical-post-train-artifacts/data/stage1')
MODEL = '/data/WSH/medical-post-train-artifacts/models/Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218'

def dump(f, row):
    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')

def stats(rows):
    import numpy as np
    result = {}
    for source in ('all', 'medical_o1', 'huatuo'):
        selected = [r for r in rows if source == 'all' or r['source'] == source]
        item = dict(count=len(selected))
        for k in ('total_tokens', 'supervised_tokens', 'reasoning_tokens', 'answer_tokens'):
            values = [r[k] for r in selected]
            item[k] = dict(total=sum(values), mean=float(np.mean(values)), **{f'p{p}':float(np.percentile(values, p)) for p in (50,90,95,99)}, max=max(values)) if values else None
        result[source] = item
    return result

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--run-id'); args=parser.parse_args()
    run_id=args.run_id or 's1_data_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=BULK/run_id; out.mkdir(exist_ok=False)
    git=Path('experiments/stage1')/run_id; git.mkdir(parents=True, exist_ok=False)
    started=time.time()
    metadata=dict(run_id=run_id, stage=1, run_class='DIAGNOSTIC', purpose='Full-source governance and fixed 20k selection', command=sys.argv, started=datetime.now(timezone.utc).isoformat(), git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(), dirty_state=subprocess.check_output(['git','status','--porcelain'],text=True), source_hashes={str(p):sha256(p) for base in ('src','scripts') for p in Path(base).rglob('*.py')}, raw_manifest_sha256=sha256(BULK/'raw_manifest.json'), model=MODEL, seed=42, planned_train_per_source=10000, planned_val_per_source=500, max_sequence_length=2048, near_rules=dict(candidate_char3_jaccard=.65, confirmation_char5_jaccard=.85, confirmation_sequence_matcher=.90, semantic_dedup=False), artifact_root=str(out))
    write_json(git/'manifest.json',metadata); write_json(out/'environment.json',environment())
    write_json(out/'status.json',dict(status='RUNNING'))
    try:
        rows=[json.loads(line) for line in (BULK/'benchmark_questions.jsonl').open()]
        benchmark_count=len(rows)
        counts=Counter(); rejects=[]
        raw_manifest=json.loads((BULK/'raw_manifest.json').read_text())
        for source, adapter in [('medical_o1',medical_o1),('huatuo',huatuo)]:
            meta=next(m for m in raw_manifest['sources'] if m['source']==source)
            raw=json.load(open(meta['path']))
            for i, record in enumerate(raw):
                counts[source+':raw']+=1
                sid=f'{source}:{meta["revision"]}:{i}'
                reason=None
                try:
                    messages=adapter(record)
                    reason=quality_reason(messages)
                except ValueError as e:
                    reason='schema_or_reserved_output:'+str(e)
                if reason:
                    rejects.append(dict(sample_id=sid,source=source,reason=reason,matched_target=None,similarity=None,cluster_id=None))
                    counts[source+':'+reason]+=1
                    continue
                question='\n'.join(m['content'] for m in messages if m['role']=='user')
                rows.append(dict(sample_id=sid,source=source,source_row=i,source_revision=meta['revision'],original_id=record.get('id'),question=question,messages=messages))
            del raw
        texts=[normalize_question(r['question']) for r in rows]
        anomalous_benchmarks=[]
        for i,text in enumerate(texts):
            if not text:
                assert i<benchmark_count, 'Empty normalized SFT question requires explicit quality rejection'
                anomalous_benchmarks.append(dict(sample_id=rows[i]['sample_id'],question=rows[i]['question'],reason='official_question_contains_only_punctuation',action='retained_in_exclusion_manifest; no meaningful lexical match available'))
                texts[i]='unusablebenchmark'+text_hash(rows[i]['sample_id'])
        write_json(out/'benchmark_anomalies.json',anomalous_benchmarks)
        print(json.dumps(dict(event='loaded',records=len(rows),benchmarks=benchmark_count,counts=dict(counts))),flush=True)
        graph=DuplicateGraph(texts)
        edge_for={}
        with (out/'duplicate_edges.jsonl').open('x') as edges:
            def edge(i,j,reason,sim):
                dump(edges,dict(left=rows[i]['sample_id'],right=rows[j]['sample_id'],reason=reason,similarity=sim))
                edge_for[i]=(j,reason,sim)
            graph.join(edge,lambda n: print(json.dumps(dict(event='dedup_progress',processed=n,total=len(rows))),flush=True))
        clusters=defaultdict(list)
        for i in range(len(rows)):
            clusters[graph.root(i)].append(i)
        chosen=[]
        with (out/'clusters.jsonl').open('x') as f:
            for members in clusters.values():
                cluster_id='cluster:'+text_hash('\n'.join(sorted(rows[i]['sample_id'] for i in members)))
                benchmark=[i for i in members if i<benchmark_count]
                candidates=[i for i in members if i>=benchmark_count]
                dump(f,dict(cluster_id=cluster_id,members=[rows[i]['sample_id'] for i in members],benchmark_members=len(benchmark)))
                if not candidates: continue
                # Reasoning-source priority protects the required smaller source quota;
                # tie-break uses seeded hashes, independent of model performance.
                keep=None if benchmark else min(candidates,key=lambda i:(rows[i]['source']!='medical_o1',text_hash('42:'+rows[i]['sample_id'])))
                for i in candidates:
                    r=rows[i]; r['cluster_id']=cluster_id; r['question_hash']=text_hash(texts[i])
                    if i==keep:
                        chosen.append(r); counts[r['source']+':clean_unique']+=1
                    else:
                        reason='benchmark_cluster_overlap' if benchmark else 'sft_duplicate_cluster'
                        target=rows[benchmark[0] if benchmark else keep]['sample_id']
                        direct=edge_for.get(i)
                        rejects.append(dict(sample_id=r['sample_id'],source=r['source'],reason=reason,matched_target=target,similarity=direct[2] if direct else None,similarity_scope='direct_edge; cluster target may be transitive',matched_edge_target=rows[direct[0]]['sample_id'] if direct else None,cluster_id=cluster_id))
                        counts[r['source']+':'+reason]+=1
        with (out/'clean_candidates.jsonl').open('x') as f:
            for r in chosen: dump(f,r)
        print(json.dumps(dict(event='dedup_complete',counts=dict(counts))),flush=True)
        del graph, rows, texts, clusters
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(MODEL,local_files_only=True)
        train=[];val=[];scanned=[]
        with (out/'train.jsonl').open('x') as tf, (out/'validation.jsonl').open('x') as vf:
            for source in ('medical_o1','huatuo'):
                candidates=sorted((r for r in chosen if r['source']==source),key=lambda r:text_hash('42:split:'+r['sample_id']))
                accepted=0
                for r in candidates:
                    encoded=encode_conversation(tokenizer,r['messages'])
                    compact={k:v for k,v in encoded.items() if k not in ('input_ids','labels')}
                    scanned.append(dict(sample_id=r['sample_id'],source=source,**compact))
                    if encoded['total_tokens']>2048:
                        rejects.append(dict(sample_id=r['sample_id'],source=source,reason='sequence_over_2048_no_truncation',matched_target=None,similarity=None,cluster_id=r['cluster_id'],**compact))
                        counts[source+':sequence_over_2048_no_truncation']+=1
                        continue
                    split='validation' if accepted<500 else 'train'
                    record=dict(**r,**encoded,split=split)
                    dump(vf if split=='validation' else tf,record)
                    (val if split=='validation' else train).append(dict(sample_id=r['sample_id'],source=source,cluster_id=r['cluster_id'],**compact))
                    accepted+=1
                    if accepted>=10500: break
                counts[source+':selected']=accepted
        with (out/'rejections.jsonl').open('x') as f:
            for r in rejects: dump(f,r)
        write_json(out/'train_ids.json',train);write_json(out/'validation_ids.json',val)
        token_stats=dict(train=stats(train),validation=stats(val),selection_candidates=stats(scanned),overlength_policy='reject complete example, never truncate; only scanned selection candidates tokenized')
        write_json(out/'token_statistics.json',token_stats)
        success=len(train)==20000 and len(val)==1000
        status='PASS' if success else 'BLOCKED'
        summary=dict(run_id=run_id,status=status,counts=dict(counts),benchmark_count=benchmark_count,train_count=len(train),validation_count=len(val),planned_train_count=20000,elapsed_seconds=time.time()-started,token_statistics=token_stats,reason=None if success else 'Insufficient clean source quota after fixed leakage/quality/length rules. No training authorized on an incomplete quota.')
        artifacts=[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(out.iterdir()) if p.is_file() and p.name!='status.json']
        write_json(git/'artifacts_manifest.json',artifacts);write_json(git/'summary.json',summary)
        write_json(out/'status.json',dict(status=status,ended=datetime.now(timezone.utc).isoformat()))
        print(json.dumps(summary),flush=True)
    except BaseException:
        write_json(out/'status.json',dict(status='FAILED',traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
