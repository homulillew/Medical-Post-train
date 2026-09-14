#!/usr/bin/env python3
"""Freeze fresh validation clusters without reading sealed test contents."""
from collections import Counter
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,now,encounter
from medical_posttrain.data.exam import options_schema,canonical_answer,messages
from medical_posttrain.data.stage1 import text_hash
from medical_posttrain.sampling.dynamic import Stream
from frontier_diagnostic import gates,lines

def main():
    pair=gates();idx=ROOT/'experiments/stage4';cfg=read(pair['frozen_config']['path'])
    frontier=read(idx/'frontier_diagnostic_v1/manifest.json');random=read(ROOT/'experiments/stage5/ablation_eval_512_v1.json')
    for ref in [frontier['partition'],frontier['clusters'],frontier['source'],random['dataset']]:
        actual=record(ref['path']);assert all(actual[k]==ref[k] for k in actual)
    part=read(frontier['partition']['path']);membership={};test=set()
    for c in lines(frontier['clusters']['path']):
        membership.update({m:c['cluster_id'] for m in c['members']})
        if any(m.startswith(('cmexam_test:','cmb_test:')) for m in c['members']):test.add(c['cluster_id'])
    pool=list(lines(cfg['pool']['path']))
    excluded={'monitor512':{membership[p] for p in part['monitor']},'selection1024':{membership[p] for p in part['selection']},
        'frontier1000':{membership[p] for p in read(frontier['prompt_ids']['path'])},'random3x_eval512':{membership[p] for p in random['prompt_ids']},
        'rl_pool':{r['cluster_id'] for r in pool},'sft_train_validation':set(),'sealed_tests':test}
    for src in frontier['sft_exclusion_sources']:
        assert record(src['path'])==src
        excluded['sft_train_validation'].update(r['cluster_id'] for r in lines(src['path']))
    forbidden=set().union(*excluded.values())
    raw=list(csv.DictReader(open(frontier['source']['path'])))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    rows=[];seen=set();rejected=Counter()
    for pid in sorted(part['diagnostic_reserve'],key=lambda p:text_hash('20260914:loss_objective_eval:'+p)):
        cid=membership[pid]
        if cid in forbidden or cid in seen:rejected['excluded_or_duplicate_cluster']+=1;continue
        row=raw[int(pid.rsplit(':',1)[1])]
        try:
            opts=options_schema(row['Options']);ans=canonical_answer(row['Answer'],''.join(opts))
            assert row['Question'].strip()
            assert not any(t in row['Question']+row['Options'] for t in ['<think','<answer','<|im_','<|endoftext|>'])
            r=dict(prompt_id=pid,cluster_id=cid,question=row['Question'],options=opts,answer_set=ans,split='validation_auxiliary')
            ids=tok.apply_chat_template(messages(r),tokenize=True,return_dict=False,add_generation_prompt=True,enable_thinking=True)
            assert len(ids)+1024<=4096
        except (ValueError,AssertionError):rejected['invalid_schema_or_context']+=1;continue
        seen.add(cid);rows.append(r)
    available=len(rows);rows=rows[:512];assert len(rows)>=384,'BLOCKED: insufficient clean reserve'
    out=Path(read(idx/'grpo_diagnostic_preregistration_v1.json')['artifact_path'])
    immutable(out/'loss_objective_eval_dataset.json',rows)
    overlap={k:len({r['cluster_id'] for r in rows}&v) for k,v in excluded.items()};assert not any(overlap.values())
    manifest=dict(name='loss_objective_eval_512_v1',timestamp=now(),count=len(rows),eligible_clean_reserve=available,selection_seed=20260914,
        source=frontier['source'],partition=frontier['partition'],clusters=frontier['clusters'],dataset=record(out/'loss_objective_eval_dataset.json'),
        prompt_ids=[r['prompt_id'] for r in rows],cluster_ids=[r['cluster_id'] for r in rows],overlap_counts=overlap,
        excluded_sources=dict(frontier=record(idx/'frontier_diagnostic_v1/manifest.json'),random3x=record(ROOT/'experiments/stage5/ablation_eval_512_v1.json'),rl_pool=cfg['pool'],sft=frontier['sft_exclusion_sources']),
        rejections=dict(rejected),selection_rule='SHA256 text_hash(seed20260914:loss_objective_eval:prompt_id); first512 unique clean clusters',
        test_content_read=False,selection_content_read=False,selection_ids_only=True,evaluation_outputs_generated=0)
    immutable(idx/'loss_objective_eval_manifest_v1.json',manifest)
    immutable(idx/'loss_objective_eval_isolation_v1.json',dict(result='PASS',timestamp=now(),count=len(rows),unique_clusters=len({r['cluster_id'] for r in rows}),
        overlap_counts=overlap,manifest=record(idx/'loss_objective_eval_manifest_v1.json'),verifier=record(Path(__file__)),sealed_test_content_read=False))
    stream=Stream([r['prompt_id'] for r in pool],cfg['seed'],cfg['stream_domain']);schedule=[]
    for i in range(64):
        w=Path(pair['runs']['vanilla']['path'])/'windows'/f'{i:04d}'
        gg=[g for b in sorted((w/'batches').iterdir()) for g in read(b/'scored.json')['groups']]
        assert len(gg)==8
        es=[]
        for g in gg:
            expected=encounter(stream,len(schedule)*8+len(es),'auxiliary','policy')
            keys=['prompt_id','encounter_index','candidate_epoch','candidate_cursor','request_seed']
            assert all(g[k]==expected[k] for k in keys)
            es.append({k:g[k] for k in keys})
        schedule.append(dict(window_index=i,encounters=es,source_commit=record(w/'commit.json')))
    immutable(idx/'loss_objective_vanilla_schedule_v1.json',dict(timestamp=now(),windows=schedule,groups=512,stream_domain=cfg['stream_domain'],seed=cfg['seed'],fresh_responses_required=True))
    print(dict(count=len(rows),clean_available=available,overlap=overlap,schedule_groups=512))
if __name__=='__main__':main()
