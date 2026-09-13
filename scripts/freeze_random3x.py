#!/usr/bin/env python3
"""Freeze full auxiliary schedule, source-only selection and fresh isolated evaluation."""
from collections import Counter
import csv
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from medical_posttrain.rl.common import read,record,immutable,durable,prepare,now,encounter
from medical_posttrain.data.exam import options_schema,canonical_answer,messages
from medical_posttrain.data.stage1 import text_hash
from medical_posttrain.sampling.dynamic import Stream
from frontier_diagnostic import gates,lines
from random3x_runtime import selected_encounters


def main():
    pair=gates();idx=ROOT/'experiments/stage5';protocol=idx/'aux_random3x_protocol_v2.json'
    assert not protocol.exists()
    cfg=read(pair['frozen_config']['path'])
    schedule_path=idx/'aux_random3x_dynamic_schedule_v1.json';schedule=read(schedule_path)
    pool=list(lines(cfg['pool']['path']));stream=Stream([r['prompt_id'] for r in pool],cfg['seed'],cfg['stream_domain'])
    frozen_selections=[];cursor=0
    for i,w in enumerate(schedule['windows']):
        for src in w['sources']+[w['commit']]:assert record(src['path'])==src
        assert len(w['encounters'])==w['generated_groups']>=8 and w['generated_groups']%8==0
        for e in w['encounters']:
            expected=encounter(stream,cursor,'schedule','policy')
            assert all(e[k]==expected[k] for k in ['prompt_id','encounter_index','candidate_epoch','candidate_cursor','request_seed'])
            cursor+=1
        frozen_selections.append(dict(window_index=i,selected=[dict(encounter_index=gi,prompt_id=pid) for gi,pid in selected_encounters(w['encounters'],i)]))
    assert len(schedule['windows'])==64 and cursor==1360
    frontier=read(ROOT/'experiments/stage4/frontier_diagnostic_v1/manifest.json')
    assert record(frontier['partition']['path'])==frontier['partition']
    part=read(frontier['partition']['path']);membership={};test=set()
    assert record(frontier['clusters']['path'])==frontier['clusters']
    for c in lines(frontier['clusters']['path']):
        membership.update({m:c['cluster_id'] for m in c['members']})
        if any(m.startswith(('cmexam_test:','cmb_test:')) for m in c['members']):test.add(c['cluster_id'])
    excluded={'monitor512':{membership[p] for p in part['monitor']},'selection1024':{membership[p] for p in part['selection']},
        'frontier1000':{membership[p] for p in read(frontier['prompt_ids']['path'])},'rl_pool':{r['cluster_id'] for r in pool},
        'sft_train_validation':set(),'sealed_tests':test}
    for src in frontier['sft_exclusion_sources']:
        assert record(src['path'])==src
        excluded['sft_train_validation'].update(r['cluster_id'] for r in lines(src['path']))
    forbidden=set().union(*excluded.values())
    assert record(frontier['source']['path'])['sha256']==frontier['source']['sha256']
    raw=list(csv.DictReader(open(frontier['source']['path'])))
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True)
    rows=[];seen=set();rejected=Counter()
    for pid in sorted(part['diagnostic_reserve'],key=lambda p:text_hash('20260914:ablation_eval:'+p)):
        cid=membership[pid]
        if cid in forbidden or cid in seen:
            rejected['excluded_or_duplicate_cluster']+=1;continue
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
    available=len(rows);rows=rows[:512];assert len(rows)>0
    aux_cfg=dict(cfg,mode='auxiliary_ablation',sampling_mode='random3x',target_training_groups=512,pause_after_windows=4)
    out=prepare('aux_random3x','AUXILIARY_ABLATION',aux_cfg)
    immutable(out/'ablation_eval_dataset.json',rows)
    manifest=dict(name='ablation_eval_512_v1',timestamp=now(),count=len(rows),eligible_clean_reserve=available,selection_seed=20260914,
        source=frontier['source'],partition=frontier['partition'],clusters=frontier['clusters'],
        excluded_sources=dict(frontier_manifest=record(ROOT/'experiments/stage4/frontier_diagnostic_v1/manifest.json'),rl_pool=cfg['pool'],sft=frontier['sft_exclusion_sources']),
        dataset=record(out/'ablation_eval_dataset.json'),prompt_ids=[r['prompt_id'] for r in rows],
        overlap_counts={k:len({r['cluster_id'] for r in rows}&v) for k,v in excluded.items()},rejections=dict(rejected),
        limitation=None if len(rows)==512 else 'Insufficient clean reserve; maximum clean N retained without relaxing exclusions',
        test_content_read=False,selection_rule='Existing text_hash(SHA256) helper; seed20260914:ablation_eval:ID; first unique clean clusters',
        difficulty_category_stratification='Unavailable in frozen source schema')
    assert not any(manifest['overlap_counts'].values())
    immutable(idx/'ablation_eval_512_v1.json',manifest)
    immutable(idx/'aux_random3x_selected_identities_v2.json',frozen_selections)
    init=read(cfg['initialization']['path']);policies={'sft':dict(adapter_path=init['adapter_path'],adapter_sha256=init['adapter_sha256'])}
    for name,r in pair['runs'].items():
        w=Path(r['path'])/'windows/0063';commit=read(w/'commit.json')
        assert commit['state_after']['training_groups']==512 and commit['state_after']['optimizer_steps']==128
        policies[name]=dict(adapter_path=str(w/'checkpoint/adapter'),adapter_sha256=record(w/'checkpoint/adapter/adapter_model.safetensors')['sha256'])
    scripts=['random3x_runtime.py','verify_random3x.py','evaluate_random3x.py','finish_random3x.py','freeze_random3x.py','analyze_signal_density.py','analyze_frontier.py']
    p=dict(status='FROZEN_BEFORE_GENERATION',timestamp=now(),run_id=out.name,artifact_path=str(out),run_class='AUXILIARY_ABLATION',
        source_prior_protocol=record(idx/'aux_random3x_protocol_v1.json'),formal_pair=record(ROOT/'experiments/stage4/formal_pair.json'),
        config=aux_cfg,run_config=record(out/'config.json'),schedule=record(schedule_path),
        selected_identities=record(idx/'aux_random3x_selected_identities_v2.json'),selection_seed=20260914,
        selection_rule='SHA256(f"{selection_seed}:{zero_based_window_index}:{encounter_index}:{prompt_id}") ascending; first8; train in hash rank order',
        candidate_budget=dict(groups=cursor,windows=64,training_groups=512,optimizer_steps=128,training_trajectories=2048),
        selection_inputs=['window_index','encounter_index','prompt_id','selection_seed'],
        technical_retry_policy='Zero automatic retries. Any transport/incomplete group aborts the run and preserves evidence; no refill. Valid wrong, malformed-answer or low-quality completed outputs remain candidates.',
        generation='Fresh responses on same Dynamic first64 prompt IDs/order/counts/seeds. Matched generated-group schedule, NOT exact token matched.',
        operational_overrides=['online.select_groups=ScheduleSelector','online.monitor disabled (aux eval only after training)','audited restore_boundary','bounded actor GPU release guard'],
        real_resume=dict(pause_training_groups=32,pause_windows=4,optimizer_steps=8,method='SIGTERM owned worker process group; observe dead; fresh process reload same checkpoint; next commit step10'),
        evaluation_manifest=record(idx/'ablation_eval_512_v1.json'),evaluation_policies=policies,
        random3x_evaluation_endpoint='This run final512 only; no selection',evaluation_order=['sft','vanilla','random3x','dynamic'],
        evaluation_decoding=dict(n=1,temperature=0.,top_p=1.,top_k=-1,max_tokens=1024,seed=20260914),
        bootstrap_seed=20260914,bootstrap_resamples=10000,
        hypotheses={'selection_effect':'Dynamic512 exceeds Random3x512 on fresh matched auxiliary validation','exposure_effect':'Random3x512 exceeds Vanilla512'},
        interpretation_rule='Exploratory single-seed paired CI. No equivalence claim from nonsignificance. Mechanism SUPPORTED only if Dynamic-Random3x CI strictly positive; other patterns retained as PARTIALLY_SUPPORTED/NOT_SUPPORTED/INCONCLUSIVE with uncertainty.',
        project_state_snapshot=read(ROOT/'project_state.json'),project_state=record(ROOT/'project_state.json'),
        forbidden=['selection1024 inference','primary final tests','paid API','Stage6','primary retraining','formal source/config mutation'],
        runtime_sources={n:record(ROOT/'scripts'/n) for n in scripts},
        resources=dict(gpu='Single48GB',checkpoint_payload_estimate_bytes=64*1409125569,expected_training_hours='approximately4-6h plus verification and2048greedy eval responses; no budget reduction'),
        final_handoff='experiments/handoffs/stage4_causal_ablation_to_chatgpt_v1.json',narrative_author='ChatGPT/user')
    immutable(protocol,p);immutable(out/'auxiliary_protocol.json',p)
    durable(idx/'aux_random3x_status.json',dict(status='PREPARED',run_id=out.name,timestamp=now(),READY_FOR_STAGE5='NO'))
    print(dict(run_id=out.name,evaluation_count=len(rows),remaining_clean=available,generated_groups=cursor))


if __name__=='__main__':main()
