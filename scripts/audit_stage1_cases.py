"""Retain cleaning cases and resolve rejection edges without altering frozen data."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import uuid
from medical_posttrain.data.medical import medical_o1,huatuo
from medical_posttrain.data.stage1 import encode_conversation
from medical_posttrain.evidence import sha256,write_json

def main():
    from transformers import AutoTokenizer
    root=Path('experiments/stage1/s1_data_20260908T144854_bac3a7');data=json.loads((root/'manifest.json').read_text());bulk=Path(data['artifact_root'])
    run_id='s1_data_audit_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')+'_'+uuid.uuid4().hex[:6]
    out=bulk.parent/run_id;out.mkdir(exist_ok=False);git=Path('experiments/stage1')/run_id;git.mkdir(exist_ok=False)
    write_json(git/'manifest.json',dict(run_id=run_id,stage=1,run_class='DIAGNOSTIC',purpose='Resolve retained duplicate edges and inspect source-length extremes, without changing frozen selection',source_data_run=data['run_id'],source_rejections_sha256=sha256(bulk/'rejections.jsonl'),script_sha256=sha256(__file__),command=sys.argv,artifact_root=str(out)))
    adjacent={};edge_counts=Counter()
    for line in (bulk/'duplicate_edges.jsonl').open():
        edge=json.loads(line);edge_counts[edge['reason']]+=1
        for a,b in [('left','right'),('right','left')]:adjacent.setdefault(edge[a],dict(matched_edge_target=edge[b],edge_reason=edge['reason'],similarity=edge['similarity']))
    rejections=[json.loads(line) for line in (bulk/'rejections.jsonl').open()]
    with (out/'rejections_resolved.jsonl').open('x') as f:
        for r in rejections:
            if r['reason'] in ('benchmark_cluster_overlap','sft_duplicate_cluster'):
                assert r['sample_id'] in adjacent
                r=dict(r,**adjacent[r['sample_id']])
            f.write(json.dumps(r,ensure_ascii=False)+'\n')
    raw={source:json.load(open(bulk.parent/'raw'/f'{source}.json')) for source in ('medical_o1','huatuo')}
    tokenizer=AutoTokenizer.from_pretrained(data['model'],local_files_only=True)
    cases=[];extremes=[]
    for source,adapter in [('medical_o1',medical_o1),('huatuo',huatuo)]:
        ranked=sorted(enumerate(raw[source]),key=lambda item:len(json.dumps(item[1],ensure_ascii=False)),reverse=True)[:5]
        for index,record in ranked:
            try:
                messages=adapter(record);encoded=encode_conversation(tokenizer,messages)
                row=dict(source=source,source_row=index,selection='five longest source records by serialized character count; diagnostic only',**{k:v for k,v in encoded.items() if k not in ('input_ids','labels')})
                extremes.append(row)
            except ValueError as error:extremes.append(dict(source=source,source_row=index,error=str(error)))
    with (out/'length_extremes.jsonl').open('x') as f:
        for r in extremes:f.write(json.dumps(r)+'\n')
    for source in ('medical_o1','huatuo'):
        for reason in ('encoding_corruption','extreme_repetition','benchmark_cluster_overlap','sft_duplicate_cluster'):
            match=next((r for r in rejections if r['source']==source and r['reason']==reason),None)
            if not match:continue
            record=raw[source][int(match['sample_id'].rsplit(':',1)[1])]
            question=record.get('Question') or record['conversations'][0]['value']
            cases.append(dict(case_id=f'S1-DATA-{len(cases)+1:03d}',run_id=run_id,stage=1,category='BAD_CASE',subtype=reason,prompt_id=match['sample_id'],prompt_excerpt=question[:350],rejection=match,observation='This source record was rejected by the frozen governance rule.',hypothesis=None,alternative_explanations=['Lexical similarity is conservative and may remove related but distinct questions.'] if 'cluster' in reason else [],followup='Retained raw row and rejection/graph evidence; no change to frozen train/validation.',artifact_refs=[str(bulk/'rejections.jsonl'),str(bulk/'duplicate_edges.jsonl')]))
    for r in extremes:
        if r.get('total_tokens',0)>2048:
            cases.append(dict(case_id=f'S1-DATA-{len(cases)+1:03d}',run_id=run_id,stage=1,category='BAD_CASE',subtype='overlength_source_answer_truncation_risk',prompt_id=f'{r["source"]}:{r["source_row"]}',observation=f'Actual full encoding has {r["total_tokens"]} tokens. Not a selected train/validation record; hard truncation at 2048 would risk losing target content.',hypothesis=None,alternative_explanations=[],followup='Reject over-cap candidates before selection; never truncate final answers.',artifact_refs=[str(out/'length_extremes.jsonl')],metrics=r))
    write_json(git/'cases.json',cases)
    summary=dict(run_id=run_id,status='PASS',duplicate_edge_counts=dict(edge_counts),rejections_resolved=len(rejections),case_count=len(cases),extremes=extremes,not_observed=['selected example over cap','selected final-answer truncation','real Huatuo multi-turn'],frozen_data_changed=False)
    write_json(git/'summary.json',summary);write_json(git/'artifacts_manifest.json',[dict(path=str(p),sha256=sha256(p),bytes=p.stat().st_size) for p in out.iterdir() if p.is_file()]);print(json.dumps(summary))

if __name__=='__main__':main()
