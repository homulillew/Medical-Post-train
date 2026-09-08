"""Recompute fixed Stage 1 data gates from raw records and retained graph evidence."""
import argparse
from collections import Counter
import json
from pathlib import Path
from medical_posttrain.data.medical import medical_o1,huatuo
from medical_posttrain.data.stage1 import normalize_question,text_hash
from medical_posttrain.data.stage1 import encode_conversation
from medical_posttrain.evidence import sha256,write_json

def verify(root):
    root=Path(root);summary=json.loads((root/'summary.json').read_text());manifest=json.loads((root/'manifest.json').read_text())
    bulk=Path(manifest['artifact_root']);artifacts=json.loads((root/'artifacts_manifest.json').read_text())
    for artifact in artifacts: assert sha256(artifact['path'])==artifact['sha256']
    raw=json.loads((bulk.parent/'raw_manifest.json').read_text())
    assert sha256(bulk.parent/'raw_manifest.json')==manifest['raw_manifest_sha256']
    for source in raw['sources']:assert sha256(source['path'])==source['sha256']
    benchmark=[json.loads(line) for line in Path(raw['exclusion']['path']).open()]
    assert Counter(r['source'] for r in benchmark)==dict(cmexam_train=54497,cmexam_val=6811,cmexam_test=6811,cmb_test=11200)
    assert all(set(r)=={'sample_id','source','source_row','question'} for r in benchmark)
    clusters={}
    for line in (bulk/'clusters.jsonl').open():
        r=json.loads(line);clusters[r['cluster_id']]=r
    all_ids=set();all_clusters=set();all_questions=set();counters={};token_sums={}
    raw_sources={m['source']:json.load(open(m['path'])) for m in raw['sources'] if m['source'] in ('medical_o1','huatuo')}
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(manifest['model'],local_files_only=True)
    for split,expected in [('train',10000),('validation',500)]:
        counts=Counter();totals=Counter();ids=[]
        with (bulk/f'{split}.jsonl').open() as f:
            for line in f:
                r=json.loads(line);source=r['source'];counts[source]+=1
                assert r['sample_id'] not in all_ids and r['cluster_id'] not in all_clusters and r['question_hash'] not in all_questions
                assert r['question_hash']==text_hash(normalize_question(r['question']))
                cluster=clusters[r['cluster_id']]
                assert not cluster['benchmark_members'] and r['sample_id'] in cluster['members']
                origin=raw_sources[source][r['source_row']]
                expected_messages=(medical_o1 if source=='medical_o1' else huatuo)(origin)
                assert r['messages']==expected_messages
                rebuilt=encode_conversation(tokenizer,expected_messages)
                assert all(r[key]==value for key,value in rebuilt.items()), 'Native tokenizer/assistant mask differs from frozen token cache'
                all_ids.add(r['sample_id']);all_clusters.add(r['cluster_id']);all_questions.add(r['question_hash']);ids.append(r['sample_id'])
                assert 0<len(r['input_ids'])==len(r['labels'])==r['total_tokens']<=2048
                assert all(label in (-100,token) for token,label in zip(r['input_ids'],r['labels']))
                assert r['supervised_tokens']==sum(x!=-100 for x in r['labels'])>0
                assert r['labels'].count(151645)==r['assistant_turns']
                if source=='huatuo':assert r['reasoning_tokens']==0
                for k in ('total_tokens','supervised_tokens','reasoning_tokens','answer_tokens'):totals[k]+=r[k]
        assert counts==dict(medical_o1=expected,huatuo=expected)
        assert ids==[r['sample_id'] for r in json.loads((bulk/f'{split}_ids.json').read_text())]
        for k,total in totals.items():assert summary['token_statistics'][split]['all'][k]['total']==total
        counters[split]=dict(counts);token_sums[split]=dict(totals)
    assert summary['status']=='PASS' and summary['train_count']==20000 and summary['validation_count']==1000
    return dict(status='PASS',run_id=manifest['run_id'],checks=['all artifact/source hashes','complete question-only benchmark projection','unique train/validation IDs and clusters','no benchmark-containing selected cluster','original source-to-messages parity','all 21000 native template/token/label reconstructions','complete labels and supervised EOS','exact balanced quotas','recomputed token totals'],counts=counters,token_sums=token_sums,artifact_root=str(bulk),data_manifest_sha256=sha256(root/'artifacts_manifest.json'),verifier_sha256=sha256(__file__),limitations=['lexical near-duplicate detection does not prove absence of semantic paraphrases','two official punctuation-only train questions have no usable lexical content'])

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--output');args=parser.parse_args()
    output=Path(args.output) if args.output else Path(args.run)/'data_verification.json'
    assert not output.exists(),'Retain earlier receipts; choose a new --output path'
    receipt=verify(args.run);write_json(output,receipt);print(json.dumps(receipt))
