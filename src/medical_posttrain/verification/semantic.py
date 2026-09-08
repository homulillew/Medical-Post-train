import gc
import json
import time
from pathlib import Path
from medical_posttrain.evidence import write_json,sha256

MODELS=[('abhinand/MedEmbed-small-v0.1','40a5850d046cfdb56154e332b4d7099b63e8d50e'),('BAAI/bge-m3','5617a9f61b028005a4858fdac845db406aefb181')]

def run(ev):
    import torch,psutil
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(8)
    anchor='患者有糖尿病，每日口服二甲双胍500毫克两次，无青霉素过敏史。'
    pairs=[('synonym','患者患糖尿病，每天服用二甲双胍两次，每次500毫克，没有青霉素过敏史。'),('unrelated','今天的天气晴朗，火车准时到达车站。'),('negation','患者无糖尿病，每日口服二甲双胍500毫克两次，有青霉素过敏史。'),('number','患者有糖尿病，每日口服二甲双胍5000毫克两次，无青霉素过敏史。'),('entity','患者有糖尿病，每日口服华法林500毫克两次，无青霉素过敏史。'),('repeated_fluff',anchor+'需要综合评估并密切观察。'*200),('long_truncation',anchor*600+'患者有青霉素过敏史。')]
    results=[]
    for name,revision in MODELS:
        dest=Path('/data/WSH/medical-post-train-artifacts/models')/name.replace('/','--')/revision
        t=time.monotonic()
        ignored=['onnx/*','openvino/*','*.onnx','*.h5','*.ot','*.msgpack']
        if name != 'BAAI/bge-m3': ignored.append('*.bin')
        snapshot_download(name,revision=revision,local_dir=dest,ignore_patterns=ignored,max_workers=4)
        files=[dict(path=str(p),size=p.stat().st_size,sha256=sha256(p)) for p in dest.rglob('*') if p.is_file() and '.cache' not in p.parts]
        write_json(ev.path/(name.split('/')[-1]+'_snapshot.json'),dict(model=name,revision=revision,files=files))
        model=SentenceTransformer(str(dest),device='cpu',local_files_only=True)
        load=time.monotonic()-t
        texts=[anchor]+[p[1] for p in pairs]
        lengths=[len(model.tokenizer.encode(s)) for s in texts]
        encoded_lengths=model.tokenize(texts)['attention_mask'].sum(dim=1).tolist()
        model.encode(['预热'],normalize_embeddings=True)
        t=time.monotonic(); vectors=model.encode(texts,normalize_embeddings=True,batch_size=1); elapsed=time.monotonic()-t
        scores=(vectors[1:]@vectors[0]).tolist()
        ranking=[pairs[i][0] for i in sorted(range(len(scores)),key=lambda i:scores[i],reverse=True)]
        row=ev.metric(event='semantic',model=name,revision=revision,cosines=dict(zip([p[0] for p in pairs],scores)),ranking=ranking,load_download_seconds=load,encoding_seconds=elapsed,seconds_per_text=elapsed/len(texts),cpu_rss_bytes=psutil.Process().memory_info().rss,max_seq_length=model.max_seq_length,untruncated_token_lengths=lengths,actual_encoded_lengths=encoded_lengths,synonym_beats_all_contradictions=all(scores[0]>scores[i] for i in (2,3,4)))
        results.append(row)
        if not row['synonym_beats_all_contradictions']:
            ev.case('semantic_contradiction_ranking','At least one negation/number/entity replacement ranks above the synonym',category='REWARD_CASE',metrics=row,texts=texts)
        del model,vectors; gc.collect()
    write_json(ev.path/'synthetic_pairs.json',dict(anchor=anchor,pairs=pairs))
    return dict(gates={'semantic_diagnostic':True},results=results,decision='Diagnostic only; no encoder replacement without decision record')
