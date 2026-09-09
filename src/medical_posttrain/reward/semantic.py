"""Fixed, batch-independent chunked embedding cosine; not a clinical judge."""
from collections import Counter
import gc
import json
from pathlib import Path
import time
from medical_posttrain.evidence import write_json
from medical_posttrain.evidence.stage2 import jsonlines, dump_lines, record
from medical_posttrain.verification.semantic import MODELS

MODEL_ROOT = Path('/data/WSH/medical-post-train-artifacts/models')
CHUNK_POLICY = dict(version=1, max_content_tokens=480, overlap_tokens=0,
                    segmentation='contiguous encoder token IDs, special tokens added per chunk',
                    pooling='content-token-length weighted mean of L2-normalized chunk embeddings; L2 normalize again',
                    mapping='clip(cosine,0,1)', empty_reasoning=0, missing_reference=0,
                    truncation='none: every encoder content token retained')


class Encoder:
    def __init__(self, model_id, revision, device='cpu'):
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(8)
        self.path = MODEL_ROOT/model_id.replace('/','--')/revision
        self.model_id, self.revision = model_id, revision
        self.model = SentenceTransformer(str(self.path), device=device, local_files_only=True)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        self.device = device
        assert 480+self.tokenizer.num_special_tokens_to_add(False) <= self.model.max_seq_length

    def encode(self, texts, batch_size=16):
        import numpy as np
        import torch
        chunks, owners, weights, metadata = [], [], [], []
        for i,text in enumerate(texts):
            encoding=self.tokenizer.backend_tokenizer.encode(text, add_special_tokens=False)
            ids=encoding.ids
            offsets=[]
            if text.strip():
                encoding.truncate(480, stride=0, direction='right')
                pieces=[encoding]+encoding.overflowing
                assert [t for piece in pieces for t in piece.ids] == ids
                for start,piece in zip(range(0,len(ids),480),pieces):
                    content=piece.ids
                    chunks.append(self.tokenizer.backend_tokenizer.post_process(piece, add_special_tokens=True).ids)
                    owners.append(i); weights.append(len(content)); offsets.append([start,start+len(content)])
            metadata.append(dict(encoder_tokens=len(ids), chunk_count=len(offsets), chunk_spans=offsets,
                                 truncated_tokens=0, unk_tokens=sum(t==self.tokenizer.unk_token_id for t in ids),
                                 reason='empty_text' if not text.strip() else 'encoded'))
        vectors=[]
        with torch.inference_mode():
            for start in range(0,len(chunks),batch_size):
                batch=chunks[start:start+batch_size]; length=max(map(len,batch))
                ids=torch.full((len(batch),length),self.tokenizer.pad_token_id,device=self.device,dtype=torch.long)
                mask=torch.zeros_like(ids)
                for i,seq in enumerate(batch):
                    ids[i,:len(seq)]=torch.tensor(seq,device=self.device);mask[i,:len(seq)]=1
                output=self.model(dict(input_ids=ids,attention_mask=mask))['sentence_embedding'].float()
                output=torch.nn.functional.normalize(output,p=2,dim=1)
                vectors.extend(output.cpu().numpy())
        dim=self.model.get_sentence_embedding_dimension()
        result=np.zeros((len(texts),dim),dtype=np.float32)
        for owner,weight,vector in zip(owners,weights,vectors):result[owner]+=weight*vector
        norm=np.linalg.norm(result,axis=1,keepdims=True)
        result/=np.maximum(norm,1e-12)
        assert np.isfinite(result).all()
        return result,metadata


def controlled_pairs(pool):
    """12 explicit synthetic logical fixtures ×15, plus 20 train-derived pairs.

    Labels describe controlled textual consistency, not prescribing validity.
    Every fixture's comparison conclusion is artificial; no patient advice.
    """
    pairs=[]
    scenarios=[('糖尿病','二甲双胍','高血压','阿司匹林'),('肺炎','阿莫西林','哮喘','华法林'),
               ('高血压','氨氯地平','糖尿病','地高辛'),('哮喘','沙丁胺醇','肺炎','布洛芬'),
               ('贫血','硫酸亚铁','甲亢','泼尼松'),('甲减','左甲状腺素','贫血','氯雷他定'),
               ('胃炎','奥美拉唑','肝炎','胰岛素'),('心衰','呋塞米','胃炎','利福平'),
               ('痛风','别嘌醇','心衰','阿托品'),('癫痫','丙戊酸钠','偏头痛','甲硝唑'),
               ('过敏性鼻炎','氯雷他定','肺炎','卡托普利'),('甲亢','甲巯咪唑','甲减','青霉素')]
    for i,(disease,drug,other_disease,other_drug) in enumerate(scenarios):
        age=28+i*3; dose=10+i*5; duration=5+i; sex='男' if i%2==0 else '女'; opposite='女' if sex=='男' else '男'
        anchor=f'记录显示，患者{age}岁，{sex}性，有{disease}，无药物过敏。原方案是每次口服{drug}{dose}毫克，每日两次，连续{duration}天。因出现不良反应，现已停止该药。按照题目给定规则，结论为C。'
        positive=f'患者为{age}岁的{sex}性，患有{disease}，没有药物过敏史。此前每天服用两次{drug}，每次{dose}毫克，疗程{duration}天。由于发生了不良反应，现已停用。根据本题所给规则，最终选择C。'
        variants={
            'synonym':positive,
            'unrelated':'火车抵达车站后，旅客依次下车，随后工作人员清扫站台。',
            'negation':anchor.replace('无药物过敏','有药物过敏'),
            'dose_number':anchor.replace(f'{dose}毫克',f'{dose*10}毫克'),
            'age':anchor.replace(f'{age}岁',f'{age+40}岁'),
            'sex':anchor.replace(f'{sex}性',f'{opposite}性'),
            'disease_entity':anchor.replace(disease,other_disease),
            'drug_entity':anchor.replace(drug,other_drug),
            'treatment_direction':anchor.replace('现已停止该药','现应继续该药'),
            'duration':anchor.replace(f'{duration}天',f'{duration*10}天'),
            'causality':anchor.replace('因出现不良反应，现已停止该药','因停止该药，现已出现不良反应'),
            'repeated_fluff':'需要结合具体情况综合判断。'*50,
            'question_only':f'患者{age}岁，{sex}性，有{disease}，曾服用{drug}，请问应选择哪个选项？',
            'correct_conclusion_wrong_explanation':anchor.replace('无药物过敏','有药物过敏').replace('现已停止该药','现应继续该药'),
            'wrong_conclusion_high_overlap':anchor.replace('结论为C','结论为D'),
        }
        for kind,candidate in variants.items():
            pairs.append(dict(pair_id=f'synthetic_{i:02d}_{kind}',anchor_id=f'synthetic_{i:02d}',category=kind,
                              anchor=anchor,candidate=candidate,positive=kind=='synonym',source='synthetic_controlled',
                              label_scope='textual perturbation; not medically validated dosing or clinical facts',
                              alteration=kind,expected_similarity_order='synonym > inconsistent or irrelevant candidate'))
    usable=[r for r in pool if r['reference_explanation'].strip()][:10]
    for i,row in enumerate(usable):
        anchor=row['reference_explanation']
        for kind,candidate,positive in [('train_surface_paraphrase',anchor.replace('本题','此题').replace('因此','所以').replace('患者','病人'),True),
                                        ('train_unrelated','星空中的猎户座在冬季夜晚十分醒目，天文台记录了恒星的位置。',False)]:
            pairs.append(dict(pair_id=f'train_{i:02d}_{kind}',anchor_id=row['prompt_id'],category=kind,anchor=anchor,
                              candidate=candidate,positive=positive,source='CMExam_train',source_row=row['source_row'],
                              source_revision=row['source_revision'],label_scope='surface invariance or unrelated text; source explanation not clinically adjudicated'))
    assert len(pairs)==200
    return pairs


def diagnostic(run):
    import numpy as np
    import psutil
    pairs=controlled_pairs(jsonlines(run.config['pool']))
    dump_lines(run.out/'controlled_pairs.jsonl',pairs)
    texts=list(dict.fromkeys(t for p in pairs for t in (p['anchor'],p['candidate'])))
    index={t:i for i,t in enumerate(texts)}
    results=[]
    for name,revision in MODELS:
        start=time.monotonic();encoder=Encoder(name,revision,run.config['device']);load=time.monotonic()-start
        fixture=['预热','患者没有药物过敏史。']
        ours,_=encoder.encode(fixture)
        native=encoder.model.encode(fixture,normalize_embeddings=True)
        native_error=float(np.max(np.abs(ours-native)))
        assert native_error < 1e-5, 'Short-text encoder differs from native SentenceTransformer'
        start=time.monotonic();vectors,lengths=encoder.encode(texts);elapsed=time.monotonic()-start
        scores=[]
        for p in pairs:
            cosine=float(vectors[index[p['anchor']]]@vectors[index[p['candidate']]])
            scores.append(dict(**p,cosine=cosine,semantic=float(np.clip(cosine,0,1))))
        label=name.split('/')[-1]
        dump_lines(run.out/f'{label}_scores.jsonl',scores)
        np.save(run.out/f'{label}_vectors.npy',vectors)
        write_json(run.out/f'{label}_encoding.json',dict(texts=texts,metadata=lengths,chunk_policy=CHUNK_POLICY))
        pos=[p['cosine'] for p in scores if p['positive']];neg=[p['cosine'] for p in scores if not p['positive']]
        positives={p['anchor_id']:p['cosine'] for p in scores if p['positive']}
        bycat={}
        for kind in sorted({p['category'] for p in scores}):
            rows=[p for p in scores if p['category']==kind];values=[p['cosine'] for p in rows]
            bycat[kind]=dict(count=len(rows),mean=float(np.mean(values)),min=min(values),max=max(values),
                            synonym_ranks_higher_fraction=None if rows[0]['positive'] else sum(positives[p['anchor_id']]>p['cosine'] for p in rows)/len(rows))
        result=dict(model_id=name,revision=revision,positive_mean=float(np.mean(pos)),negative_mean=float(np.mean(neg)),
                    pairwise_auc=float(np.mean([float(a>b)+.5*float(a==b) for a in pos for b in neg])),
                    within_anchor_ranking_accuracy=sum(positives[p['anchor_id']]>p['cosine'] for p in scores if not p['positive'])/len(neg),
                    categories=bycat,encoding_seconds=elapsed,load_seconds=load,unique_texts=len(texts),
                    seconds_per_unique_text=elapsed/len(texts),device=run.config['device'],cpu_rss_bytes=psutil.Process().memory_info().rss,
                    gpu_memory_bytes=0,native_short_text_max_error=native_error,encoder_max_seq_length=encoder.model.max_seq_length,
                    total_encoder_tokens=sum(x['encoder_tokens'] for x in lengths),max_encoder_tokens=max(x['encoder_tokens'] for x in lengths),
                    chunked_texts=sum(x['chunk_count']>1 for x in lengths),truncated_tokens=0,
                    model_files=[record(p) for p in sorted(encoder.path.rglob('*')) if p.is_file() and '.cache' not in p.parts],
                    scores=record(run.out/f'{label}_scores.jsonl'))
        results.append(result);run.metric(event='semantic_diagnostic_complete',model_id=name,encoding_seconds=elapsed,ranking=result['within_anchor_ranking_accuracy'],categories=bycat)
        del encoder,vectors;gc.collect()
    run.finish(dict(status='PASS',pair_count=len(pairs),category_counts=dict(Counter(p['category'] for p in pairs)),
                    results=results,chunk_policy=CHUNK_POLICY,
                    caveat='Controlled pairs share 12 synthetic anchors plus 10 train explanations; descriptive sensitivity, not independent clinical validation.',
                    optimizer_updates=0))
