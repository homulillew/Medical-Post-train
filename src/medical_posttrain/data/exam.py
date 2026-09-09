"""CMExam train-only pool, with inherited sealed heldout cluster exclusions."""
from collections import Counter
import csv
import re
from pathlib import Path
from medical_posttrain.data.stage1 import normalize_question, text_hash
from medical_posttrain.reward.parser import canonical_answer
from medical_posttrain.evidence import sha256, write_json
from medical_posttrain.evidence.stage2 import read, jsonlines, dump_lines, record

FORMAT_INSTRUCTION = ('请先进行必要分析，将分析写在 <think>...</think> 内，最后只用 '
                      '<answer>C</answer> 形式给出最终选项。多选按字母排序，例如 <answer>AC</answer>。'
                      'answer 标签闭合后不要追加其他内容。')


def options_schema(text):
    matches = list(re.finditer(r'^\s*([A-E])[\s.、:：)]+', text, re.M))
    if not matches or text[:matches[0].start()].strip():
        raise ValueError('options_schema')
    options = {m[1]: text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
               for i, m in enumerate(matches)}
    if len(options) != len(matches) or not 2 <= len(options) <= 5 or any(not v for v in options.values()):
        raise ValueError('duplicate_or_empty_options')
    if list(options) != list('ABCDE'[:len(options)]):
        raise ValueError('noncontiguous_options')
    return options


def messages(row):
    return [dict(role='user', content=row['question']+'\n'+'\n'.join(k+' '+v for k,v in row['options'].items())+'\n'+FORMAT_INSTRUCTION)]


def prepare_pool(run):
    cfg = run.config
    raw = read(cfg['raw_manifest'])
    clusters_path = Path(cfg['clusters'])
    assert sha256(clusters_path) == cfg['clusters_sha256']
    clusters = jsonlines(clusters_path)
    membership, forbidden = {}, set()
    for c in clusters:
        for member in c['members']: membership[member] = c['cluster_id']
        if any(x.startswith(('cmexam_val:', 'cmexam_test:', 'cmb_test:')) for x in c['members']):
            forbidden.add(c['cluster_id'])
    train, rejected, counts = [], [], Counter()
    provenance = []
    for split, expected in [('train',54497), ('val',6811)]:
        meta = next(x for x in raw['sources'] if x['source'] == 'cmexam_'+split)
        assert sha256(meta['path']) == meta['sha256']
        with open(meta['path']) as f:
            reader=csv.DictReader(f); fields=reader.fieldnames
            assert fields == ['Question','Options','Answer','Explanation']
            source = list(reader)
        assert len(source) == expected
        provenance.append(dict(**meta, actual_count=len(source), fields=fields, difficulty_annotation=False, category_annotation=False))
        if split == 'val':
            val_ids = [f'cmexam_val:{meta["revision"]}:{i}' for i in range(len(source))]
            # Test-overlap or repeated validation clusters go to reserve; monitor
            # and selection use different clean cluster representatives.
            test_clusters = {c['cluster_id'] for c in clusters if any(m.startswith(('cmexam_test:', 'cmb_test:')) for m in c['members'])}
            representatives, seen = [], set()
            for sid in sorted(val_ids, key=lambda x:text_hash('42:val:'+x)):
                cid = membership[sid]
                if cid in seen or cid in test_clusters: continue
                seen.add(cid); representatives.append(sid)
            assert len(representatives) >= 1536
            monitor, selection = representatives[:512], representatives[512:1536]
            used = set(monitor+selection)
            partition = dict(monitor=monitor, selection=selection, diagnostic_reserve=[s for s in val_ids if s not in used],
                             cluster_ids={sid:membership[sid] for sid in val_ids}, seed=42,
                             generation_performed=False, rule='clean cluster representatives for monitor/selection; all remaining official val IDs reserved')
            write_json(run.out/'validation_partitions.json', partition)
            continue
        for i,row in enumerate(source):
            sid=f'cmexam_train:{meta["revision"]}:{i}'; cid=membership[sid]
            try:
                options=options_schema(row['Options'])
                answer=canonical_answer(row['Answer'], ''.join(options))
                if not normalize_question(row['Question']): raise ValueError('empty_normalized_question')
                if any(tag in row['Question']+'\n'+row['Options'] for tag in ('<think','<answer','<|im_','<|endoftext|>')):
                    raise ValueError('reserved_control_token')
                if cid in forbidden: raise ValueError('heldout_cluster_overlap')
                train.append(dict(prompt_id=sid, source='CMExam', source_revision=meta['revision'], source_row=i,
                                  question=row['Question'], options=options, answer_set=answer,
                                  reference_explanation=row['Explanation'], explanation_missing=not bool(row['Explanation'].strip()),
                                  split='train', cluster_id=cid))
            except ValueError as error:
                reason=str(error); counts[reason]+=1
                rejected.append(dict(prompt_id=sid, reason=reason, cluster_id=cid))
    chosen, seen = [], set()
    for row in sorted(train, key=lambda r:text_hash('42:pool:'+r['prompt_id'])):
        if row['cluster_id'] in seen:
            rejected.append(dict(prompt_id=row['prompt_id'],reason='train_duplicate_cluster',cluster_id=row['cluster_id']))
            counts['train_duplicate_cluster']+=1
        else: seen.add(row['cluster_id']); chosen.append(row)
    assert len(chosen)>=15000
    pool=chosen[:15000]
    ordered=sorted(pool,key=lambda r:text_hash('42:profiling:'+r['prompt_id']))
    formal=[r['prompt_id'] for r in ordered[:1000]]
    smoke=[r['prompt_id'] for r in ordered[1000:1050]]
    dump_lines(run.out/'candidate_pool.jsonl', pool)
    dump_lines(run.out/'rejections.jsonl', rejected)
    write_json(run.out/'profiling_prompt_ids.json', formal); write_json(run.out/'smoke_prompt_ids.json', smoke)
    write_json(run.out/'source_validation.json', provenance)
    write_json(run.out/'cluster_provenance.json', dict(source=record(clusters_path), inherited_rules=cfg['near_rules'],
                test_access='No test source parsed in Stage2; only inherited sealed cluster ID membership consumed'))
    run.finish(dict(status='PASS', candidate_pool=len(pool), clean_unique_available=len(chosen), raw_train_count=54497,
                    rejections=dict(counts), answer_cardinality=dict(Counter(len(r['answer_set']) for r in pool)),
                    missing_explanations=sum(r['explanation_missing'] for r in pool),
                    pool=record(run.out/'candidate_pool.jsonl'), formal_selection=record(run.out/'profiling_prompt_ids.json'),
                    smoke_selection=record(run.out/'smoke_prompt_ids.json'), validation_partition=record(run.out/'validation_partitions.json'),
                    optimizer_updates=0))
