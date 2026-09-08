"""Pinned source download and question-only benchmark projection (no evaluation)."""
import csv
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import requests

ROOT = Path('/data/WSH/medical-post-train-artifacts/data/stage1')
SOURCES = {
    'medical_o1': ('https://huggingface.co/datasets/FreedomIntelligence/medical-o1-reasoning-SFT/resolve/fc2c9e8a37b38f38da6d449564a8c350b244aef4/medical_o1_sft_Chinese.json', 'fc2c9e8a37b38f38da6d449564a8c350b244aef4'),
    'huatuo': ('https://huggingface.co/datasets/FreedomIntelligence/HuatuoGPT2-SFT-GPT4-140K/resolve/4077ffaeb123e49b8b8a0283f42957a5570a52ce/HuatuoGPT2-GPT4-SFT-140K.json', '4077ffaeb123e49b8b8a0283f42957a5570a52ce'),
    'cmb_test': ('https://huggingface.co/datasets/FreedomIntelligence/CMB/resolve/935fbc09edf1303d89872b21265ff597f426ac0d/CMB-Exam/CMB-test/CMB-test-choice-question-merge.json', '935fbc09edf1303d89872b21265ff597f426ac0d'),
    **{f'cmexam_{split}': (f'https://raw.githubusercontent.com/williamliujl/CMExam/fadb22c89beb1b7115dc36460ba792eb96b7b972/data/{"test_with_annotations" if split == "test" else split}.csv', 'fadb22c89beb1b7115dc36460ba792eb96b7b972') for split in ('train', 'val', 'test')},
}

def sha(path):
    with open(path, 'rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def fetch(item):
    key, (url, revision) = item
    path = ROOT / 'raw' / (key + ('.csv' if key.startswith('cmexam') else '.json'))
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            with path.with_suffix(path.suffix+'.partial').open('wb') as f:
                for chunk in response.iter_content(1024*1024):
                    f.write(chunk)
        os.replace(path.with_suffix(path.suffix+'.partial'), path)
    result = dict(source=key, revision=revision, url=url, path=str(path), sha256=sha(path), bytes=path.stat().st_size)
    print(json.dumps(result), flush=True)
    return result

def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        sources = list(pool.map(fetch, SOURCES.items()))
    out = ROOT / 'benchmark_questions.jsonl'
    counts = {}
    with out.open('x') as f:
        for meta in sources:
            source = meta['source']
            if source.startswith('cmexam'):
                # Standard CSV parsing handles multiline quoted cells. Only Question is
                # accessed. Other columns never leave this isolated projection step.
                with open(meta['path'], newline='', encoding='utf-8-sig') as raw:
                    questions = [row['Question'] for row in csv.DictReader(raw)]
            elif source == 'cmb_test':
                questions = [row['question'] for row in json.load(open(meta['path']))]
            else:
                continue
            counts[source] = len(questions)
            for i, question in enumerate(questions):
                assert isinstance(question, str) and question.strip()
                f.write(json.dumps(dict(sample_id=f'{source}:{meta["revision"]}:{i}', source=source, source_row=i, question=question), ensure_ascii=False)+'\n')
    manifest = dict(sources=sources, exclusion=dict(path=str(out), sha256=sha(out), counts=counts, fields_accessed=['Question (CMExam)', 'question (CMB)'], purpose='DEDUP_ONLY', answers_used=False, difficulty_used=False))
    (ROOT/'raw_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(manifest['exclusion']), flush=True)

if __name__ == '__main__':
    main()
