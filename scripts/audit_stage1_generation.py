"""Recompute final generation measurements from raw output text and token IDs."""
import argparse
import json
from pathlib import Path

from medical_posttrain.evidence import sha256, write_json
from evaluate_stage1 import aggregate, measure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/stage1/generation_audit.json')
    args = parser.parse_args()
    assert not Path(args.output).exists(), 'Preserve earlier audit receipts'
    index = Path('experiments/stage1')
    selected = json.loads((index / 'selected_runs.json').read_text())
    formal_manifest = json.loads((index / selected['formal'] / 'manifest.json').read_text())
    config = json.loads((Path(formal_manifest['artifact_root']) / 'config.json').read_text())
    evaluation_dir = index / selected['evaluation']
    manifest = json.loads((evaluation_dir / 'manifest.json').read_text())
    root = Path(manifest['artifact_root'])
    summary = json.loads((evaluation_dir / 'summary.json').read_text())
    assert summary['status'] == 'PASS'
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config['model'], local_files_only=True)
    prompts = json.loads((root / 'prompts.json').read_text())
    assert len(prompts) == 50
    prompt_ids = {}
    for row in prompts:
        ids = tokenizer.apply_chat_template(row['messages'], tokenize=True,
            return_dict=False, add_generation_prompt=True, enable_thinking=True)
        assert ids == row['prompt_ids']
        prompt_ids[row['sample_id']] = ids
    files = [root / 'prompts.json', evaluation_dir / 'summary.json']
    count = 0
    for cap in summary['limits']:
        for model in ('base', 'sft'):
            path = root / f'{model}_{cap}.jsonl'
            rows = [json.loads(line) for line in path.open()]
            assert len(rows) == 50
            for row in rows:
                assert row['prompt_ids'] == prompt_ids[row['sample_id']]
                # vLLM's default detokenization removes EOS and retains the Qwen
                # reasoning delimiters, matching the native tokenizer's behavior.
                decoded = tokenizer.decode(row['output_ids'], skip_special_tokens=True)
                assert decoded == row['text'], (row['sample_id'], 'text-token mismatch')
                recomputed = measure(row['text'], row['output_ids'], row['finish_reason'], tokenizer)
                assert all(row[k] == v for k, v in recomputed.items()), row['sample_id']
                count += 1
            assert aggregate(rows) == summary['results'][f'{model}_{cap}']
            files.append(path)
    write_json(args.output, dict(status='PASS', stage=1, evaluation_run=selected['evaluation'],
        prompt_count=50, outputs_redecoded_and_remeasured=count,
        checks=['native prompt tokenization', 'output IDs decode to raw text',
            'all format/length/repetition measurements recomputed', 'source aggregates recomputed'],
        script_sha256=sha256(__file__), sources={str(p): sha256(p) for p in files}))
    print(f'PASS: {count} raw generations redecoded and remeasured')


if __name__ == '__main__':
    main()
