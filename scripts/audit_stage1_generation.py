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
    from tokenizers.decoders import DecodeStream
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
    one_shot_differences = []
    for cap in summary['limits']:
        for model in ('base', 'sft'):
            path = root / f'{model}_{cap}.jsonl'
            rows = [json.loads(line) for line in path.open()]
            assert len(rows) == 50
            for row in rows:
                assert row['prompt_ids'] == prompt_ids[row['sample_id']]
                # vLLM 0.24 FastIncrementalDetokenizer uses a prompt-prefilled
                # DecodeStream. One-shot decode may append a replacement char
                # when the length cap cuts through a multi-token UTF-8 sequence.
                stream = DecodeStream(ids=row['prompt_ids'], skip_special_tokens=True)
                decoded = ''.join(stream.step(tokenizer._tokenizer, token) or ''
                    for token in row['output_ids'])
                assert decoded == row['text'], (row['sample_id'], 'text-token mismatch')
                one_shot = tokenizer.decode(row['output_ids'], skip_special_tokens=True)
                if one_shot != decoded:
                    one_shot_differences.append(dict(sample_id=row['sample_id'], model=model,
                        cap=cap, finish_reason=row['finish_reason'],
                        one_shot_suffix=one_shot[-40:], native_incremental_suffix=decoded[-40:]))
                recomputed = measure(row['text'], row['output_ids'], row['finish_reason'], tokenizer)
                assert all(row[k] == v for k, v in recomputed.items()), row['sample_id']
                count += 1
            assert aggregate(rows) == summary['results'][f'{model}_{cap}']
            files.append(path)
    write_json(args.output, dict(status='PASS', stage=1, evaluation_run=selected['evaluation'],
        prompt_count=50, outputs_redecoded_and_remeasured=count,
        checks=['native prompt tokenization', 'output IDs reproduce raw text through native prompt-prefilled DecodeStream',
            'all format/length/repetition measurements recomputed', 'source aggregates recomputed'],
        one_shot_decode_differences=one_shot_differences,
        decoding_source='Installed vLLM 0.24 FastIncrementalDetokenizer: tokenizers.decoders.DecodeStream(ids=prompt_token_ids, skip_special_tokens=True)',
        previous_failure='experiments/stage1/generation_audit_attempt1.json',
        script_sha256=sha256(__file__), sources={str(p): sha256(p) for p in files}))
    print(f'PASS: {count} raw generations redecoded and remeasured')


if __name__ == '__main__':
    main()
