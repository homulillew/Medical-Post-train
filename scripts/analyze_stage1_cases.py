"""Mine completed Stage 1 evidence without changing training or selecting a model.

Thresholds below label diagnostic examples, not medical quality or acceptance.
The script refuses partial formal/evaluation runs and preserves prior outputs.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from medical_posttrain.evidence import sha256, write_json


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).open()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/stage1/case_coverage.json')
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), 'Preserve previous case analyses; choose a new output'
    index = Path('experiments/stage1')
    selected = read_json(index / 'selected_runs.json')
    formal_root = Path(read_json(index / selected['formal'] / 'manifest.json')['artifact_root'])
    formal = read_json(formal_root / 'summary.json')
    assert formal['run_class'] == 'FORMAL' and formal['unique_examples'] == 20000
    assert formal['coverage_fraction'] == 1 and formal['global_step'] == 1250
    evaluation_dir = index / selected['evaluation']
    evaluation = read_json(evaluation_dir / 'summary.json')
    assert evaluation['status'] == 'PASS'
    evaluation_root = Path(read_json(evaluation_dir / 'manifest.json')['artifact_root'])
    prompts = {row['sample_id']: row for row in read_json(evaluation_root / 'prompts.json')}
    cases = []
    counts = Counter()
    source_files = set()
    representative_limit = 3

    def retain(subtype, category, rows, observation, context=None):
        counts[subtype] += 1
        if sum(c['subtype'] == subtype for c in cases) >= representative_limit:
            return
        row = rows[-1]
        prompt = prompts[row['sample_id']]
        cases.append(dict(
            case_id=f'S1-GEN-{len(cases)+1:03d}', run_id=selected['evaluation'], stage=1,
            category=category, subtype=subtype, prompt_id=row['sample_id'],
            prompt=prompt['messages'], ground_truth=prompt['reference'],
            ground_truth_scope='Source-authored validation target; not independently adjudicated',
            responses=[{k: r[k] for k in ('model', 'max_response_tokens', 'text',
                'finish_reason', 'total_tokens', 'reasoning_tokens', 'answer_tokens',
                'think_closed', 'answer_closed', 'format_valid', 'repetition_max')} for r in rows],
            parsed_answers=[], reward_components={}, metric_context=context or {},
            observation=observation,
            why_interesting='Paired structural/length evidence; not a medical correctness score.',
            hypothesis=None,
            alternative_explanations=['Single sampled response per policy and prompt; source style and sampling variation can affect behavior.'],
            followup='Review paired text; confirm behavior on Stage 2 train-only profiling before freezing RL generation.',
            artifact_refs=[str(evaluation_root / f"{r['model']}_{r['max_response_tokens']}.jsonl") for r in rows],
        ))

    for cap in evaluation['limits']:
        source_files.update(evaluation_root / f'{model}_{cap}.jsonl' for model in ('base', 'sft'))
        base = read_rows(evaluation_root / f'base_{cap}.jsonl')
        sft = read_rows(evaluation_root / f'sft_{cap}.jsonl')
        assert len(base) == len(sft) == 50
        for before, after in zip(base, sft):
            assert before['sample_id'] == after['sample_id']
            assert before['prompt_ids'] == after['prompt_ids']
            pair = [before, after]
            if not before['format_valid'] and after['format_valid']:
                retain('paired_format_improvement', 'GOOD_CASE', pair, 'The same prompt changes from failing to satisfying the explicit think/answer output grammar.')
            if after['format_valid'] and after['total_tokens'] <= .65 * before['total_tokens']:
                retain('paired_shorter_structured_output', 'GOOD_CASE', pair, 'SFT output is at most 65% of the Base output length and has a complete answer block; completeness/correctness of medical content is not implied.')
            if after['format_valid'] and after['reasoning_tokens'] > 0 and (not before['think_closed'] or not before['answer_closed']):
                retain('paired_closed_reasoning_and_answer', 'GOOD_CASE', pair, 'SFT produces nonempty, closed reasoning and answer blocks where Base lacks at least one required block; this is a structural observation.')
            if before['format_valid'] and not after['format_valid']:
                retain('paired_format_regression', 'REGRESSION_CASE', pair, 'The Base output satisfies the grammar and the SFT output does not.')
            if not before['truncated'] and after['truncated']:
                retain('paired_truncation_regression', 'REGRESSION_CASE', pair, 'Only the SFT response reaches the shared length cap.')
            for row in pair:
                if not row['think_closed']:
                    retain('generation_missing_think_close', 'BAD_CASE', [row], 'No complete think block is present.', {'condition': row['model'], 'cap': cap})
                if not row['answer_closed']:
                    retain('generation_missing_answer_close', 'BAD_CASE', [row], 'No nonempty closed answer block is present; Base may still contain an untagged semantic answer.', {'condition': row['model'], 'cap': cap})
                if not row['format_valid']:
                    retain('format_anomaly', 'BAD_CASE', [row], 'Output fails the exact target grammar.', {'condition': row['model'], 'cap': cap})
                if row['empty_answer'] or (row['answer_closed'] and row['answer_tokens'] <= 3):
                    retain('empty_or_near_empty_tagged_answer', 'BAD_CASE', [row], 'Closed answer content has at most three tokens. A short multiple-choice answer is not automatically incorrect.')
                if row['repetition_max'] >= 3:
                    retain('generation_repetition', 'BAD_CASE', [row], 'A sentence fragment of at least 12 characters occurs at least three times.')
            if after['source'] == 'huatuo' and after['reasoning_tokens'] > 512:
                retain('huatuo_long_reasoning_over512', 'BAD_CASE', pair, 'SFT emits more than 512 reasoning tokens despite empty-think Huatuo targets. This threshold labels cost, not whether thought is medically necessary.')
            if after['source'] == 'medical_o1' and after['reasoning_tokens'] == 0:
                retain('medical_o1_no_reasoning', 'BAD_CASE', pair, 'No measured reasoning content in the SFT response to a reasoning-source validation prompt.')

    if len(evaluation['limits']) > 1:
        small, large = sorted(evaluation['limits'])
        shorter = read_rows(evaluation_root / f'sft_{small}.jsonl')
        longer = read_rows(evaluation_root / f'sft_{large}.jsonl')
        for before, after in zip(shorter, longer):
            assert before['sample_id'] == after['sample_id']
            if before['truncated'] and after['answer_closed'] and not after['truncated']:
                retain('cap_extension_closes_answer', 'GOOD_CASE', [before, after], 'The same SFT prompt is truncated at the smaller cap and has a closed answer at the larger cap; this is a length diagnostic, not group-accuracy boundary evidence.')
            if before['truncated'] and after['truncated']:
                retain('cap_extension_still_truncated', 'BAD_CASE', [before, after], 'Both measured caps truncate this SFT prompt.')

    required = ['paired_format_improvement', 'paired_shorter_structured_output',
        'paired_closed_reasoning_and_answer', 'paired_format_regression',
        'paired_truncation_regression', 'generation_missing_think_close',
        'generation_missing_answer_close', 'format_anomaly',
        'empty_or_near_empty_tagged_answer', 'generation_repetition',
        'huatuo_long_reasoning_over512', 'medical_o1_no_reasoning']
    if len(evaluation['limits']) > 1:
        required.extend(['cap_extension_closes_answer', 'cap_extension_still_truncated'])
    not_observed = [dict(subtype=k, scope='Completed paired validation conditions', status='NOT_OBSERVED') for k in required if not counts[k]]
    not_assessed = [dict(subtype='clinically_adjudicated_improvement_or_regression', status='NOT_ASSESSED', scope='No expert clinical adjudication in Stage 1'),
        dict(subtype='mixed_group_boundary_case', status='NOT_ASSESSED', scope='Group rollout belongs to later stages')]

    metrics_path = formal_root / 'canonical_metrics.jsonl'
    metrics = read_rows(metrics_path)
    assert len(metrics) == 1250
    losses = np.array([r['loss'] for r in metrics])
    assert np.isfinite(losses).all()
    q1, q3 = np.percentile(losses, [25, 75])
    threshold = float(q3 + 3 * (q3-q1))
    outliers = [r for r in metrics if r['loss'] > threshold]
    extrema = sorted(metrics, key=lambda r: r['loss'], reverse=True)[:3]
    # Batches are identifiable; assigning batch mean loss to a single sample would
    # fabricate a per-example measurement.
    training_diagnostics = dict(loss_min=float(losses.min()), loss_max=float(losses.max()),
        first50_mean=float(losses[:50].mean()), last50_mean=float(losses[-50:].mean()),
        descriptive_high_loss_threshold=threshold, high_loss_batches=len(outliers),
        threshold_scope='Retrospective Q3+3*IQR across the epoch; not a stability or early-stop criterion',
        highest_loss_batches=extrema,
        attribution='Loss is measured per effective batch of 16; no per-example loss is inferred.')
    if not outliers:
        not_observed.append(dict(subtype='batch_loss_above_descriptive_threshold', status='NOT_OBSERVED', scope='1250 canonical update losses'))
    else:
        for row in outliers[:3]:
            cases.append(dict(case_id=f"S1-LOSS-{row['global_step']:04d}", run_id=selected['formal'], stage=1,
                category='SYSTEM_CASE', subtype='high_loss_batch', observation='Batch loss exceeds the retrospective Q3+3*IQR label; early high loss is not necessarily instability.',
                metric_context=row, artifact_refs=[str(metrics_path)]))
    source_files.add(metrics_path)
    data_cases_path = index / selected['data_audit'] / 'cases.json'
    data_cases = read_json(data_cases_path)
    audit_root = Path(read_json(index / selected['data_audit'] / 'manifest.json')['artifact_root'])
    resolved_path = audit_root / 'rejections_resolved.jsonl'
    resolved = {r['sample_id']: r for r in read_rows(resolved_path)}
    for case in data_cases:
        if case.get('rejection') and case['prompt_id'] in resolved:
            case['original_rejection'] = case['rejection']
            case['rejection'] = resolved[case['prompt_id']]
            case['artifact_refs'].append(str(resolved_path))
    source_files.add(resolved_path)
    cases.extend(data_cases)
    source_files.add(data_cases_path)
    reasoning_cases = index / selected['source_reasoning_audit'] / 'cases.json'
    cases.extend(read_json(reasoning_cases))
    source_files.add(reasoning_cases)
    not_observed += [dict(subtype=k, status='NOT_OBSERVED', scope=scope) for k, scope in [
        ('direct_medical_o1_huatuo_duplicate_edge', 'Frozen full duplicate graph; benchmark-to-SFT edges do exist'),
        ('selected_example_over_cap_or_answer_truncation', 'All 21000 native encodings checked'),
        ('assistant_mask_failure_in_final_pipeline', 'Native tokenizer fixtures and full selected-data retokenization; historical fixture failures are retained in development logs')]]
    smoke_root = Path(read_json(index / selected['smoke'] / 'manifest.json')['artifact_root'])
    smoke_path = smoke_root / 'reload_generation/generations.jsonl'
    smoke_rows = read_rows(smoke_path)
    for number, row in enumerate(smoke_rows, 1):
        if row['format_valid'] and row['eos_terminated']:
            continue
        cases.append(dict(case_id=f'S1-SMOKE-{number:03d}', run_id=selected['smoke'],
            stage=1, category='BAD_CASE', subtype='smoke_greedy_format_or_length_failure',
            prompt_id=row['sample_id'], prompt=row['prompt_messages'],
            ground_truth=row['reference_answer'], responses=[row['output']],
            observation='Real smoke adapter reload produced a response failing the format or EOS check.',
            metric_context={k: row[k] for k in ('output_tokens', 'format_valid', 'eos_terminated')},
            hypothesis='Eight updates may not establish the target output style.',
            alternative_explanations=['Greedy decoding can promote repetitive long output; sampled final comparison uses a different decoding protocol.'],
            followup='Preserve the failure; compare final paired sampled outputs separately.',
            artifact_refs=[str(smoke_path)]))
    source_files.add(smoke_path)
    output_cases = output.with_name(output.stem + '_cases.json')
    assert not output_cases.exists()
    write_json(output_cases, cases)
    write_json(output, dict(stage=1, formal_run=selected['formal'], evaluation_run=selected['evaluation'],
        analysis_script_sha256=sha256(__file__), created_at=datetime.now(timezone.utc).isoformat(),
        diagnostic_thresholds=dict(shorter_length_ratio=.65, near_empty_answer_tokens=3, huatuo_long_reasoning_tokens=512, repetition_count=3),
        category_counts=dict(counts), count_scope='Counts over prompt-condition pairs; multiple categories may label the same prompt. Retain first three per subtype in frozen prompt order.',
        cases=[dict(case_id=c['case_id'], category=c['category'], subtype=c['subtype'], path=str(output_cases)) for c in cases],
        not_observed=not_observed, not_assessed=not_assessed, training_diagnostics=training_diagnostics,
        source_hashes={str(p): sha256(p) for p in sorted(source_files)},
        manual_review='Pending separate review record; automatic extraction does not establish medical correctness.'))
    print(json.dumps(dict(cases=len(cases), counts=dict(counts), not_observed=not_observed), ensure_ascii=False))


if __name__ == '__main__':
    main()
