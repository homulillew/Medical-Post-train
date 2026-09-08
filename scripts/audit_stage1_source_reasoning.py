"""Read-only audit of the selected longest reasoning case and duplicate boundaries."""
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
import platform
import subprocess
import sys
import uuid

from medical_posttrain.evidence import sha256, write_json


def main():
    started = datetime.now(timezone.utc).isoformat()
    index = Path('experiments/stage1')
    selected = json.loads((index / 'selected_runs.json').read_text())
    data = json.loads((index / selected['data'] / 'manifest.json').read_text())
    root = Path(data['artifact_root'])
    train_path = root / 'train.jsonl'
    rows = [json.loads(line) for line in train_path.open()]
    longest = max(rows, key=lambda r: r['reasoning_tokens'])
    # Explicit assertions tie this diagnostic to the actual observed source text.
    assert longest['source'] == 'medical_o1' and longest['source_row'] == 14099
    target = longest['messages'][-1]['content']
    assert '3.9921875mg/L' in target and '9.9921875mg/L' in target
    assert longest['reasoning_tokens'] == 868
    run_id = 's1_source_reasoning_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:6]
    out = root.parent / run_id
    out.mkdir(exist_ok=False)
    git = index / run_id
    git.mkdir(exist_ok=False)
    write_json(git / 'manifest.json', dict(run_id=run_id, stage=1, run_class='DIAGNOSTIC',
        purpose='Inspect selected longest reasoning, arithmetic consistency and cross-source duplicate edge counts; no training/data mutation',
        started=started, artifact_root=str(out), source_data_run=selected['data'],
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        dirty_state=bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip()),
        script_sha256=sha256(__file__), command=[sys.executable, *sys.argv],
        environment=dict(python=sys.version, platform=platform.platform(), gpu_used=False),
        source_hashes={str(p): sha256(p) for p in [train_path, root / 'duplicate_edges.jsonl']}))
    concentration = Decimal(0)
    recurrence = []
    for n in range(1, 14):
        concentration = concentration / 2 + 4
        recurrence.append(dict(dose_number=n, hours=(n-1)*12, value=str(concentration)))
    assert Decimal('3.9921875') + 4 == Decimal('7.9921875')
    assert all(Decimal(r['value']) < 8 for r in recurrence)
    edges = [json.loads(line) for line in (root / 'duplicate_edges.jsonl').open()]
    edge_counts = Counter(tuple(sorted((r['left'].split(':')[0], r['right'].split(':')[0]))) for r in edges)
    cross = [r for r in edges if {r['left'].split(':')[0], r['right'].split(':')[0]} == {'medical_o1', 'huatuo'}]
    write_json(out / 'selected_longest_reasoning.json', {k: v for k, v in longest.items() if k not in ('input_ids', 'labels')})
    write_json(out / 'arithmetic_check.json', dict(observed_expression='3.9921875 + 4 = 9.9921875',
        correct_sum='7.9921875', recurrence='C_n = C_(n-1)/2 + 4, C_0 = 0',
        closed_form='C_n = 8 * (1 - 2**(-n))', limit=8, values=recurrence,
        scope='Algebra under the source question’s simplified recurrence; no clinical treatment assessment'))
    write_json(out / 'cross_source_edges.json', cross)
    case = dict(case_id='S1-SOURCE-REASONING-001', run_id=run_id, stage=1,
        category='BAD_CASE', subtype='selected_source_reasoning_arithmetic_inconsistency',
        prompt_id=longest['sample_id'], prompt=longest['messages'][0]['content'],
        ground_truth='Under the stated recurrence, peaks approach 8; the target 10 is unreachable.',
        responses=[dict(kind='original_training_target', text=target)],
        observation='The longest selected CoT has an explicit incorrect sum and contradictory claims about reaching 10. Its final answer says the target cannot be reached but also incorrectly describes approaching 10.',
        hypothesis='Unverified source-authored reasoning contains factual noise despite structural cleaning.',
        alternative_explanations=['This extreme selected case is not a random quality sample; it cannot estimate the dataset error rate.'],
        followup='Preserve frozen formal baseline and annotate its limitation. A later curated-data experiment needs a new version/run; do not silently remove this record mid-epoch.',
        artifact_refs=[str(out / 'selected_longest_reasoning.json'), str(out / 'arithmetic_check.json')])
    write_json(git / 'cases.json', [case])
    summary = dict(run_id=run_id, status='PASS', observed_source_quality_failure=True,
        selected_longest_reasoning_sample=longest['sample_id'], reasoning_tokens=868,
        direct_medical_o1_huatuo_duplicate_edges=len(cross),
        edge_source_pairs={' / '.join(k): v for k, v in sorted(edge_counts.items())},
        frozen_data_changed=False, quality_error_rate=None,
        conclusion='Structural validity and lexical isolation do not establish reasoning correctness.')
    write_json(out / 'summary.json', summary)
    write_json(git / 'summary.json', summary)
    write_json(out / 'status.json', dict(status='PASS', ended=datetime.now(timezone.utc).isoformat()))
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
