#!/usr/bin/env python3
"""Read-only progress, with no completion inference from partial counts."""
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
selected=json.loads((root/'experiments/stage3/selected_runs.json').read_text())
for name in ('smoke','formal'):
    if name not in selected:continue
    out=Path(json.loads((root/'experiments/stage3'/selected[name]/'manifest.json').read_text())['artifact_root'])
    status=json.loads((out/'status.json').read_text())
    checkpoint=out/'checkpoint.json';state=json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    print(json.dumps(dict(condition=name,run_id=out.name,status=status,progress={k:state.get(k) for k in ('batches','generated_groups','accepted_mixed_groups','all_wrong_groups','all_correct_groups','invalid_groups','overflow_mixed_groups')},output_tokens_by_disposition=state.get('output_tokens_by_disposition')),ensure_ascii=False))
