#!/usr/bin/env python3
"""Print full committed group responses for agent qualitative review."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('indices',nargs='+',type=int);a=p.parse_args()
root=Path(__file__).resolve().parents[1];selected=json.loads((root/'experiments/stage3/selected_runs.json').read_text())
run=Path(json.loads((root/'experiments/stage3'/selected['formal']/'manifest.json').read_text())['artifact_root'])
for i in a.indices:
    batch=run/'batches'/f'{i//16:04d}';assert (batch/'commit.json').exists()
    group=json.loads((batch/'scored.json').read_text())['groups'][i%16]
    decision=json.loads((batch/'commit.json').read_text())['decisions'][i%16]
    print('GROUP',i,group['group_id'],decision)
    print('QUESTION',group['responses'][0]['question']);print('OPTIONS',group['responses'][0]['options']);print('GT',group['responses'][0]['ground_truth'])
    for r in group['responses']:
        print('TRAJECTORY',r['member_index'],{k:r[k] for k in ('parsed_answer','acc','semantic','format','total_reward','parse_error','output_tokens')})
        print(r['raw_output']);print('END TRAJECTORY')
