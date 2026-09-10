#!/usr/bin/env python3
"""Wait for the detached API worker, then verify and score its full frozen probe."""
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from medical_posttrain.rl.common import read, record, immutable, durable, now


def main(out):
    index = ROOT / 'experiments/stage5/external_baselines' / out.name
    owner = read(out / 'launch.json')['pid']
    while True:
        state = read(out / 'status.json')
        durable(index / 'progress.json', state)
        if state['status'] == 'RESPONSES_COMPLETE':
            break
        assert state['status'] == 'RUNNING', state
        stat = Path(f'/proc/{owner}/stat')
        assert stat.exists() and stat.read_text().split()[2] != 'Z', 'API owner exited before completion'
        time.sleep(15)
    while (stat := Path(f'/proc/{owner}/stat')).exists() and stat.read_text().split()[2] != 'Z':
        time.sleep(1)
    analyzer = ROOT / 'scripts/analyze_stage5_external.py'
    immutable(index / 'analysis_launch.json', dict(timestamp=now(), analyzer=record(analyzer),
        command=[sys.executable, str(analyzer), '--run', str(out)]))
    subprocess.run([sys.executable, str(analyzer), '--run', str(out)], check=True)
    durable(index / 'progress.json', dict(status='EXTERNAL_PROBE_VERIFIED', timestamp=now(),
        completed=1100, planned=1100, open_qa_judge='PENDING', stage5_status='NOT_STARTED'))


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True)
    a = p.parse_args()
    main(Path(a.run))
