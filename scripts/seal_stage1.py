"""Seal completed bulk evidence and export compact training curves into Git."""
import argparse
import csv
import json
import hashlib
from pathlib import Path
from medical_posttrain.evidence import sha256,write_json

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--final',action='store_true');args=parser.parse_args()
    index=Path('experiments/stage1');artifacts={};inventory=[]
    for manifest_path in sorted(index.glob('*/manifest.json')):
        git=manifest_path.parent;manifest=json.loads(manifest_path.read_text());bulk=Path(manifest['artifact_root'])
        status_path=bulk/'status.json'
        status=json.loads(status_path.read_text()) if status_path.exists() else dict(status='UNKNOWN')
        if status['status']=='RUNNING':
            if args.final:raise ValueError(f'Cannot seal active run {git.name}')
            continue
        if (bulk/'reload_generation').exists() and not (bulk/'reload_generation/receipt.json').exists():
            if args.final:raise ValueError(f'Reload follow-up still incomplete: {git.name}')
            continue
        write_json(git/'status.json',status)
        for name in ('summary.json','resume_receipt.json'):
            if (bulk/name).exists():write_json(git/name,json.loads((bulk/name).read_text()))
        if (bulk/'reload_generation/receipt.json').exists():write_json(git/'reload_receipt.json',json.loads((bulk/'reload_generation/receipt.json').read_text()))
        if (bulk/'canonical_metrics.jsonl').exists():
            rows=[json.loads(line) for line in (bulk/'canonical_metrics.jsonl').open()]
            columns=['global_step','sample_cursor','coverage_fraction','loss','grad_norm','learning_rate','processed_tokens','supervised_tokens','update_seconds','tokens_per_second','nvml_peak_bytes','allocated_peak_bytes','reserved_peak_bytes']
            with (git/'training_curve.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
        run_files=[]
        for path in sorted(bulk.rglob('*')):
            if path.is_file():
                item=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size)
                run_files.append(item);artifacts[str(path)]=item
        snapshot=git/'artifact_snapshots'/(hashlib.sha256(json.dumps(run_files,sort_keys=True).encode()).hexdigest()+'.json')
        if not snapshot.exists():write_json(snapshot,run_files)
        write_json(git/'sealed_artifacts.json',run_files)
        inventory.append(dict(run_id=manifest['run_id'],run_class=manifest['run_class'],status=status['status'],artifact_root=str(bulk),files=len(run_files)))
    # Source download/projection logs preceded the common run wrapper. Their exact
    # start time was not collected; preserve this limitation and the failed attempt.
    bootstrap=[]
    bulk_data=Path('/data/WSH/medical-post-train-artifacts/data')
    for path in [bulk_data/'stage1-download.log',bulk_data/'stage1-download-attempt2.log',bulk_data/'stage1-governance-001.log',bulk_data/'stage1-governance-002.log',bulk_data/'stage1-case-audit.log']:
        if path.exists():
            item=dict(path=str(path),sha256=sha256(path),bytes=path.stat().st_size);bootstrap.append(item);artifacts[str(path)]=item
    write_json(index/'bootstrap_evidence.json',dict(run_id='s1_source_bootstrap_20260908_001',run_class='DIAGNOSTIC',stage=1,started_at=None,start_time_note='Not captured before the wrapper existed; never inferred from file mtime',commands=['.venv-train/bin/python scripts/prepare_stage1.py (attempts 1 and 2)'],failure='Attempt 1 requested nonexistent CMExam data/test.csv; corrected to pinned official test_with_annotations.csv',artifacts=bootstrap))
    write_json(index/'run_inventory.json',inventory)
    if args.final:write_json(index/'final_artifacts.json',list(artifacts.values()))

if __name__=='__main__':main()
