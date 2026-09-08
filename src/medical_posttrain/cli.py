import argparse
import json
from pathlib import Path
from medical_posttrain.config import ProbeConfig, read_config
from medical_posttrain.evidence import write_json
from medical_posttrain.runtime import environment, launch, prepare, worker

def main():
    parser = argparse.ArgumentParser(prog="mpt")
    subs = parser.add_subparsers(dest="command", required=True)
    env = subs.add_parser("env")
    env.add_argument("action", choices=["probe"])
    for name in ("compatibility", "sft"):
        p = subs.add_parser(name)
        p.add_argument("--config", required=True)
        p.add_argument("--mode", choices=["smoke"], default="smoke")
    r = subs.add_parser("run")
    r.add_argument("action", choices=["prepare", "inspect", "resume"])
    r.add_argument("path")
    r.add_argument("--checkpoint")
    w = subs.add_parser("worker")
    w.add_argument("path")
    a = parser.parse_args()
    if a.command == "env":
        from dataclasses import asdict
        launch(prepare(asdict(ProbeConfig(purpose="environment"))))
    elif a.command in {"compatibility", "sft"}:
        c = read_config(a.config)
        if a.command == "sft" and c["purpose"] != "lora":
            parser.error("sft smoke requires purpose=lora")
        launch(prepare(c))
    elif a.command == "worker":
        worker(a.path)
    elif a.action == "prepare":
        print(prepare(read_config(a.path)))
    elif a.action == "inspect":
        p = Path(a.path)
        for f in sorted(p.rglob("status.json")):
            print(f, f.read_text())
    else:
        if not a.checkpoint:
            parser.error("resume requires explicit --checkpoint (never an ambiguous latest)")
        parent=Path(a.path)
        attempts=sorted(parent.glob('attempt_*'))
        if not attempts or json.loads((attempts[-1]/'status.json').read_text())['status']!='PASS':
            parser.error('Stage 0 controlled resume requires a completed parent with committed checkpoint evidence')
        from medical_posttrain.evidence import sha256
        checkpoint=Path(a.checkpoint).resolve()
        artifacts=[json.loads(l) for l in (attempts[-1]/'artifacts.jsonl').read_text().splitlines()]
        recorded=next((x for x in artifacts if x['path']==str(checkpoint)),None)
        if recorded is None or sha256(checkpoint)!=recorded['sha256']:
            parser.error('Checkpoint is not a verified artifact of this parent run')
        old = json.loads((parent/"manifest.json").read_text())
        c = old["config"] | {"purpose":"reload", "resume":str(Path(a.checkpoint).resolve())}
        root = prepare(c)
        write_json(root / "parent.json", dict(parent_run_id=old["run_id"], checkpoint=c["resume"]))
        launch(root)

if __name__ == "__main__":
    main()
