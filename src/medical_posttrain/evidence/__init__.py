import hashlib
import json
import os
from pathlib import Path
from datetime import datetime, timezone

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)

class Evidence:
    def __init__(self, directory, config):
        self.path = Path(directory)
        self.config = config
        self.run_id = self.path.parent.name
        self.bulk = Path(config["artifact_root"]) / self.run_id / self.path.name
        self.bulk.mkdir(parents=True, exist_ok=False)

    def metric(self, **row):
        row = dict(run_id=self.run_id, timestamp=now(), **row)
        with (self.path / "metrics.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)
        return row

    def case(self, subtype, observation, category="SYSTEM_CASE", **details):
        import uuid
        case_id = f"{self.run_id}_{subtype}_{uuid.uuid4().hex[:6]}"
        row = dict(case_id=case_id, run_id=self.run_id, stage=0, category=category,
                   subtype=subtype, observation=observation, hypothesis=None,
                   alternative_explanations=[], followup="See Stage 0 report and linked raw evidence",
                   reproduction_command=json.loads((self.path / "command.json").read_text()), **details)
        write_json(self.path / "cases" / f"{case_id}.json", row)

    def artifact(self, path, role):
        path = Path(path).resolve()
        row = dict(path=str(path), role=role, size=path.stat().st_size, sha256=sha256(path))
        with (self.path / "artifacts.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")
        return row
