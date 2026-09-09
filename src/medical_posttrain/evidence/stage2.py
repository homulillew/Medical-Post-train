"""Stage 2 durable run records and physical-attempt lineage."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile
from . import now, sha256, write_json
from medical_posttrain.runtime import environment

ROOT = Path(__file__).resolve().parents[3]
BULK = Path('/data/WSH/medical-post-train-artifacts/runs')
INDEX = ROOT / 'experiments/stage2'


def read(path):
    return json.loads(Path(path).read_text())


def jsonlines(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def dump_lines(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush(); os.fsync(f.fileno())


def record(path):
    path = Path(path)
    return dict(path=str(path.resolve()), sha256=sha256(path), bytes=path.stat().st_size)


def source_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for base in ('src', 'scripts', 'configs')
            for p in (ROOT/base).rglob('*') if p.is_file() and p.suffix in ('.py', '.json', '.yaml')}


def prepare(purpose, run_class, config):
    run_id = 's2_' + purpose + '_' + time.strftime('%Y%m%dT%H%M%S', time.gmtime()) + '_' + uuid.uuid4().hex[:6]
    out = BULK/run_id; out.mkdir()
    git = INDEX/run_id; git.mkdir(parents=True)
    write_json(out/'config.json', config)
    sources = source_hashes()
    with zipfile.ZipFile(out/'source.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources: archive.write(ROOT/path, path)
    manifest = dict(run_id=run_id, stage=2, purpose=purpose, run_class=run_class,
                    artifact_root=str(out), config=config, config_sha256=sha256(out/'config.json'),
                    source_hashes=sources, git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                    dirty_state=subprocess.check_output(['git','status','--porcelain'],text=True), created_at=now(),
                    optimizer_updates=0)
    write_json(git/'manifest.json', manifest); write_json(out/'manifest.json', manifest)
    write_json(out/'status.json', dict(status='PREPARED', timestamp=now()))
    return out


def launch(out, action):
    out = Path(out)
    attempt = out/f'attempt_{len(list(out.glob("attempt_*")))+1:03d}'
    attempt.mkdir()
    argv = [sys.executable, str(ROOT/'scripts/run_stage2.py'), 'worker', '--run', str(out), '--action', action, '--attempt', str(attempt)]
    write_json(attempt/'command.json', argv)
    with (attempt/'stdout.log').open('x') as stdout, (attempt/'stderr.log').open('x') as stderr:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
    write_json(attempt/'launch.json', dict(pid=proc.pid, timestamp=now(), detached=True))
    print(json.dumps(dict(run=str(out), attempt=str(attempt), pid=proc.pid)), flush=True)


class Run:
    def __init__(self, out, attempt):
        self.out, self.attempt = Path(out), Path(attempt)
        self.manifest = read(self.out/'manifest.json')
        self.config = self.manifest['config']
        self.run_id = self.manifest['run_id']
        self.git = INDEX/self.run_id
        self.start = time.monotonic()
        assert sha256(self.out/'config.json') == self.manifest['config_sha256']
        write_json(self.attempt/'environment.json', environment())
        write_json(self.attempt/'code.json', dict(source_hashes=source_hashes()))
        write_json(self.attempt/'status.json', dict(status='RUNNING', started=now(), pid=os.getpid()))
        write_json(self.out/'status.json', dict(status='RUNNING', started=now(), pid=os.getpid(), action=self.attempt.name))

    def metric(self, **row):
        row = dict(run_id=self.run_id, timestamp=now(), **row)
        with (self.attempt/'metrics.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def finish(self, summary):
        summary = dict(run_id=self.run_id, **summary)
        write_json(self.out/'summary.json', summary); write_json(self.git/'summary.json', summary)
        write_json(self.attempt/'status.json', dict(status='PASS', ended=now(), wall_seconds=time.monotonic()-self.start))
        write_json(self.out/'status.json', dict(status='PASS', ended=now()))
        write_json(self.git/'artifacts_manifest.json', [record(p) for p in sorted(self.out.rglob('*')) if p.is_file()])


def select(name, out):
    path = INDEX/'selected_runs.json'; selected = read(path) if path.exists() else {}
    selected[name] = Path(out).name; write_json(path, selected)


def selected_path(name):
    run_id = read(INDEX/'selected_runs.json')[name]
    return Path(read(INDEX/run_id/'manifest.json')['artifact_root'])
