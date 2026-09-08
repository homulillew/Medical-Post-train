import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time
import traceback
import uuid
import zipfile
from medical_posttrain.evidence import Evidence, now, sha256, write_json

ROOT = Path(__file__).resolve().parents[3]
BOUNDS = {
    'lora': {'base':'Qwen3-8B','dtype':'bf16','lora_rank':32,'lora_alpha':64,'microbatch':1,'sequence_lengths':[512,1024,2048],'updates':4,'lr':1e-4,'attention':'sdpa','gradient_checkpointing':True},
    'reload': {'resume_step':3,'continue_updates':1,'reference_step':4},
    'vllm': {'dtype':'bf16','tp':1,'max_model_len':4096,'max_num_seqs':16,'memory_utilization':.65,'max_tokens':32,'temperature':0,'sleep_level':1,'actor_switch_updates':1},
    'length': {'train_prompts':32,'group_size':4,'response_limits':[512,1024],'temperature':.6,'top_p':1,'top_k':-1,'selection':'first 32 train records, no label filtering'},
    'minibatch': {'synthetic_prompts':8,'group_size':4,'mini_prompts':[8,4],'epochs':1,'total_optimizer_updates':3,'lr':1e-5,'temperature':.6,'clip_low':.0003,'clip_high':.0004},
    'verl': {'strategy':'fsdp2','gpu_count':1,'sequence':128,'optimizer_updates':1,'attention':'sdpa','remove_padding':False},
    'numeric': {'devices':['cpu','cuda'],'real_model':False},
    'semantic': {'synthetic_pairs':7,'device':'cpu','threads':8},
}

def command(args):
    p = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    return dict(argv=args, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)

def environment():
    return dict(python=sys.version, executable=sys.executable, hostname=platform.node(),
                platform=platform.platform(), packages={d.metadata["Name"]: d.version for d in md.distributions()},
                gpu=command(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader"]),
                env_allowlist={k:os.environ[k] for k in ("CUDA_VISIBLE_DEVICES", "VLLM_USE_V1", "HF_HUB_OFFLINE", "PYTORCH_ALLOC_CONF") if k in os.environ})

def prepare(config, run_id=None):
    run_id = run_id or f"s0_{config['purpose']}_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:6]}"
    root = ROOT / "experiments/stage0" / run_id
    root.mkdir(exist_ok=False)
    write_json(root / "manifest.json", dict(run_id=run_id, stage=0, run_class=config["run_class"], purpose=config["purpose"], created_at=now(), planned_budget=BOUNDS.get(config['purpose'],{'single_probe':True}), config=config))
    return root

def launch(root):
    root = Path(root).resolve()
    # Executing venv/bin/mpt does not activate PATH; JIT tools must use this runtime.
    os.environ['PATH']=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')
    attempts = sorted(root.glob("attempt_*"))
    if attempts:
        s = json.loads((attempts[-1] / "status.json").read_text())
        if s["status"] in {"RUNNING", "LAUNCHED"}:
            raise ValueError("Previous attempt is active; inspect PID/heartbeat before recovery")
    attempt = root / f"attempt_{len(attempts)+1:03d}"
    attempt.mkdir()
    config = json.loads((root / "manifest.json").read_text())["config"]
    config = config | {'diagnostic_contract':BOUNDS.get(config['purpose'],{'single_probe':True})}
    write_json(attempt / "resolved_config.json", config)
    write_json(attempt / "environment.json", environment())
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for parent in ("src", "scripts", "configs", "env") for p in (ROOT / parent).rglob("*") if p.is_file() and "__pycache__" not in str(p) and p.suffix not in {".log", ".pyc"}}
    write_json(attempt / "code.json", dict(revision=command(["git", "rev-parse", "HEAD"])["stdout"].strip(), status=command(["git", "status", "--porcelain"])["stdout"], source_sha256=source_hashes))
    with zipfile.ZipFile(attempt / "source.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in source_hashes:
            archive.write(ROOT / name, name)
    argv = [sys.executable, "-m", "medical_posttrain.cli", "worker", str(attempt)]
    write_json(attempt / "command.json", argv)
    (attempt / "metrics.jsonl").touch()
    (attempt / "artifacts.jsonl").touch()
    write_json(attempt / "checkpoint_refs.json", [])
    with (attempt / "stdout.log").open("x") as out, (attempt / "stderr.log").open("x") as err:
        proc = subprocess.Popen(argv, cwd=ROOT, stdout=out, stderr=err, start_new_session=True)
    write_json(attempt / "launch.json", dict(pid=proc.pid, launched_at=now(), detached=True, timeout_seconds=config["timeout_seconds"]))
    print(attempt, flush=True)
    return attempt

def worker(path):
    path = Path(path)
    config = json.loads((path / "resolved_config.json").read_text())
    ev = Evidence(path, config)
    start = time.monotonic()
    stop = threading.Event()
    write_json(path / "status.json", dict(status="RUNNING", pid=os.getpid(), started_at=now()))
    def heartbeat():
        while not stop.wait(5):
            elapsed = time.monotonic()-start
            write_json(path / "heartbeat.json", dict(pid=os.getpid(), timestamp=now(), elapsed_seconds=elapsed))
            if elapsed > config["timeout_seconds"]:
                write_json(path / "status.json", dict(status="FAILED", reason="bounded diagnostic timeout", ended_at=now()))
                import signal
                os.killpg(os.getpgrp(), signal.SIGTERM)
    thread = threading.Thread(target=heartbeat, daemon=True)
    thread.start()
    try:
        from medical_posttrain.verification.probes import dispatch
        result = dispatch(ev)
        write_json(path / "summary.json", result)
        status = dict(status="PASS", ended_at=now(), elapsed_seconds=time.monotonic()-start)
    except BaseException as exc:
        traceback.print_exc()
        ev.case("probe_failure", str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        status = dict(status="FAILED", reason=str(exc), ended_at=now(), elapsed_seconds=time.monotonic()-start)
    finally:
        stop.set()
        thread.join()
    write_json(path / "status.json", status)
    (path / "observations.md").write_text(f"Stage 0 {config['purpose']}: {status['status']}. See raw metrics, cases, and summary.\n")
    write_json(path / "artifacts_manifest.json", [dict(path=str(p.relative_to(path)),size=p.stat().st_size,sha256=sha256(p)) for p in path.rglob("*") if p.is_file() and p.name not in {"artifacts_manifest.json", "stdout.log", "stderr.log"}])
    if status["status"] != "PASS":
        raise SystemExit(1)
