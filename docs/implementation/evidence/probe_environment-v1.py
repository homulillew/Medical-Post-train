#!/usr/bin/env python3
"""Bounded environment diagnostic; never loads a model or changes stage state."""
import argparse
import datetime as dt
import importlib.metadata as metadata
import json
import platform
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid


def command(args):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=20)
        return {"argv": args, "returncode": p.returncode,
                "stdout": p.stdout.strip(), "stderr": p.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"argv": args, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cuda-smoke", action="store_true",
                        help="Run only a 64x64 BF16 forward/backward; no model weights")
    args = parser.parse_args()
    start = time.monotonic()
    now = dt.datetime.now(dt.timezone.utc)
    report = {"schema_version": 1,
              "run_id": f"s0_environment_{now:%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}",
              "run_class": "DIAGNOSTIC", "purpose": "environment_probe",
              "stage": 0, "started_at": now.isoformat(),
              "command": [sys.executable, *sys.argv],
              "script_sha256": __import__("hashlib").sha256(Path(__file__).read_bytes()).hexdigest(),
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "packages": {}, "commands": [],
              "does_not_complete_any_stage": True}
    for name in ["pip", "uv", "conda", "torch", "transformers", "peft", "datasets",
                 "vllm", "verl", "flash-attn", "bitsandbytes", "accelerate", "triton",
                 "ray", "sentence-transformers", "safetensors", "huggingface-hub",
                 "nvidia-nccl-cu13", "nvidia-cuda-runtime-cu13", "jsonschema"]:
        try:
            report["packages"][name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            report["packages"][name] = None
    for argv in [["git", "rev-parse", "HEAD"], ["git", "status", "--porcelain"],
                 ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,compute_cap",
                  "--format=csv"], ["nvcc", "--version"], ["lscpu"], ["free", "-b"],
                 ["df", "-BT", "/", "/data", "/tmp", "/dev/shm"],
                 ["findmnt", "-T", str(Path.cwd())],
                 ["systemctl", "is-system-running"],
                 ["loginctl", "show-user", "ubuntu", "-p", "Linger"]]:
        report["commands"].append(command(argv))
    report["paths"] = {name: shutil.which(name) for name in
                       ["python", "pip", "uv", "conda", "nvcc", "tmux", "systemctl", "docker"]}
    report["os_release"] = Path("/etc/os-release").read_text()
    report["status"] = "COMPLETE"
    if args.cuda_smoke:
        try:
            import torch
            torch.manual_seed(0)
            report["torch_runtime"] = {
                "version": torch.__version__, "cuda": torch.version.cuda,
                "nccl": torch.cuda.nccl.version(), "cudnn": torch.backends.cudnn.version(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
                "device_count": torch.cuda.device_count()}
            a = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16, requires_grad=True)
            value = (a @ a).float().mean()
            value.backward()
            torch.cuda.synchronize()
            finite = bool(torch.isfinite(a.grad).all())
            report["smoke"] = {"seed": 0, "matrix_shape": [64, 64], "value": value.item(),
                               "finite_gradient": finite,
                               "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
            if not finite:
                report["status"] = "FAILED"
        except Exception as e:
            report["status"] = "FAILED"
            report["smoke_error"] = repr(e)
    report["elapsed_seconds"] = time.monotonic() - start
    report["ended_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(json.dumps({k: report[k] for k in ["run_id", "status", "elapsed_seconds"]}))
    return 0 if report["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
