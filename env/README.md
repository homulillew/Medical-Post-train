# Stage 0 runtime

Use Python **3.12.7** on Linux x86_64, a CUDA 13 capable NVIDIA driver, and the two isolated venvs. The system/conda base is not modified. `train.lock` contains exact versions and allowed distribution SHA-256 hashes; `source_manifest.json` fixes the separately built verl source and wheel. `environment_manifest.json` points to actual import/ABI evidence, not just resolution.

```bash
uv venv --python /home/ubuntu/anaconda3/bin/python .venv-train
uv pip sync --python .venv-train/bin/python env/train.lock
git clone https://github.com/verl-project/verl /tmp/mpt-verl-source
git -C /tmp/mpt-verl-source checkout 483b8a009ba3a97563edee3a19887e4862b8094a
uv build --wheel --out-dir /tmp/mpt-wheels /tmp/mpt-verl-source
uv pip install --python .venv-train/bin/python --no-deps /tmp/mpt-wheels/verl-0.9.0-py3-none-any.whl -e .
uv pip check --python .venv-train/bin/python
uv venv --python /home/ubuntu/anaconda3/bin/python .venv-analysis
uv pip sync --python .venv-analysis/bin/python env/analysis.lock
uv pip install --python .venv-analysis/bin/python --no-deps -e .
.venv-train/bin/mpt env probe
```

The no-deps source installation does not override dependency constraints: verl's install_requires are included in requirements.in and resolved in train.lock. `pip check` must pass afterwards. A rebuilt wheel can have different ZIP timestamps; verify pinned source and compare installed content or retain the original wheel using source_manifest's hash. Build tooling is isolated by uv; the emitted build log is retained. Optional torchtitan/veomni/automodel/megatron engines are not installed and their explicit warnings do not select another actor implementation.

Core stack: torch 2.11.0+cu130 / CUDA runtime 13.0, vLLM 0.24.0, verl 0.9.0 fixed source, Transformers 5.5.3, PEFT 0.18.1; complete versions including Ray and NCCL are in the measured environment manifest. Native flash-attn is deliberately absent: actor uses explicit SDPA with remove_padding=False and fused kernels=False. vLLM selects its own compiled attention backend; actual selected backend is retained in logs. `/usr/bin/nvcc` 12 is not used to compile CUDA 13 extensions. Standard torch NCCL is retained; the upstream Dockerfile's incompatible override was not applied.

SOCKS proxy support (`socksio==1.0.0`) was added after a real HF download failure. No credentials or proxy URLs are stored. Set HF_HUB_DISABLE_XET=1 if retrying the recorded Xet download stall; downloads always use the pinned model revision in configs. The bulk artifact root is `/data/WSH/medical-post-train-artifacts`, outside Git. Preserve or copy it before deleting this machine.

Examples:

```bash
.venv-train/bin/mpt compatibility --config configs/stage0/snapshot.json
.venv-train/bin/mpt sft --mode smoke --config configs/stage0/lora.json
.venv-train/bin/mpt run inspect experiments/stage0/RUN_ID
.venv-train/bin/mpt run resume experiments/stage0/RUN_ID --checkpoint /absolute/checkpoint_step3.pt
.venv-analysis/bin/python -m pytest tests -q
.venv-analysis/bin/python scripts/verify_stage0.py --bulk-hashes
```

Every invocation prepares a unique Stage 0 run and starts an independent process. stdout/stderr, PID, heartbeat, resolved config, source hashes/archive, metrics and checkpoint references live under its attempt directory. `run resume` starts a new child run referencing an explicit trusted local checkpoint. It never treats an adapter-only load as resume. Formal training has no CLI entry point in this MVP. systemd reboot reconciliation and Stage 1 data cursor recovery remain future gates.

Final LoRA compatibility policy: `VLLM_USE_V2_MODEL_RUNNER=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, and **`VLLM_BATCH_INVARIANT=1`** are explicitly set before importing vLLM. The installed native LoRA shrink resolves split_k=1. Default-mode LoRA repeated logprobs were unstable; V1 alone and extra warmup alone did not fix them. The final run passed cold/warm controls, three sleep cycles and real adapter synchronization with the native batch-invariant option. Preserve these flags in future rollout workers and account for the measured throughput cost. This does not alter the BF16 base or replace native LoRA with merged/quantized inference.
