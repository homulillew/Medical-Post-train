"""Strict Stage 0 config. Formal runs deliberately have no entry point yet."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

QWEN_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"

@dataclass
class ProbeConfig:
    purpose: str
    stage: int = 0
    run_class: str = "DIAGNOSTIC"
    seed: int = 42
    model: str = f"/data/WSH/medical-post-train-artifacts/models/Qwen3-8B/{QWEN_REVISION}"
    artifact_root: str = "/data/WSH/medical-post-train-artifacts/runs"
    adapter: str | None = None
    resume: str | None = None
    timeout_seconds: int = 7200
    vllm_runner: str = 'v1'
    batch_invariant: bool = True

    def __post_init__(self):
        if self.stage != 0 or self.run_class not in {"DIAGNOSTIC", "SMOKE"}:
            raise ValueError("This runtime only authorizes Stage 0 diagnostics")
        if self.purpose not in {"environment", "snapshot", "lora", "reload", "vllm", "verl", "numeric", "minibatch", "semantic", "length", "template"}:
            raise ValueError(f"Unknown probe: {self.purpose}")
        if not 1 <= self.timeout_seconds <= 14400:
            raise ValueError("Stage 0 timeout must be in [1, 14400] seconds")
        if self.vllm_runner not in {'v1','v2'}:
            raise ValueError('vllm_runner must be explicitly v1 or v2')
        if type(self.batch_invariant) is not bool:
            raise ValueError('batch_invariant must be a boolean')
        if Path(self.model).name != QWEN_REVISION:
            raise ValueError("Qwen3-8B immutable revision required")

def read_config(path):
    return asdict(ProbeConfig(**json.loads(Path(path).read_text())))
