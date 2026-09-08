#!/usr/bin/env python3
"""Validate takeover deliverables. This NEVER verifies completion of a Stage."""
import copy
import hashlib
import json
from pathlib import Path
import re

from jsonschema import Draft202012Validator


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads((ROOT / path).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    required = ["IMPLEMENTATION_PLAN", "ENVIRONMENT_AUDIT", "DEPENDENCY_STRATEGY",
                "REPOSITORY_ARCHITECTURE", "EXECUTION_PLAN", "COMPUTE_BUDGET",
                "CHECKPOINT_AND_RESUME", "EXPERIMENT_DATA_MODEL", "RISK_REGISTER",
                "OPEN_QUESTIONS", "UPSTREAM_FINDINGS", "DECISIONS"]
    for name in required:
        path = ROOT / "docs/implementation" / f"{name}.md"
        require(path.is_file() and path.stat().st_size > 500, f"Missing planning content: {name}")
    for path in (ROOT / "schemas").glob("*.json"):
        Draft202012Validator.check_schema(json.loads(path.read_text()))
    state = read_json("project_state.json")
    validator = Draft202012Validator(read_json("schemas/project_state.schema.json"))
    validator.validate(state)
    require(state["phase"] in {"TAKEOVER_PLANNING", "IMPLEMENTATION"}, "This checker only guards planning and Stage 0 implementation")
    require(not state["stage_verifier_implemented"], "Planning is not Stage verification")
    for stage, entry in state["stages"].items():
        require(entry["status"] == "NOT_STARTED", f"Stage {stage} must remain NOT_STARTED")
        require(not entry["run_ids"], f"Stage {stage} has unplanned training run IDs")
        require(all(value in (0, []) for value in entry["progress"].values()),
                f"Stage {stage} has invented progress")
        require(all(entry[k] is None for k in ["verification_receipt", "stage_report", "interview_story"]),
                f"Stage {stage} has premature evidence")
    contract = read_json("contracts/stage_budgets.json")
    require(state["contract_sha256"] == sha(ROOT / "contracts/stage_budgets.json"), "Contract hash drift")
    for path, expected in contract["source_documents"].items():
        require(sha(ROOT / path) == expected, f"Research contract source changed: {path}")
    b = contract["stage_budgets"]
    require(b["1"]["train_examples"] == 20000 and b["1"]["planned_epochs"] == 1,
            "SFT budget changed")
    require(b["2"]["candidate_pool"] == 15000 and b["2"]["profiling_unique_prompts"] == 1000
            and b["2"]["completed_responses"] == 4000, "Profiling budget changed")
    require(b["3"]["accepted_integration_mixed_groups"] == 256, "Refill budget changed")
    require(b["4"]["vanilla_training_groups"] == b["4"]["dynamic_accepted_mixed_groups"] == 5000
            and b["4"]["accepted_trajectories_per_variant"] == 20000, "GSPO budget changed")
    require(all(b[s]["group_size"] == 4 for s in ["2", "3", "4"]), "Group size changed")
    require(b["5"]["primary_checkpoints"] == 3 and b["5"]["cmexam_questions_per_checkpoint"] == 6811
            and b["5"]["cmb_clean_questions_per_checkpoint"] == 2000, "Evaluation budget changed")
    require(b["6"]["consistency_prompts"] == 100 and b["6"]["minimum_real_requests_total"] == 100
            and b["6"]["concurrency_conditions"] == [1, 4, 8, 16], "Serving budget changed")

    # Negative schema probes: partial numeric progress cannot even pass structural gates.
    rejected = 0
    for status in ["FULL_PASS", "VERIFIED", "DONE"]:
        for count in [0, 320, 4999]:
            bad = copy.deepcopy(state)
            e = bad["stages"]["4"]
            e.update(status=status, run_ids=["synthetic_vanilla", "synthetic_dynamic"],
                     stage_report="synthetic.md", interview_story="synthetic.md",
                     verification_receipt={"path": "synthetic.json", "sha256": "a" * 64,
                                           "verifier_commit": "b" * 40,
                                           "contract_sha256": state["contract_sha256"], "result": "PASS"})
            e["progress"] = {"vanilla_training_groups": 5000, "dynamic_accepted_mixed_groups": count}
            require(not validator.is_valid(bad), f"Schema accepted incomplete {status}/{count}")
            rejected += 1
    for stage in state["stages"]:
        bad = copy.deepcopy(state)
        bad["stages"][stage]["status"] = "DONE"
        require(not validator.is_valid(bad), f"Schema accepted unsupported DONE in stage {stage}")
        rejected += 1

    # Local document links; ignore external URLs, anchors, and illustrative code fences.
    checked_links = 0
    for path in [ROOT / "README.md", *(ROOT / "docs/implementation").glob("*.md")]:
        text = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for target in re.findall(r"\]\(([^)]+)\)", text):
            target = target.split("#", 1)[0]
            if not target or "://" in target or target.startswith("mailto:"):
                continue
            require((path.parent / target).exists(), f"Broken link {path.name}: {target}")
            checked_links += 1
    env = read_json("docs/implementation/evidence/environment-20260908-v2.json")
    require(env["purpose"] == "environment_probe" and env["stage"] == 0
            and env["does_not_complete_any_stage"], "Probe mislabeled as stage execution")
    require(env["script_sha256"] == sha(ROOT / "scripts/probe_environment.py"), "Probe source drift")
    dep = read_json("docs/implementation/evidence/dependency-resolution.json")
    for field, filename in [("input_sha256", "candidate-metadata-input.txt"),
                            ("output_sha256", "candidate-metadata-resolved.txt")]:
        require(dep[field] == sha(ROOT / "docs/implementation/evidence" / filename), "Resolver evidence drift")
    print(json.dumps({"planning_check": "PASS", "negative_state_probes_rejected": rejected,
                      "local_links_checked": checked_links, "stage_completion_verified": False,
                      "stages": {s: e["status"] for s, e in state["stages"].items()}}, indent=2))


if __name__ == "__main__":
    main()
