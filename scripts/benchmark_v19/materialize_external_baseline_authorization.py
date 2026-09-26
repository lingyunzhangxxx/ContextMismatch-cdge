#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
GROUPS = {
    "input": ("governance_reset_prompt", "session_isolation"),
    "activation": ("caa", "cast"),
    "representation": ("loreft", "reps"),
}


def bound(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["fit", "evaluation"], required=True)
    parser.add_argument("--group", choices=sorted(GROUPS))
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--target-node", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fit-report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (len(args.target_node) == 3 and args.target_node[0] == "a" and args.target_node[1:].isdigit()):
        raise ValueError("invalid target node")
    code = args.code_root
    protocol = code / "protocol/QWEN3_8B_EXTERNAL_BASELINES_V1.json"
    source = code / "protocol/EXTERNAL_BASELINE_SOURCE_MANIFEST_V1.json"
    capture = Path("/workspace/context-mismatch-qwen3-8b/inputs/governance-capture-v1/capture_manifest.json")
    gradient = Path("/workspace/context-mismatch-qwen3-8b/inputs/gradient-capture-v1/margin_gradient_manifest.json")
    common = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-external-baseline-{args.stage}-{args.group or 'all'}-{code.name}",
        "created_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "code_root": str(code),
        "immutable_code_bundle_manifest_sha256": sha256_file(code / "bundle.sha256"),
        "execution_allowed": True,
        "execution_node": args.target_node,
        "protocol_sha256": sha256_file(protocol),
        "source_manifest_sha256": sha256_file(source),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "monitoring_contract": {
            "continuous_watch_required": True,
            "poll_seconds": 1,
            "archive_helper": bound(code / "archive_external_baseline_v83_evidence.sh"),
            "pull_helper": bound(code / "pull_external_baseline_v83_evidence.sh"),
        },
    }
    if args.stage == "fit":
        if args.group or args.checkpoint or args.fit_report:
            raise ValueError("fit authorization received evaluation arguments")
        common.update({
            "stage": "qwen3_8b_external_baseline_fit",
            "capture_manifest_sha256": sha256_file(capture),
            "gradient_manifest_sha256": sha256_file(gradient),
            "capture_manifest": bound(capture),
            "gradient_manifest": bound(gradient),
            "operator_dev_accessed": False,
            "resource_contract": {"partition": "a01", "nodes": 1, "ntasks": 1, "cpus_per_task": 32, "mem_mib": 196608, "npu_type": "910B3", "npus": 0, "time_limit": "08:00:00", "node": args.target_node},
        })
    else:
        if args.group is None:
            raise ValueError("evaluation authorization requires a group")
        common.update({
            "stage": "qwen3_8b_external_baseline_evaluation",
            "group": args.group,
            "methods": list(GROUPS[args.group]),
            "benchmark_manifest_sha256": sha256_file(code / "artifacts/benchmark_manifest.jsonl"),
            "crossover_contract_sha256": sha256_file(code / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json"),
            "design_audit_sha256": sha256_file(code / "artifacts/governance_crossover_operator_dev_design_audit_v2.json"),
            "model_manifest_sha256": sha256_file(code / "artifacts/node_model_verification.json"),
            "expected_rows_per_method": 3072,
            "expected_key_sha256": "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e",
            "operator_dev_accessed_only_during_evaluation": True,
            "resource_contract": {"partition": "a01", "nodes": 1, "ntasks": 1, "cpus_per_task": 8, "mem_mib": 131072, "npu_type": "910B3", "npus": 1, "time_limit": "08:00:00", "node": args.target_node},
        })
        if args.group == "input":
            if args.checkpoint or args.fit_report:
                raise ValueError("input baselines must not bind a fitted checkpoint")
        else:
            if args.checkpoint is None or args.fit_report is None:
                raise ValueError("fitted baseline evaluation requires fit evidence")
            common["checkpoint"] = bound(args.checkpoint)
            common["fit_report"] = bound(args.fit_report)
            common["checkpoint_sha256"] = sha256_file(args.checkpoint)
            common["fit_report_sha256"] = sha256_file(args.fit_report)
    atomic_write_text(args.output, json.dumps(common, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
