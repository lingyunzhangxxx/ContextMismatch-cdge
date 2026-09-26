#!/usr/bin/env python3
"""Create one immutable SHA-bound Qwen3.5 adapter-smoke authorization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


WORK_ROOT = "/workspace/context-mismatch-qwen3-5-9b"


def _source(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc")
    parser.add_argument("--replication-protocol", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--archive-helper", type=Path, required=True)
    parser.add_argument("--archive-coordinator", type=Path, required=True)
    parser.add_argument("--pull-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if str(args.code_root) != f"{WORK_ROOT}/code-v83":
        raise ValueError("Qwen3.5 adapter smoke requires immutable code-v83")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")

    protocol = json.loads(args.replication_protocol.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    model = json.loads(args.model_contract.read_text())
    if protocol.get("status") != "frozen_before_any_qwen3_5_9b_cdge_forward":
        raise ValueError("cross-model replication protocol is not frozen")
    for field, expected in {
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if protocol.get(field) != expected:
            raise ValueError(f"replication safety mismatch: {field}")
    smoke = protocol.get("stages", {}).get("adapter_smoke", {})
    for field, expected in {
        "engineering_only": True,
        "scientific_effect_inference_forbidden": True,
        "expected_rows": 1,
        "exact_zero_hook_identity_required": True,
    }.items():
        if smoke.get(field) != expected:
            raise ValueError(f"adapter-smoke contract mismatch: {field}")
    if crossover.get("base_model", {}).get("revision") != protocol.get("base_model", {}).get(
        "revision"
    ):
        raise ValueError("cross-model contract revision mismatch")
    if model.get("contract") != "context-mismatch-qwen35-9b-bf16-v1":
        raise ValueError("unexpected model contract")
    if model.get("non_quantized") is not True or model.get("production_rollout_approved") is not False:
        raise ValueError("model contract safety mismatch")
    if sha256_file(args.model_contract) != protocol["frozen_lineage"]["model_contract_sha256"]:
        raise ValueError("model contract lineage mismatch")
    if sha256_file(args.manifest) != protocol["frozen_lineage"]["benchmark_manifest_sha256"]:
        raise ValueError("benchmark manifest lineage mismatch")

    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    created = args.created_utc or (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created):
        raise ValueError("invalid created UTC")

    authorization = {
        "schema_version": 1,
        "authorization_id": "qwen3-5-9b-cdge-adapter-smoke-code-v83",
        "created_utc": created,
        "stage": "qwen35_cdge_adapter_smoke",
        "method": "C-DGE-V4.1",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "execution_allowed": True,
        "replication_protocol_sha256": sha256_file(args.replication_protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_contract_sha256": sha256_file(args.model_contract),
        "expected_rows": 1,
        "engineering_only": True,
        "scientific_effect_inference_forbidden": True,
        "exact_zero_hook_identity_required": True,
        "node_selection_snapshot": _source(args.selector_snapshot),
        "model_contract": _source(args.model_contract),
        "monitoring_contract": {
            "continuous_watch_required": True,
            "poll_seconds": 1,
            "archive_helper": _source(args.archive_helper),
            "archive_coordinator": _source(args.archive_coordinator),
            "pull_helper": _source(args.pull_helper),
        },
        "resource_contract": {
            "partition": "a01",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "mem_mib": 131072,
            "npu_type": "910B3",
            "npus": 1,
            "time_limit": "01:00:00",
            "node": args.selected_node,
        },
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    print(json.dumps(authorization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
