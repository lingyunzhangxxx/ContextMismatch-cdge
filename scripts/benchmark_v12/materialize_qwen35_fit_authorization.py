#!/usr/bin/env python3
"""Materialize a SHA-bound code-v85 authorization for Qwen3.5 C-DGE fit."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


EXPECTED_KEY_SHA = "488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35"
EXPECTED_PROTECTED_FAMILIES = {
    "fresh": 768,
    "verification": 1536,
    "obedience_reset": 1536,
    "supported_user_authority": 144,
    "factual_boundary_memory": 24,
}


def _source(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _validate_manifest(stage: str, path: Path) -> None:
    value = json.loads(path.read_text())
    expected_rows = {"governance": 6144, "gradient": 6144, "protected": 4008}[
        stage
    ]
    expected_stage = f"qwen35_cdge_{stage}_capture"
    if value.get("stage") != expected_stage:
        raise ValueError(f"{stage} manifest stage mismatch")
    for field, expected in {
        "partition": "subspace_fit",
        "rows": expected_rows,
        "complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"{stage} manifest mismatch: {field}")
    if stage in {"governance", "gradient"}:
        if value.get("unique_job_keys") != 6144:
            raise ValueError(f"{stage} manifest unique-key count mismatch")
        if value.get("observed_key_sha256") != EXPECTED_KEY_SHA:
            raise ValueError(f"{stage} manifest key SHA mismatch")
    if stage == "gradient" and value.get("base_model_parameter_gradients") is not False:
        raise ValueError("gradient manifest permits base-model gradients")
    if stage == "protected" and value.get("family_rows") != EXPECTED_PROTECTED_FAMILIES:
        raise ValueError("protected manifest family counts mismatch")


def _validate_receipt(stage: str, path: Path) -> dict:
    value = json.loads(path.read_text())
    run_id = str(value.get("run_id", ""))
    expected_prefix = f"qwen3-5-9b-cdge-{stage}-capture-"
    if not run_id.startswith(expected_prefix):
        raise ValueError(f"{stage} receipt run mismatch")
    expected_archive = run_id + ".tar.gz"
    for field in (
        "cluster_shared_copy_verified",
        "host_data_copy_verified",
        "local_copy_verified",
    ):
        if value.get(field) is not True:
            raise ValueError(f"{stage} receipt is not three-copy verified: {field}")
    if value.get("archive_name") != expected_archive:
        raise ValueError(f"{stage} receipt archive-name mismatch")
    archive_sha = str(value.get("archive_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
        raise ValueError(f"{stage} receipt archive SHA is invalid")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError(f"{stage} receipt lacks terminal Slurm provenance")
    return {
        "run_id": run_id,
        "job_id": int(value["job_id"]),
        "archive_name": expected_archive,
        "archive_sha256": archive_sha,
        "receipt_sha256": sha256_file(path),
        "three_copy_verified": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "code_root",
        "selector_snapshot",
        "replication_protocol",
        "crossover_contract",
        "v3_template",
        "v4_template",
        "composite_template",
        "capture_manifest",
        "gradient_manifest",
        "protected_manifest",
        "governance_receipt",
        "gradient_receipt",
        "protected_receipt",
        "archive_helper",
        "archive_coordinator",
        "pull_helper",
        "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    expected_root = "/workspace/context-mismatch-qwen3-5-9b/code-v85"
    if str(args.code_root) != expected_root:
        raise ValueError("code-v85 required")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    selection = snapshot.get("selection_contract", {})
    if selection.get("npus") != 0 or selection.get("cpus") != 32:
        raise ValueError("fit selector resource mismatch")
    if selection.get("mem_mib") != 196608:
        raise ValueError("fit selector memory mismatch")
    protocol = json.loads(args.replication_protocol.read_text())
    if protocol.get("status") != "frozen_before_any_qwen3_5_9b_cdge_forward":
        raise ValueError("replication protocol is not frozen")
    for stage, path in (
        ("governance", args.capture_manifest),
        ("gradient", args.gradient_manifest),
        ("protected", args.protected_manifest),
    ):
        _validate_manifest(stage, path)
    receipts = {
        "governance": _validate_receipt("governance", args.governance_receipt),
        "gradient": _validate_receipt("gradient", args.gradient_receipt),
        "protected": _validate_receipt("protected", args.protected_receipt),
    }
    created = args.created_utc or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-5-9b-cdge-fit-code-v85",
        "created_utc": created,
        "stage": "qwen35_cdge_fit",
        "method": "C-DGE-V4.1",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(
            args.code_root / "bundle.sha256"
        ),
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "execution_allowed": True,
        "replication_protocol_sha256": sha256_file(args.replication_protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "v3_template_sha256": sha256_file(args.v3_template),
        "v4_template_sha256": sha256_file(args.v4_template),
        "composite_template_sha256": sha256_file(args.composite_template),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "gradient_manifest_sha256": sha256_file(args.gradient_manifest),
        "protected_manifest_sha256": sha256_file(args.protected_manifest),
        "capture_manifests": {
            "governance": _source(args.capture_manifest),
            "gradient": _source(args.gradient_manifest),
            "protected": _source(args.protected_manifest),
        },
        "capture_archive_receipts": receipts,
        "node_selection_snapshot": _source(args.selector_snapshot),
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
            "cpus_per_task": 32,
            "mem_mib": 196608,
            "npu_type": "910B3",
            "npus": 0,
            "time_limit": "06:00:00",
            "node": args.selected_node,
        },
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
