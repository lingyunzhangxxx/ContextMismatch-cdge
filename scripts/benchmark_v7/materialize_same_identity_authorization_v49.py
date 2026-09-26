#!/usr/bin/env python3
"""Create a fresh operators authorization after repairing bundle completeness."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
REMOTE_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = REMOTE_ROOT / "code-v49"
CONTROL_ROOT = REMOTE_ROOT / "code-v49"
SOURCE_CODE_ROOT = REMOTE_ROOT / "code-v48"
EXPECTED_ROWS = 3072
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
METHOD_SHARDS = {
    "operators": ("fixed_negative_vector", "symmetric_rank_one"),
}
EXPECTED_SOURCE_AUTH_SHA256 = {
    "operators": "466361f5a43be9ea09adf31ef5d5f8112d9c443bf5b438034c77fc0a6edb1bae",
}
EXPECTED_REPLACED_JOB = {"operators": "9173"}
EXPECTED_AGGREGATE_CONTRACT = {
    "partition": "a01",
    "npu_type": "910B3",
    "npus": 1,
    "cpus": 8,
    "mem_mib": 131072,
    "allow_nodes": ["a06"],
    "exclude_nodes": [],
}
PER_JOB_RESOURCE_CONTRACT = {
    "partition": "a01",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 8,
    "mem_mib": 131072,
    "npu_type": "910B3",
    "npus": 1,
    "time_limit": "12:00:00",
}
BLOCKED_NODE_STATES = {
    "DOWN", "DRAIN", "DRAINED", "FAIL", "FAILING", "FUTURE", "INVAL",
    "MAINT", "NO_RESPOND", "PLANNED", "POWER_DOWN", "POWERING_DOWN",
    "REBOOT_REQUESTED", "RESERVED", "UNKNOWN",
}


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    resolved = str(path)
    if not resolved.startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("artifact must remain inside the cluster project root")
    return {"path": resolved, "sha256": sha256_file(path)}


def _verify_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"invalid artifact record: {label}")
    path = Path(str(record.get("path", "")))
    expected = record.get("sha256")
    if not str(path).startswith(f"{REMOTE_ROOT}/") or not re.fullmatch(r"[0-9a-f]{64}", str(expected)):
        raise ValueError(f"invalid artifact binding: {label}")
    if not path.is_file() or sha256_file(path) != expected:
        raise ValueError(f"artifact SHA mismatch: {label}")


def _verify_source_records(value: object, label: str = "sources") -> None:
    if isinstance(value, dict) and set(value) == {"path", "sha256"}:
        _verify_record(value, label)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _verify_source_records(child, f"{label}.{key}")
        return
    raise ValueError(f"unexpected source binding: {label}")


def _node_states(value: str) -> set[str]:
    return {token for token in re.split(r"[+~#$*!%-]+", value.upper()) if token}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--materializer-root", type=Path, required=True)
    parser.add_argument("--source-authorization", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--replaces-job-id", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != CODE_ROOT:
        raise ValueError("bundle-completeness retry requires immutable code-v49")
    if args.materializer_root != CONTROL_ROOT:
        raise ValueError("authorization materialization requires immutable code-v49")
    if not re.fullmatch(r"a[0-9]{2}", args.selected_node):
        raise ValueError("invalid selected node")
    bundle = args.code_root / "bundle.sha256"
    materializer_bundle = args.materializer_root / "bundle.sha256"
    if not bundle.is_file():
        raise FileNotFoundError("code-v49 bundle manifest is missing")
    if not materializer_bundle.is_file():
        raise FileNotFoundError("code-v44 control-plane bundle manifest is missing")

    source = json.loads(args.source_authorization.read_text())
    shard = source.get("method_shard")
    if shard not in METHOD_SHARDS:
        raise ValueError("invalid source method shard")
    if sha256_file(args.source_authorization) != EXPECTED_SOURCE_AUTH_SHA256[shard]:
        raise ValueError("unexpected source authorization SHA")
    if args.replaces_job_id != EXPECTED_REPLACED_JOB[shard]:
        raise ValueError("replacement job identity mismatch")
    expected_source = {
        "stage": "dge_same_identity_supplement",
        "code_root": str(SOURCE_CODE_ROOT),
        "execution_allowed": True,
        "explicit_user_execution_authorization": True,
        "method_shard": shard,
        "methods": list(METHOD_SHARDS[shard]),
        "expected_rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "operator_dev_accessed_for_fit": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for key, expected in expected_source.items():
        if source.get(key) != expected:
            raise ValueError(f"source authorization mismatch: {key}")
    _verify_record(source.get("model_manifest"), "model_manifest")
    if source["model_manifest"]["sha256"] != source.get("model_manifest_sha256"):
        raise ValueError("model manifest binding mismatch")
    _verify_source_records(source.get("sources"))
    if source.get("retry_after_verified_runtime_restore") is not True:
        raise ValueError("source authorization lacks verified runtime-retry lineage")

    failed_run = REMOTE_ROOT / "runs/qwen3-8b-dge-same-identity-operators-20260729T093107Z"
    failed_status = json.loads((failed_run / "exit_status.json").read_text())
    failed_test = failed_run / "tests.txt"
    if failed_status.get("job_id") != "9173" or failed_status.get("exit_code") != 1:
        raise ValueError("bundle-completeness failure status mismatch")
    if failed_status.get("hostname") != "a06" or (failed_run / "COMPLETE").exists():
        raise ValueError("bundle-completeness failure allocation mismatch")
    failed_text = failed_test.read_text()
    missing_name = "QWEN3_8B_CONSENSUS_V5_BEHAVIOR_SUPPLEMENT_V1.json"
    if "FileNotFoundError" not in failed_text or missing_name not in failed_text:
        raise ValueError("unexpected bundle-completeness failure cause")

    code_bindings = {
        "supplement_contract_sha256": args.code_root / "protocol/QWEN3_8B_DGE_SAME_IDENTITY_SUPPLEMENT_V1.json",
        "crossover_contract_sha256": args.code_root / "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json",
        "benchmark_manifest_sha256": args.code_root / "artifacts/benchmark_manifest.jsonl",
        "design_audit_sha256": args.code_root / "artifacts/governance_crossover_operator_dev_design_audit_v2.json",
    }
    for key, path in code_bindings.items():
        if not path.is_file() or sha256_file(path) != source.get(key):
            raise ValueError(f"code-v49 scientific binding changed: {key}")
    consensus_path = args.code_root / "protocol/QWEN3_8B_CONSENSUS_V5_BEHAVIOR_SUPPLEMENT_V1.json"
    consensus_sha = "c5994684d8fac6dd0c8dfbfbde742557e5558dfb9a3eadbf95c59660555226b5"
    if not consensus_path.is_file() or sha256_file(consensus_path) != consensus_sha:
        raise ValueError("consensus behavior supplement binding mismatch")

    selection = json.loads(args.selector_snapshot.read_text())
    if selection.get("node") != args.selected_node:
        raise ValueError("selector snapshot/node mismatch")
    if selection.get("selection_contract") != EXPECTED_AGGREGATE_CONTRACT:
        raise ValueError("aggregate selector contract mismatch")
    if _node_states(str(selection.get("state", ""))) & BLOCKED_NODE_STATES:
        raise ValueError("selector snapshot contains a blocked node state")
    if selection.get("partition") != "a01" or selection.get("npu_type") != "910B3":
        raise ValueError("selector snapshot partition/NPU mismatch")
    if int(selection.get("npu_free", -1)) < 1:
        raise ValueError("selector snapshot lacks NPU capacity")
    if int(selection.get("cpu_free", -1)) < 8:
        raise ValueError("selector snapshot lacks CPU capacity")
    if int(selection.get("mem_free_mib", -1)) < 131072:
        raise ValueError("selector snapshot lacks memory capacity")

    value = dict(source)
    value.update(
        {
            "schema_version": 2,
            "authorization_id": f"qwen3-8b-dge-same-identity-{shard}-code-v49",
            "created_utc": args.created_utc,
            "code_root": str(args.code_root),
            "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
            "authorization_materializer_root": str(args.materializer_root),
            "authorization_materializer_bundle_manifest_sha256": sha256_file(materializer_bundle),
            "authorization_materializer": _record(Path(__file__)),
            "execution_node": args.selected_node,
            "slurm_partition": "a01",
            "resource_contract": {
                **PER_JOB_RESOURCE_CONTRACT,
                "node": args.selected_node,
            },
            "node_selection_snapshot": _record(args.selector_snapshot),
            "aggregate_migration_resource_contract": EXPECTED_AGGREGATE_CONTRACT,
            "parent_authorization": _record(args.source_authorization),
            "replaces_job_id": args.replaces_job_id,
            "orchestration_only_migration": False,
            "scientific_identity_unchanged": False,
            "diagnostics_interface_patch": True,
            "bundle_completeness_patch": True,
            "consensus_v5_behavior_supplement_sha256": consensus_sha,
            "scientific_output_semantics_unchanged": True,
            "failed_bundle_completeness_attempt": {
                "job_id": 9173,
                "run_id": failed_run.name,
                "exit_status": _record(failed_run / "exit_status.json"),
                "tests": _record(failed_test),
                "rows_completed": 0,
            },
            "execution_allowed": True,
            "explicit_user_execution_authorization": True,
            "operator_dev_accessed_for_fit": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
