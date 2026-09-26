#!/usr/bin/env python3
"""Materialize the immutable V5 protected-capture recovery authorization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


HOME_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = WORK_ROOT / "code-v52"
SOURCE_RUN = (
    HOME_ROOT
    / "runs/qwen3-8b-governance-consensus-router-audit-capture-20260729T064709Z"
)
SOURCE_AUTH = SOURCE_RUN / "QWEN3_8B_GOVERNANCE_CONSENSUS_ROUTER_AUDIT_CAPTURE_CODE_V36.json"
SOURCE_GOVERNANCE_MANIFEST = SOURCE_RUN / "governance_audit_capture/capture_manifest.json"
SOURCE_EXIT_STATUS = SOURCE_RUN / "exit_status.json"
SOURCE_AUTH_SHA256 = "1503ec72ad6e1db673c7af71691a739e5606f74da178d78b3364e7a83b993940"
SOURCE_GOVERNANCE_MANIFEST_SHA256 = (
    "76d6b0be89efbbdec61e237f1589ddf2a51eba54c743a5c903bca204827eae78"
)
SOURCE_EXIT_STATUS_SHA256 = "5b239a05dea8901c58537fcb011c3db33336bfdb93c35585337b4d010b16ae10"
FAILED_RECOVERY_RUN = (
    WORK_ROOT
    / "runs/qwen3-8b-governance-consensus-router-capture-recovery-20260729T133415Z"
)
FAILED_RECOVERY_AUTH_SHA256 = "94503f4d7cfd95fc3162800341e67caf6b8b46bf098b26ef28063f7b14e7c649"
FAILED_RECOVERY_EXIT_SHA256 = "de4b5d1950210a5f246fd8f3d7671a0bdadbc1f6c9cd57d77783dc6d4f0880ab"
FAILED_RECOVERY_TESTS_SHA256 = "b66059f89c08667134442d0892febcf5501fabd9857a6be809766558abf627fd"
EXPECTED_KEY_SHA256 = "4e6301fce193060c68bb284f7b3349aec61766eb1daa60ff85d200ef9d981900"
EXPECTED_FAMILIES = {
    "fresh": 768,
    "verification": 1536,
    "obedience_reset": 1536,
    "supported_user_authority": 144,
    "factual_boundary_memory": 24,
}
SELECTION_CONTRACT = {
    "partition": "a01",
    "npu_type": "910B3",
    "npus": 1,
    "cpus": 8,
    "mem_mib": 131072,
    "allow_nodes": [],
    "exclude_nodes": [],
}
RESOURCE_CONTRACT = {
    "partition": "a01",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 8,
    "mem_mib": 131072,
    "npu_type": "910B3",
    "npus": 1,
    "time_limit": "08:00:00",
}
BLOCKED_NODE_STATES = {
    "DOWN", "DRAIN", "DRAINED", "FAIL", "FAILING", "FUTURE", "INVAL",
    "MAINT", "NO_RESPOND", "PLANNED", "POWER_DOWN", "POWERING_DOWN",
    "REBOOT_REQUESTED", "RESERVED", "UNKNOWN",
}


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = str(path)
    if not value.startswith((f"{HOME_ROOT}/", f"{WORK_ROOT}/")):
        raise ValueError(f"artifact is outside the authorized roots: {path}")
    return {"path": value, "sha256": sha256_file(path)}


def _node_states(value: str) -> set[str]:
    return {token for token in re.split(r"[+~#$*!%-]+", value.upper()) if token}


def validate_selection(selection: dict, selected_node: str) -> None:
    if not re.fullmatch(r"a[0-9]{2}", selected_node):
        raise ValueError("invalid selected node")
    if selection.get("node") != selected_node:
        raise ValueError("selector snapshot/node mismatch")
    if selection.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError("selector resource contract mismatch")
    if _node_states(str(selection.get("state", ""))) & BLOCKED_NODE_STATES:
        raise ValueError("selector snapshot contains a blocked node state")
    for field, minimum in (("npu_free", 1), ("cpu_free", 8), ("mem_free_mib", 131072)):
        if int(selection.get(field, -1)) < minimum:
            raise ValueError(f"selector snapshot lacks capacity: {field}")


def validate_governance_manifest(manifest: dict, source_auth_sha256: str) -> None:
    required = {
        "complete": True,
        "stage": "governance_consensus_router_audit_capture",
        "crossover_stage": "discovery",
        "partition": "component_discovery",
        "rows": 6144,
        "unique_job_keys": 6144,
        "counterfactual_pairs": 3072,
        "shard_count": 24,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "observed_key_sha256": EXPECTED_KEY_SHA256,
        "authorization_sha256": source_auth_sha256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"source governance manifest mismatch: {field}")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 24:
        raise ValueError("source governance shard count mismatch")
    if sum(int(record.get("rows", -1)) for record in shards) != 6144:
        raise ValueError("source governance shard row sum mismatch")
    names = set()
    for record in shards:
        name = str(record.get("file", ""))
        digest = str(record.get("sha256", ""))
        if Path(name).name != name or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("unsafe governance shard record")
        if name in names:
            raise ValueError("duplicate governance shard name")
        names.add(name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != CODE_ROOT:
        raise ValueError("V5 recovery authorization requires immutable code-v52")
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        args.created_utc,
    ):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    bundle = args.code_root / "bundle.sha256"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    for path, expected in (
        (SOURCE_AUTH, SOURCE_AUTH_SHA256),
        (SOURCE_GOVERNANCE_MANIFEST, SOURCE_GOVERNANCE_MANIFEST_SHA256),
        (SOURCE_EXIT_STATUS, SOURCE_EXIT_STATUS_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source recovery binding mismatch: {path}")

    source_auth = json.loads(SOURCE_AUTH.read_text())
    source_status = json.loads(SOURCE_EXIT_STATUS.read_text())
    source_manifest = json.loads(SOURCE_GOVERNANCE_MANIFEST.read_text())
    for field, expected in {
        "stage": "governance_consensus_router_audit_capture",
        "code_root": str(HOME_ROOT / "code-v36"),
        "execution_allowed": True,
        "expected_governance_rows": 6144,
        "expected_protected_rows": 4008,
        "expected_governance_key_sha256": EXPECTED_KEY_SHA256,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if source_auth.get(field) != expected:
            raise ValueError(f"source authorization mismatch: {field}")
    for field, expected in {
        "job_id": "9140",
        "exit_code": 1,
        "hostname": "a07",
        "stage": "governance_consensus_router_audit_capture",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if source_status.get(field) != expected:
            raise ValueError(f"source exit status mismatch: {field}")
    if (SOURCE_RUN / "COMPLETE").exists():
        raise ValueError("source failed run unexpectedly contains COMPLETE")
    if (SOURCE_RUN / "protected_audit_capture/capture_manifest.json").exists():
        raise ValueError("source failed run unexpectedly contains protected evidence")
    validate_governance_manifest(source_manifest, SOURCE_AUTH_SHA256)
    for record in source_manifest["shards"]:
        path = SOURCE_GOVERNANCE_MANIFEST.parent / record["file"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ValueError(f"source governance shard SHA mismatch: {path}")

    selection = json.loads(args.selector_snapshot.read_text())
    validate_selection(selection, args.selected_node)
    code_files = {
        "router_contract_sha256": "protocol/QWEN3_8B_GROUP_ROBUST_CONSENSUS_GOVERNANCE_EDITOR_V5.json",
        "editor_contract_sha256": "protocol/QWEN3_8B_ADAPTIVE_GOVERNANCE_EDITOR_V1.json",
        "crossover_contract_sha256": "protocol/GOVERNANCE_TASK_CROSSOVER_V1.json",
        "operator_site_manifest_sha256": "artifacts/QWEN3_8B_OPERATOR_SITE_MANIFEST_V1.json",
        "benchmark_manifest_sha256": "artifacts/benchmark_manifest.jsonl",
        "component_discovery_design_audit_sha256": "artifacts/governance_crossover_discovery_design_audit_v2.json",
        "v5_audit_controls_sha256": "protocol/mitigation_controls_v5_audit.json",
        "mechanism_contract_sha256": "protocol/QWEN3_8B_MECHANISM_DISCOVERY_CONTRACT_V1.json",
        "operator_contract_sha256": "protocol/QWEN3_8B_OPERATOR_CONTRACT_V1.json",
        "execution_contract_sha256": "protocol/QWEN3_8B_OPERATOR_SITE_AND_EXECUTION_CONTRACT_V1.json",
        "behavior_analysis_sha256": "artifacts/qwen3-8b-full-behavior-analysis.json",
    }
    bindings = {}
    for field, relative in code_files.items():
        path = args.code_root / relative
        value = sha256_file(path)
        if source_auth.get(field) != value:
            raise ValueError(f"scientific input changed from source capture: {field}")
        bindings[field] = value
    if source_manifest.get("router_contract_sha256") != bindings["router_contract_sha256"]:
        raise ValueError("source manifest/router SHA mismatch")
    if source_manifest.get("model_manifest_sha256") != source_auth["model_manifest"]["sha256"]:
        raise ValueError("source manifest/model SHA mismatch")
    model_manifest = Path(source_auth["model_manifest"]["path"])
    if sha256_file(model_manifest) != source_auth["model_manifest"]["sha256"]:
        raise ValueError("source model manifest SHA mismatch")

    failed_auth = FAILED_RECOVERY_RUN / "execution_authorization.json"
    failed_exit = FAILED_RECOVERY_RUN / "exit_status.json"
    failed_tests = FAILED_RECOVERY_RUN / "tests.txt"
    for path, expected in (
        (failed_auth, FAILED_RECOVERY_AUTH_SHA256),
        (failed_exit, FAILED_RECOVERY_EXIT_SHA256),
        (failed_tests, FAILED_RECOVERY_TESTS_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"failed recovery lineage mismatch: {path}")
    failed_status = json.loads(failed_exit.read_text())
    if failed_status.get("job_id") != "9194" or failed_status.get("exit_code") != 1:
        raise ValueError("failed recovery exit identity mismatch")
    failed_text = failed_tests.read_text()
    if failed_text.count("ModuleNotFoundError") != 4 or "/WORK/" not in failed_text:
        raise ValueError("unexpected failed recovery cause")
    if (FAILED_RECOVERY_RUN / "governance_audit_capture").exists() or (
        FAILED_RECOVERY_RUN / "protected_audit_capture"
    ).exists():
        raise ValueError("failed recovery unexpectedly produced capture output")

    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-governance-consensus-router-capture-recovery-code-v52",
        "created_utc": args.created_utc,
        "stage": "governance_consensus_router_capture_recovery",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        **bindings,
        "design_audit_sha256": bindings["component_discovery_design_audit_sha256"],
        "governance_crossover_stage": "discovery",
        "governance_partition": "component_discovery",
        "expected_governance_rows": 6144,
        "expected_rows": 6144,
        "expected_governance_key_sha256": EXPECTED_KEY_SHA256,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "protected_partition": "component_discovery",
        "expected_protected_rows": 4008,
        "expected_protected_family_rows": EXPECTED_FAMILIES,
        "source_job_id": 9140,
        "source_run_id": SOURCE_RUN.name,
        "source_run_root": str(SOURCE_RUN),
        "source_governance_capture_root": str(SOURCE_GOVERNANCE_MANIFEST.parent),
        "source_governance_manifest": _record(SOURCE_GOVERNANCE_MANIFEST),
        "source_capture_authorization": _record(SOURCE_AUTH),
        "source_exit_status": _record(SOURCE_EXIT_STATUS),
        "source_job_overall_success": False,
        "source_governance_stage_complete": True,
        "source_protected_stage_rows": 0,
        "source_failure_after_governance_capture": "execution/mitigation-controls mismatch",
        "recompute_governance_forward": False,
        "replaces_failed_recovery_job_id": 9194,
        "engineering_retry_only": True,
        "scientific_inputs_unchanged": True,
        "failed_recovery_attempt": {
            "run_id": FAILED_RECOVERY_RUN.name,
            "authorization": _record(failed_auth),
            "exit_status": _record(failed_exit),
            "tests": _record(failed_tests),
            "rows_completed": 0,
            "failure": "unittest absolute /WORK path import parsing",
        },
        "model_manifest": dict(source_auth["model_manifest"]),
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "resource_contract": {**RESOURCE_CONTRACT, "node": args.selected_node},
        "node_selection_snapshot": _record(args.selector_snapshot),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "authorization": str(args.output),
        "authorization_sha256": sha256_file(args.output),
        "selected_node": args.selected_node,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
