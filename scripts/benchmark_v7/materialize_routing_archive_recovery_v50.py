#!/usr/bin/env python3
"""Materialize a fresh routing authorization after loss of live Slurm evidence.

This control-plane-only retry reuses the immutable code-v49 scientific bundle.
It verifies that Job 9172 produced complete run-local scientific evidence, but
does not treat that evidence as a substitute for a terminal Slurm record.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


REMOTE_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = REMOTE_ROOT / "code-v49"
CONTROL_ROOT = REMOTE_ROOT / "code-v50-control"
SOURCE_AUTH = (
    REMOTE_ROOT
    / "authorizations/QWEN3_8B_DGE_SAME_IDENTITY_ROUTING_CODE_V40_RETRY_20260729T092241Z.json"
)
PRIOR_RUN = (
    REMOTE_ROOT
    / "runs/qwen3-8b-dge-same-identity-routing-20260729T092241Z"
)
SOURCE_AUTH_SHA256 = "242fa40cc121888ecff47f52cdb8dd7ad25db08a0377aaa7778b2e214175e94f"
PRIOR_EXIT_SHA256 = "741dc10e5f938e656ebd64c80ad02066ebb4e5bb27c6e1fdb749aa0c23246027"
PRIOR_AUDIT_SHA256 = "c2521a67a3b3d8cf615ed46508bb16254ec86fe7cdbe6fdf34aa88e2525bcf11"
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
CONSENSUS_SHA256 = "c5994684d8fac6dd0c8dfbfbde742557e5558dfb9a3eadbf95c59660555226b5"
METHODS = ["negative_only_expert", "dge_without_structural_routing"]
SELECTION_CONTRACT = {
    "partition": "a01",
    "npu_type": "910B3",
    "npus": 1,
    "cpus": 8,
    "mem_mib": 131072,
    "allow_nodes": ["a06"],
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
    "time_limit": "12:00:00",
    "node": "a06",
}
BLOCKED_NODE_STATES = {
    "DOWN", "DRAIN", "DRAINED", "FAIL", "FAILING", "FUTURE", "INVAL",
    "MAINT", "NO_RESPOND", "PLANNED", "POWER_DOWN", "POWERING_DOWN",
    "REBOOT_REQUESTED", "RESERVED", "UNKNOWN",
}


def _record(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not str(path).startswith(f"{REMOTE_ROOT}/"):
        raise ValueError(f"artifact outside project root: {path}")
    return {"path": str(path), "sha256": sha256_file(path)}


def _verify_record(record: object, label: str) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"invalid record: {label}")
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"invalid digest: {label}")
    if not str(path).startswith(f"{REMOTE_ROOT}/") or sha256_file(path) != digest:
        raise ValueError(f"record SHA mismatch: {label}")


def _verify_nested_records(value: object, label: str = "sources") -> None:
    if isinstance(value, dict) and set(value) == {"path", "sha256"}:
        _verify_record(value, label)
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _verify_nested_records(child, f"{label}.{key}")
        return
    raise ValueError(f"unexpected source binding: {label}")


def _node_states(value: str) -> set[str]:
    return {token for token in re.split(r"[+~#$*!%-]+", value.upper()) if token}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--source-authorization", type=Path, required=True)
    parser.add_argument("--prior-run", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != CODE_ROOT or args.control_root != CONTROL_ROOT:
        raise ValueError("unexpected immutable code/control root")
    if args.source_authorization != SOURCE_AUTH or args.prior_run != PRIOR_RUN:
        raise ValueError("unexpected recovery lineage")
    if args.selected_node != "a06":
        raise ValueError("archive-recovery retry requires verified a06 runtime")
    if sha256_file(args.source_authorization) != SOURCE_AUTH_SHA256:
        raise ValueError("source authorization SHA mismatch")
    if sha256_file(args.prior_run / "exit_status.json") != PRIOR_EXIT_SHA256:
        raise ValueError("prior exit-status SHA mismatch")
    if sha256_file(args.prior_run / "same_identity/shard_audit.json") != PRIOR_AUDIT_SHA256:
        raise ValueError("prior shard-audit SHA mismatch")
    if not (args.prior_run / "COMPLETE").is_file():
        raise ValueError("prior run lacks COMPLETE marker")

    code_bundle = args.code_root / "bundle.sha256"
    control_bundle = args.control_root / "bundle.sha256"
    if not code_bundle.is_file() or not control_bundle.is_file():
        raise FileNotFoundError("immutable bundle manifest missing")

    source = json.loads(args.source_authorization.read_text())
    expected_source = {
        "stage": "dge_same_identity_supplement",
        "method_shard": "routing",
        "methods": METHODS,
        "expected_rows_per_method": 3072,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "execution_allowed": True,
        "explicit_user_execution_authorization": True,
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
    _verify_nested_records(source.get("sources"))

    prior_status = json.loads((args.prior_run / "exit_status.json").read_text())
    prior_audit = json.loads((args.prior_run / "same_identity/shard_audit.json").read_text())
    if prior_status != {
        "job_id": "9172",
        "exit_code": 0,
        "hostname": "a06",
        "stage": "dge_same_identity_supplement",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }:
        raise ValueError("prior run-local status mismatch")
    if (
        prior_audit.get("success") is not True
        or prior_audit.get("method_shard") != "routing"
        or prior_audit.get("methods") != METHODS
        or prior_audit.get("rows_per_method") != 3072
        or prior_audit.get("expected_key_sha256") != EXPECTED_KEY_SHA256
        or prior_audit.get("paired_baseline_logits_identical") is not True
    ):
        raise ValueError("prior scientific audit mismatch")

    selection = json.loads(args.selector_snapshot.read_text())
    if selection.get("node") != args.selected_node:
        raise ValueError("selector node mismatch")
    if selection.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError("selector contract mismatch")
    if _node_states(str(selection.get("state", ""))) & BLOCKED_NODE_STATES:
        raise ValueError("selector snapshot contains blocked state")
    if int(selection.get("npu_free", -1)) < 1:
        raise ValueError("selector snapshot lacks NPU capacity")
    if int(selection.get("cpu_free", -1)) < 8:
        raise ValueError("selector snapshot lacks CPU capacity")
    if int(selection.get("mem_free_mib", -1)) < 131072:
        raise ValueError("selector snapshot lacks memory capacity")

    consensus = args.code_root / "protocol/QWEN3_8B_CONSENSUS_V5_BEHAVIOR_SUPPLEMENT_V1.json"
    if sha256_file(consensus) != CONSENSUS_SHA256:
        raise ValueError("consensus supplement SHA mismatch")

    value = dict(source)
    value.update(
        {
            "schema_version": 3,
            "authorization_id": "qwen3-8b-dge-same-identity-routing-archive-recovery-code-v50",
            "created_utc": args.created_utc,
            "code_root": str(args.code_root),
            "immutable_code_bundle_manifest_sha256": sha256_file(code_bundle),
            "authorization_materializer_root": str(args.control_root),
            "authorization_materializer_bundle_manifest_sha256": sha256_file(control_bundle),
            "authorization_materializer": _record(Path(__file__)),
            "execution_node": args.selected_node,
            "slurm_partition": "a01",
            "resource_contract": RESOURCE_CONTRACT,
            "node_selection_snapshot": _record(args.selector_snapshot),
            "aggregate_migration_resource_contract": SELECTION_CONTRACT,
            "parent_authorization": _record(args.source_authorization),
            "replaces_job_id": "9172",
            "orchestration_only_migration": False,
            "scientific_identity_unchanged": False,
            "diagnostics_interface_patch": True,
            "bundle_completeness_patch": True,
            "consensus_v5_behavior_supplement_sha256": CONSENSUS_SHA256,
            "scientific_output_semantics_unchanged": True,
            "archive_recovery_rerun": True,
            "prior_run_local_evidence": {
                "run_id": args.prior_run.name,
                "job_id": 9172,
                "complete": _record(args.prior_run / "COMPLETE"),
                "exit_status": _record(args.prior_run / "exit_status.json"),
                "shard_audit": _record(args.prior_run / "same_identity/shard_audit.json"),
                "terminal_slurm_record_available": False,
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
