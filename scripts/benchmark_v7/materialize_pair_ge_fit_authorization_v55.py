#!/usr/bin/env python3
"""Materialize the immutable SHA-bound PAIR-GE V5.2 CPU fit authorization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


HOME_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = WORK_ROOT / "code-v55"
ROUTER_SHA256 = "93d01ad16b0ca6437b26d80576a2b06acd8097e7e4fb1f780e73cea7f793fca0"
FIT_SHA256 = "b445b7e200dd302ffa48cc76144a333fab2c406a631010f70aa4022115e1e978"
GOVERNANCE_AUTH_SHA256 = "1503ec72ad6e1db673c7af71691a739e5606f74da178d78b3364e7a83b993940"
PROTECTED_AUTH_SHA256 = "10cc2518844d46ab510141d4f4ffd99afc1b86685f58a8729c0c12649f2e1195"
GOVERNANCE_MANIFEST_SHA256 = "76d6b0be89efbbdec61e237f1589ddf2a51eba54c743a5c903bca204827eae78"
PROTECTED_MANIFEST_SHA256 = "c1f542061e8de2c65e9ee1de9bf3d6f331acec67040c59d6644be795dc184409"
CAPTURE_RECEIPT_SHA256 = "683b572c6df6768b93279a0739eaa15048772d4f9f182559226a9b3f8cad9157"
V5_1_REPORT_SHA256 = "f5c028e07ab0d26bb9f0dc0fb074739434fbe29c68e719bfb591df84305d30b5"
V5_1_RECEIPT_SHA256 = "2389d32f9961162d5d79436775e62fee72a6d808ec2c40f70b2d7b4b14310dd2"
SELECTION_CONTRACT = {
    "partition": "a01",
    "npu_type": "910B3",
    "npus": 0,
    "cpus": 32,
    "mem_mib": 196608,
    "allow_nodes": [],
    "exclude_nodes": [],
}
RESOURCE_CONTRACT = {
    "partition": "a01",
    "nodes": 1,
    "ntasks": 1,
    "cpus_per_task": 32,
    "mem_mib": 196608,
    "npu_type": "910B3",
    "npus": 0,
    "time_limit": "06:00:00",
}
BLOCKED_NODE_STATES = {
    "DOWN",
    "DRAIN",
    "DRAINED",
    "FAIL",
    "FAILING",
    "FUTURE",
    "INVAL",
    "MAINT",
    "NO_RESPOND",
    "PLANNED",
    "POWER_DOWN",
    "POWERING_DOWN",
    "REBOOT_REQUESTED",
    "RESERVED",
    "UNKNOWN",
}


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = str(path)
    if not value.startswith((f"{HOME_ROOT}/", f"{WORK_ROOT}/")):
        raise ValueError(f"artifact is outside authorized roots: {path}")
    return {"path": value, "sha256": sha256_file(path)}


def _states(value: str) -> set[str]:
    return {token for token in re.split(r"[+~#$*!%-]+", value.upper()) if token}


def _validate_selection(selection: dict, selected_node: str) -> None:
    if not re.fullmatch(r"a[0-9]{2}", selected_node):
        raise ValueError("invalid selected node")
    if selection.get("node") != selected_node:
        raise ValueError("selector snapshot/node mismatch")
    if selection.get("selection_contract") != SELECTION_CONTRACT:
        raise ValueError("selector resource contract mismatch")
    if _states(str(selection.get("state", ""))) & BLOCKED_NODE_STATES:
        raise ValueError("selector selected a blocked node")
    for field, minimum in (("cpu_free", 32), ("mem_free_mib", 196608)):
        if int(selection.get(field, -1)) < minimum:
            raise ValueError(f"selector snapshot lacks capacity: {field}")


def _validate_contracts(router: dict, fit: dict) -> None:
    for field, expected in {
        "status": "frozen_after_terminal_v5_1_falsifier_before_any_v5_2_fit",
        "method_short_name": "PAIR-GE-V5.2",
        "code_version_minimum": 55,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if router.get(field) != expected:
            raise ValueError(f"PAIR-GE router contract mismatch: {field}")
    for field, expected in {
        "status": "frozen_before_any_v5_2_fit_execution",
        "method_short_name": "PAIR-GE-V5.2",
        "code_version_minimum": 55,
        "router_contract_sha256": ROUTER_SHA256,
        "component_discovery_is_developmental_only": True,
        "operator_dev_untouched_confirmatory_selection_required": True,
        "architecture_frozen_before_v5_2_fit": True,
        "seed_frozen_before_v5_2_fit": True,
        "epoch_frozen_before_v5_2_fit": True,
        "threshold_rule_frozen_before_v5_2_fit": True,
    }.items():
        if fit.get(field) != expected:
            raise ValueError(f"PAIR-GE fit contract mismatch: {field}")
    if fit.get("training", {}).get("hidden_widths") != [96, 48]:
        raise ValueError("PAIR-GE hidden-width contract mismatch")
    if fit.get("training", {}).get("epochs") != 180:
        raise ValueError("PAIR-GE epoch contract mismatch")


def _safe(value: dict, *, name: str) -> None:
    for field, expected in {
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"{name} safety mismatch: {field}")


def _validate_capture_receipt(receipt: dict) -> None:
    for field, expected in {
        "schema_version": 1,
        "job_id": 9203,
        "run_id": "qwen3-8b-governance-consensus-router-capture-recovery-20260729T134830Z",
        "archive_sha256": "0f51da9d3960a3e72dcebe9f362287f2795066936bcf5b7534c19a9f4e040c2e",
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }.items():
        if receipt.get(field) != expected:
            raise ValueError(f"capture receipt mismatch: {field}")
    record = str(receipt.get("slurm_terminal_record", ""))
    for marker in ("JobId=9203", "JobState=COMPLETED", "ExitCode=0:0"):
        if marker not in record:
            raise ValueError(f"capture receipt lacks terminal marker: {marker}")


def _validate_v5_1(report: dict, receipt: dict) -> None:
    if report.get("method") != "GRC-DGE-V5" or report.get("fit_eligible") is not False:
        raise ValueError("V5.1 scientific falsifier mismatch")
    for field, expected in {
        "job_id": 9205,
        "archive_sha256": "6e038c155b88410a0065d3d504faf505e706751c7f4083956729b252582f0778",
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }.items():
        if receipt.get(field) != expected:
            raise ValueError(f"V5.1 receipt mismatch: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--router-contract", type=Path, required=True)
    parser.add_argument("--fit-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--train-governance-manifest", type=Path, required=True)
    parser.add_argument("--train-protected-manifest", type=Path, required=True)
    parser.add_argument("--audit-governance-manifest", type=Path, required=True)
    parser.add_argument("--audit-protected-manifest", type=Path, required=True)
    parser.add_argument("--governance-capture-authorization", type=Path, required=True)
    parser.add_argument("--protected-capture-authorization", type=Path, required=True)
    parser.add_argument("--capture-archive-receipt", type=Path, required=True)
    parser.add_argument("--v5-1-fit-report", type=Path, required=True)
    parser.add_argument("--v5-1-fit-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing PAIR-GE authorization: {args.output}")
    if args.code_root != CODE_ROOT:
        raise ValueError("PAIR-GE authorization requires immutable code-v55")
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        args.created_utc,
    ):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    bundle = args.code_root / "bundle.sha256"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    for path, expected in (
        (args.router_contract, ROUTER_SHA256),
        (args.fit_contract, FIT_SHA256),
        (args.governance_capture_authorization, GOVERNANCE_AUTH_SHA256),
        (args.protected_capture_authorization, PROTECTED_AUTH_SHA256),
        (args.audit_governance_manifest, GOVERNANCE_MANIFEST_SHA256),
        (args.audit_protected_manifest, PROTECTED_MANIFEST_SHA256),
        (args.capture_archive_receipt, CAPTURE_RECEIPT_SHA256),
        (args.v5_1_fit_report, V5_1_REPORT_SHA256),
        (args.v5_1_fit_receipt, V5_1_RECEIPT_SHA256),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"immutable PAIR-GE input SHA mismatch: {path}")

    router = json.loads(args.router_contract.read_text())
    fit = json.loads(args.fit_contract.read_text())
    governance_auth = json.loads(args.governance_capture_authorization.read_text())
    protected_auth = json.loads(args.protected_capture_authorization.read_text())
    governance = json.loads(args.audit_governance_manifest.read_text())
    protected = json.loads(args.audit_protected_manifest.read_text())
    capture_receipt = json.loads(args.capture_archive_receipt.read_text())
    v5_1_report = json.loads(args.v5_1_fit_report.read_text())
    v5_1_receipt = json.loads(args.v5_1_fit_receipt.read_text())
    _validate_contracts(router, fit)
    if governance_auth.get("stage") != "governance_consensus_router_audit_capture":
        raise ValueError("governance capture authorization stage mismatch")
    if protected_auth.get("stage") != "governance_consensus_router_capture_recovery":
        raise ValueError("protected capture authorization stage mismatch")
    _safe(governance_auth, name="governance capture authorization")
    _safe(protected_auth, name="protected capture authorization")
    for manifest, rows, auth_sha, name in (
        (governance, 6144, GOVERNANCE_AUTH_SHA256, "governance"),
        (protected, 4008, PROTECTED_AUTH_SHA256, "protected"),
    ):
        for field, expected in {
            "partition": "component_discovery",
            "rows": rows,
            "authorization_sha256": auth_sha,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }.items():
            if manifest.get(field) != expected:
                raise ValueError(f"{name} developmental manifest mismatch: {field}")
    _validate_capture_receipt(capture_receipt)
    _validate_v5_1(v5_1_report, v5_1_receipt)
    selection = json.loads(args.selector_snapshot.read_text())
    _validate_selection(selection, args.selected_node)

    sources = {
        "v3_checkpoint": _record(args.v3_checkpoint),
        "v3_fit_report": _record(args.v3_fit_report),
        "train_governance_manifest": _record(args.train_governance_manifest),
        "train_protected_manifest": _record(args.train_protected_manifest),
        "audit_governance_manifest": _record(args.audit_governance_manifest),
        "audit_protected_manifest": _record(args.audit_protected_manifest),
        "governance_capture_authorization": _record(
            args.governance_capture_authorization
        ),
        "protected_capture_authorization": _record(
            args.protected_capture_authorization
        ),
        "capture_archive_receipt": _record(args.capture_archive_receipt),
        "v5_1_fit_report": _record(args.v5_1_fit_report),
        "v5_1_fit_receipt": _record(args.v5_1_fit_receipt),
    }
    prior = router["bound_prior_evidence"]
    captures = router["bound_capture_evidence"]
    for field, source in (
        ("v3_checkpoint_sha256", sources["v3_checkpoint"]),
        ("v3_fit_report_sha256", sources["v3_fit_report"]),
        ("v5_1_fit_report_sha256", sources["v5_1_fit_report"]),
        ("v5_1_fit_receipt_sha256", sources["v5_1_fit_receipt"]),
    ):
        if prior.get(field) != source["sha256"]:
            raise ValueError(f"PAIR-GE prior source mismatch: {field}")
    for field, source in (
        ("subspace_fit_governance_manifest_sha256", sources["train_governance_manifest"]),
        ("subspace_fit_protected_manifest_sha256", sources["train_protected_manifest"]),
        ("developmental_governance_manifest_sha256", sources["audit_governance_manifest"]),
        ("developmental_protected_manifest_sha256", sources["audit_protected_manifest"]),
        ("capture_receipt_sha256", sources["capture_archive_receipt"]),
    ):
        if captures.get(field) != source["sha256"]:
            raise ValueError(f"PAIR-GE capture source mismatch: {field}")

    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-governance-pair-interaction-fit-code-v55",
        "created_utc": args.created_utc,
        "stage": "governance_pair_interaction_fit",
        "method": "PAIR-GE-V5.2",
        "code_root": str(CODE_ROOT),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "execution_node": args.selected_node,
        "resource_contract": {**RESOURCE_CONTRACT, "node": args.selected_node},
        "node_selection_snapshot": _record(args.selector_snapshot),
        "router_contract_sha256": ROUTER_SHA256,
        "fit_contract_sha256": FIT_SHA256,
        "v3_contract_sha256": sha256_file(args.v3_contract),
        **sources,
        **{f"{name}_sha256": source["sha256"] for name, source in sources.items()},
        "governance_capture_authorization_sha256": GOVERNANCE_AUTH_SHA256,
        "protected_capture_authorization_sha256": PROTECTED_AUTH_SHA256,
        "capture_archive_receipt_sha256": CAPTURE_RECEIPT_SHA256,
        "v5_1_fit_report_sha256": V5_1_REPORT_SHA256,
        "v5_1_fit_receipt_sha256": V5_1_RECEIPT_SHA256,
        "component_discovery_confirmatory": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "authorization": str(args.output),
                "authorization_sha256": sha256_file(args.output),
                "selected_node": args.selected_node,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
