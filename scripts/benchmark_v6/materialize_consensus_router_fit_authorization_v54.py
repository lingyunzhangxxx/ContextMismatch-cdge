#!/usr/bin/env python3
"""Create the immutable, dual-lineage V5.1 consensus-router fit authorization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


HOME_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = WORK_ROOT / "code-v54"
ROUTER_SHA256 = "7829eedebb61de8168cd465afb47def0005339fd333316d5d20d1cdc11f73142"
INHERITED_FIT_SHA256 = "f6c7a654dbe19f473603257d8080d273a7297fa0982c6c0da7fdea0cf93ef9ec"
RECOVERY_FIT_SHA256 = "573484e663c668543ee85192d3c17e7ed1519e86c24bfe6ea615026ea18b806e"
GOVERNANCE_AUTH_SHA256 = "1503ec72ad6e1db673c7af71691a739e5606f74da178d78b3364e7a83b993940"
PROTECTED_AUTH_SHA256 = "10cc2518844d46ab510141d4f4ffd99afc1b86685f58a8729c0c12649f2e1195"
GOVERNANCE_MANIFEST_SHA256 = "76d6b0be89efbbdec61e237f1589ddf2a51eba54c743a5c903bca204827eae78"
EXPECTED_KEY_SHA256 = "4e6301fce193060c68bb284f7b3349aec61766eb1daa60ff85d200ef9d981900"
CAPTURE_ARCHIVE_SHA256 = "0f51da9d3960a3e72dcebe9f362287f2795066936bcf5b7534c19a9f4e040c2e"
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


def _states(value: str) -> set[str]:
    return {token for token in re.split(r"[+~#$*!%-]+", value.upper()) if token}


def _safe(value: dict, *, name: str) -> None:
    for field, expected in {
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"{name} safety mismatch: {field}")


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


def _validate_contracts(router: dict, inherited: dict, recovery: dict) -> None:
    for field, expected in {
        "status": "frozen_before_any_v5_capture_or_fit",
        "method_short_name": "GRC-DGE-V5",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if router.get(field) != expected:
            raise ValueError(f"router contract mismatch: {field}")
    for field, expected in {
        "status": "frozen_while_v5_capture_job_9113_pending_before_any_capture_output",
        "method_short_name": "GRC-DGE-V5",
        "capture_outputs_examined_before_freeze": False,
        "architecture_search_forbidden": True,
        "seed_search_forbidden": True,
        "epoch_search_forbidden": True,
        "threshold_search_forbidden": True,
    }.items():
        if inherited.get(field) != expected:
            raise ValueError(f"inherited fit contract mismatch: {field}")
    for field, expected in {
        "status": "recovery_lineage_frozen_after_protected_forward_started_before_any_protected_output_was_examined",
        "method_short_name": "GRC-DGE-V5",
        "recovery_revision": "V5.1",
        "code_version_minimum": 54,
        "scientific_hyperparameters_inherited_without_change": True,
        "source_governance_authorization_sha256": GOVERNANCE_AUTH_SHA256,
        "source_governance_manifest_sha256": GOVERNANCE_MANIFEST_SHA256,
        "protected_recovery_authorization_sha256": PROTECTED_AUTH_SHA256,
        "protected_recovery_manifest_bound_by_later_fit_authorization": True,
        "protected_forward_started_before_recovery_lineage_freeze": True,
        "protected_outputs_examined_before_recovery_lineage_freeze": False,
        "capture_outputs_examined_before_freeze": False,
        "architecture_search_forbidden": True,
        "seed_search_forbidden": True,
        "epoch_search_forbidden": True,
        "threshold_search_forbidden": True,
    }.items():
        if recovery.get(field) != expected:
            raise ValueError(f"recovery fit contract mismatch: {field}")
    for section in (
        "training", "cross_fit", "final_fit", "required_audit_sources", "gates", "safety"
    ):
        if recovery.get(section) != inherited.get(section):
            raise ValueError(f"scientific fit contract changed: {section}")


def _validate_manifest(
    manifest: dict,
    *,
    name: str,
    rows: int,
    authorization_sha256: str,
) -> None:
    for field, expected in {
        "partition": "component_discovery",
        "rows": rows,
        "authorization_sha256": authorization_sha256,
        "router_contract_sha256": ROUTER_SHA256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if manifest.get(field) != expected:
            raise ValueError(f"{name} audit manifest mismatch: {field}")


def _validate_receipt(receipt: dict) -> None:
    for field, expected in {
        "schema_version": 1,
        "job_id": 9203,
        "run_id": "qwen3-8b-governance-consensus-router-capture-recovery-20260729T134830Z",
        "archive_sha256": CAPTURE_ARCHIVE_SHA256,
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }.items():
        if receipt.get(field) != expected:
            raise ValueError(f"capture archive receipt mismatch: {field}")
    terminal = str(receipt.get("slurm_terminal_record", ""))
    for marker in ("JobId=9203", "JobState=COMPLETED", "ExitCode=0:0"):
        if marker not in terminal:
            raise ValueError(f"capture receipt lacks terminal marker: {marker}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--router-contract", type=Path, required=True)
    parser.add_argument("--inherited-fit-contract", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing V5.1 fit authorization: {args.output}")
    if args.code_root != CODE_ROOT:
        raise ValueError("V5.1 fit authorization requires immutable code-v54")
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        args.created_utc,
    ):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    bundle = args.code_root / "bundle.sha256"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    expected_files = (
        (args.router_contract, ROUTER_SHA256),
        (args.inherited_fit_contract, INHERITED_FIT_SHA256),
        (args.fit_contract, RECOVERY_FIT_SHA256),
        (args.governance_capture_authorization, GOVERNANCE_AUTH_SHA256),
        (args.protected_capture_authorization, PROTECTED_AUTH_SHA256),
        (args.audit_governance_manifest, GOVERNANCE_MANIFEST_SHA256),
    )
    for path, expected in expected_files:
        if sha256_file(path) != expected:
            raise ValueError(f"immutable input SHA mismatch: {path}")

    router = json.loads(args.router_contract.read_text())
    inherited = json.loads(args.inherited_fit_contract.read_text())
    recovery = json.loads(args.fit_contract.read_text())
    governance_auth = json.loads(args.governance_capture_authorization.read_text())
    protected_auth = json.loads(args.protected_capture_authorization.read_text())
    governance = json.loads(args.audit_governance_manifest.read_text())
    protected = json.loads(args.audit_protected_manifest.read_text())
    receipt = json.loads(args.capture_archive_receipt.read_text())
    _validate_contracts(router, inherited, recovery)
    if inherited.get("router_contract_sha256") != ROUTER_SHA256:
        raise ValueError("inherited fit/router SHA mismatch")
    if recovery.get("router_contract_sha256") != ROUTER_SHA256:
        raise ValueError("recovery fit/router SHA mismatch")
    if recovery.get("inherited_pre_capture_fit_contract_sha256") != INHERITED_FIT_SHA256:
        raise ValueError("recovery inherited-fit SHA mismatch")
    if governance_auth.get("stage") != "governance_consensus_router_audit_capture":
        raise ValueError("unexpected governance capture authorization stage")
    if protected_auth.get("stage") != "governance_consensus_router_capture_recovery":
        raise ValueError("unexpected protected capture authorization stage")
    _safe(governance_auth, name="governance authorization")
    _safe(protected_auth, name="protected authorization")
    for field, expected in {
        "source_job_id": 9140,
        "source_job_overall_success": False,
        "source_governance_stage_complete": True,
        "source_protected_stage_rows": 0,
        "recompute_governance_forward": False,
        "engineering_retry_only": True,
        "scientific_inputs_unchanged": True,
    }.items():
        if protected_auth.get(field) != expected:
            raise ValueError(f"protected recovery authorization mismatch: {field}")
    _validate_manifest(
        governance, name="governance", rows=6144,
        authorization_sha256=GOVERNANCE_AUTH_SHA256,
    )
    _validate_manifest(
        protected, name="protected", rows=4008,
        authorization_sha256=PROTECTED_AUTH_SHA256,
    )
    for field, expected in {
        "complete": True,
        "unique_job_keys": 6144,
        "counterfactual_pairs": 3072,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "observed_key_sha256": EXPECTED_KEY_SHA256,
    }.items():
        if governance.get(field) != expected:
            raise ValueError(f"governance audit identity mismatch: {field}")
    if protected.get("family_rows") != EXPECTED_FAMILIES:
        raise ValueError("protected audit family count mismatch")
    _validate_receipt(receipt)

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
    }
    prior = router["bound_prior_evidence"]
    static = router["bound_static_inputs"]
    for field, source in (
        ("v3_checkpoint_sha256", sources["v3_checkpoint"]),
        ("v3_fit_report_sha256", sources["v3_fit_report"]),
    ):
        if prior.get(field) != source["sha256"]:
            raise ValueError(f"V3 evidence mismatch: {field}")
    for field, source in (
        (
            "subspace_fit_governance_capture_manifest_sha256",
            sources["train_governance_manifest"],
        ),
        (
            "subspace_fit_protected_capture_manifest_sha256",
            sources["train_protected_manifest"],
        ),
    ):
        if static.get(field) != source["sha256"]:
            raise ValueError(f"static training evidence mismatch: {field}")
    if sources["audit_governance_manifest"]["sha256"] != GOVERNANCE_MANIFEST_SHA256:
        raise ValueError("governance manifest source record mismatch")

    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-governance-consensus-router-fit-code-v54",
        "created_utc": args.created_utc,
        "stage": "governance_consensus_router_fit",
        "recovery_revision": "V5.1",
        "code_root": str(CODE_ROOT),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "execution_node": args.selected_node,
        "resource_contract": {**RESOURCE_CONTRACT, "node": args.selected_node},
        "node_selection_snapshot": _record(args.selector_snapshot),
        "router_contract_sha256": ROUTER_SHA256,
        "fit_contract_sha256": RECOVERY_FIT_SHA256,
        "inherited_fit_contract_sha256": INHERITED_FIT_SHA256,
        "v3_contract_sha256": sha256_file(args.v3_contract),
        **sources,
        **{f"{name}_sha256": source["sha256"] for name, source in sources.items()},
        "governance_capture_authorization_sha256": GOVERNANCE_AUTH_SHA256,
        "protected_capture_authorization_sha256": PROTECTED_AUTH_SHA256,
        "capture_archive_receipt_sha256": sources["capture_archive_receipt"][
            "sha256"
        ],
        "dual_capture_authorization_lineage_verified": True,
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
