#!/usr/bin/env python3
"""Create one immutable SHA-bound C-DGE-V4.1 composite eligibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
EXPECTED_LINEAGE = {
    "v3_contract_sha256": "a531ad1b3cb20b964f240f6eb3c513f6e1a1b7d749c61b79796833193129873d",
    "v3_checkpoint_sha256": "ed5262bbd27470047c3379172e58de23e6bb98a542b9fb5101f828138b11d752",
    "v3_fit_report_sha256": "0cda38b3f3ad44460e6f91278277bc1cd66349a5d34f128890d6bc60f60a5f0c",
    "v4_contract_sha256": "0f6d33ab67a02fe88f8d14aa092f5993636d3a99b3a939c2339de0fab4eb677a",
    "v4_checkpoint_sha256": "42e3af0a3501750514c98f70305ac380663cac99ffeb8b26274340e10b7fcaf8",
    "v4_fit_report_sha256": "f79e133e5cb0910df5a4a5281067571e25e551e5f49f0d0caa7dabf8d6e65c4a",
    "governance_capture_manifest_sha256": "987b8b9d17c9d723d1ddcbaba61cc28d5b585c31e20b63e1a05c0dc94872735d",
    "protected_capture_manifest_sha256": "f084c379cf59b5eeffdb67f62ce26d2cbb7418919da5e93a297dea1030a7c690",
}


def _artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--selector-snapshot", type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--v4-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-fit-report", type=Path, required=True)
    parser.add_argument("--v4-fit-authorization", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != WORK_ROOT / "code-v59":
        raise ValueError("composite eligibility audit requires exact code-v59")
    if not (
        len(args.selected_node) == 3
        and args.selected_node[0] == "a"
        and args.selected_node[1:].isdigit()
    ):
        raise ValueError("invalid selected node")

    composite = json.loads(args.composite_contract.read_text())
    fit = json.loads(args.v4_fit_report.read_text())
    old_authorization = json.loads(args.v4_fit_authorization.read_text())
    if composite.get("method_short_name") != "C-DGE-V4.1":
        raise ValueError("unexpected composite contract")
    if composite.get("status") != "prospective_composite_eligibility_audit_frozen_before_execution":
        raise ValueError("composite contract is not frozen for execution")
    if composite.get("evidence_role") != "developmental_protocol_correction_after_terminal_v4_diagnostics":
        raise ValueError("composite evidence role mismatch")
    if fit.get("fit_complete") is not True or fit.get("fit_eligible") is not False:
        raise ValueError("terminal V4 fit failure is not preserved")
    if fit.get("checkpoint_sha256") != sha256_file(args.v4_checkpoint):
        raise ValueError("V4 checkpoint/report mismatch")
    old_authorization_sha = sha256_file(args.v4_fit_authorization)
    if fit.get("authorization_sha256") != old_authorization_sha:
        raise ValueError("V4 fit authorization/report mismatch")
    for field, expected in {
        "stage": "governance_abstaining_router_fit",
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if old_authorization.get(field) != expected:
            raise ValueError(f"old V4 fit authorization mismatch: {field}")

    path_by_lineage = {
        "v3_contract_sha256": args.v3_contract,
        "v3_checkpoint_sha256": args.v3_checkpoint,
        "v3_fit_report_sha256": args.v3_fit_report,
        "v4_contract_sha256": args.v4_contract,
        "v4_checkpoint_sha256": args.v4_checkpoint,
        "v4_fit_report_sha256": args.v4_fit_report,
        "governance_capture_manifest_sha256": args.capture_manifest,
        "protected_capture_manifest_sha256": args.protected_capture_manifest,
    }
    for field, path in path_by_lineage.items():
        observed = sha256_file(path)
        if observed != EXPECTED_LINEAGE[field]:
            raise ValueError(f"source lineage mismatch: {field}")
        if composite["frozen_lineage"].get(field) != observed:
            raise ValueError(f"contract lineage mismatch: {field}")

    snapshot = json.loads(args.selector_snapshot.read_text())
    expected_selection = {
        "allow_nodes": [],
        "cpus": 16,
        "exclude_nodes": [],
        "mem_mib": 196608,
        "npu_type": "910B3",
        "npus": 0,
        "partition": "a01",
    }
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    if snapshot.get("selection_contract") != expected_selection:
        raise ValueError("selector resource contract mismatch")
    if int(snapshot.get("cpu_free", -1)) < 16:
        raise ValueError("selector snapshot lacks requested CPU capacity")
    if int(snapshot.get("mem_free_mib", -1)) < 196608:
        raise ValueError("selector snapshot lacks requested memory")

    bundle = args.code_root / "bundle.sha256"
    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-cdge-v4-1-composite-eligibility-code-v59",
        "created_utc": args.created_utc,
        "stage": "governance_composite_eligibility_audit",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "resource_contract": {
            "partition": "a01",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 16,
            "mem_mib": 196608,
            "npu_type": "910B3",
            "npus": 0,
            "time_limit": "02:00:00",
            "node": args.selected_node,
        },
        "node_selection_snapshot": _artifact(args.selector_snapshot),
        "composite_contract": _artifact(args.composite_contract),
        "v4_contract": _artifact(args.v4_contract),
        "v3_contract": _artifact(args.v3_contract),
        "v4_checkpoint": _artifact(args.v4_checkpoint),
        "v4_fit_report": _artifact(args.v4_fit_report),
        "v4_fit_authorization": _artifact(args.v4_fit_authorization),
        "v3_checkpoint": _artifact(args.v3_checkpoint),
        "v3_fit_report": _artifact(args.v3_fit_report),
        "capture_manifest": _artifact(args.capture_manifest),
        "protected_capture_manifest": _artifact(args.protected_capture_manifest),
        "expected_total_rows": 10152,
        "candidate_eligibility_pending": True,
        "candidate_may_be_locked": False,
        "confirmatory": False,
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
