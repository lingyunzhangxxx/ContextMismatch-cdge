#!/usr/bin/env python3
"""Materialize the code-v61 CANN-loading recovery of the frozen C-DGE audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v9.materialize_composite_eligibility_authorization_v59 import (
    EXPECTED_LINEAGE,
)


WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
BASE_CONTRACT_SHA = "d3c92a4f467c070d5bcba1159d388ea4f714c5b64ce0b7691486e1bd781e8cf3"
RECOVERY_CONTRACT_SHA = "6d5641331365b926c8b8e46c53fbe92572d88d20ed1e95901593c4052ce1574d"
FAILED_ARTIFACT_SHAS = {
    "failed_authorization": "1132b477547d4b8e8ed45776818290d745c8f6295fe151a616d0d543b5b965b5",
    "failed_tests": "a5ee0e602e46b106da39d94256c6b6ffd43968dcba320e7c26e9a1605eac7330",
    "failed_exit_status": "e7a2d954e337135a1bd1ddbf7e705907feba388a87f3b68084ce5b4589932ff6",
    "failed_held_job_record": "f219875e7236eddaf97a96f9c1a1704e5c6cb4308e6ef160e73bd4ac7c3d0a0e",
    "failed_report": "21319f9dc87965bfc67129784d129c3318e0e2610eab6ce89fceb7af2fca6050",
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
    parser.add_argument("--recovery-contract", type=Path, required=True)
    parser.add_argument("--v4-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-fit-report", type=Path, required=True)
    parser.add_argument("--v4-fit-authorization", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--failed-authorization", type=Path, required=True)
    parser.add_argument("--failed-tests", type=Path, required=True)
    parser.add_argument("--failed-exit-status", type=Path, required=True)
    parser.add_argument("--failed-held-job-record", type=Path, required=True)
    parser.add_argument("--failed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_root != WORK_ROOT / "code-v61":
        raise ValueError("engineering recovery requires exact code-v61")
    if not (
        len(args.selected_node) == 3
        and args.selected_node[0] == "a"
        and args.selected_node[1:].isdigit()
    ):
        raise ValueError("invalid selected node")

    composite = json.loads(args.composite_contract.read_text())
    recovery = json.loads(args.recovery_contract.read_text())
    fit = json.loads(args.v4_fit_report.read_text())
    old_authorization = json.loads(args.v4_fit_authorization.read_text())
    if sha256_file(args.composite_contract) != BASE_CONTRACT_SHA:
        raise ValueError("base composite contract changed")
    if sha256_file(args.recovery_contract) != RECOVERY_CONTRACT_SHA:
        raise ValueError("code-v61 recovery contract changed")
    if recovery.get("status") != "engineering_recovery_frozen_before_code_v61_execution":
        raise ValueError("recovery contract is not frozen")
    if recovery.get("base_scientific_contract_sha256") != BASE_CONTRACT_SHA:
        raise ValueError("recovery/base contract mismatch")
    change = recovery.get("only_authorized_change", {})
    if change.get("source_environment") != "/usr/local/Ascend/cann/set_env.sh":
        raise ValueError("unexpected recovery environment source")
    if change.get("torch_device_backend_autoload_disabled") is not False:
        raise ValueError("backend autoload must remain enabled")
    if change.get("npu_resource_requested") is not False:
        raise ValueError("code-v61 recovery must remain CPU-only")
    if change.get("scientific_python_unchanged") is not True:
        raise ValueError("unexpected recovery change")
    if composite.get("method_short_name") != "C-DGE-V4.1":
        raise ValueError("unexpected composite contract")
    if fit.get("fit_complete") is not True or fit.get("fit_eligible") is not False:
        raise ValueError("terminal V4 fit failure is not preserved")
    if fit.get("checkpoint_sha256") != sha256_file(args.v4_checkpoint):
        raise ValueError("V4 checkpoint/report mismatch")
    if fit.get("authorization_sha256") != sha256_file(args.v4_fit_authorization):
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
            raise ValueError(f"old V4 authorization mismatch: {field}")

    source_paths = {
        "v3_contract_sha256": args.v3_contract,
        "v3_checkpoint_sha256": args.v3_checkpoint,
        "v3_fit_report_sha256": args.v3_fit_report,
        "v4_contract_sha256": args.v4_contract,
        "v4_checkpoint_sha256": args.v4_checkpoint,
        "v4_fit_report_sha256": args.v4_fit_report,
        "governance_capture_manifest_sha256": args.capture_manifest,
        "protected_capture_manifest_sha256": args.protected_capture_manifest,
    }
    for field, path in source_paths.items():
        observed = sha256_file(path)
        if observed != EXPECTED_LINEAGE[field]:
            raise ValueError(f"source lineage mismatch: {field}")
        if composite["frozen_lineage"].get(field) != observed:
            raise ValueError(f"composite lineage mismatch: {field}")
    failed_paths = {
        "failed_authorization": args.failed_authorization,
        "failed_tests": args.failed_tests,
        "failed_exit_status": args.failed_exit_status,
        "failed_held_job_record": args.failed_held_job_record,
        "failed_report": args.failed_report,
    }
    for name, path in failed_paths.items():
        if sha256_file(path) != FAILED_ARTIFACT_SHAS[name]:
            raise ValueError(f"failed-run lineage mismatch: {name}")
    failed_status = json.loads(args.failed_exit_status.read_text())
    if failed_status.get("job_id") != "9262" or failed_status.get("exit_code") != 1:
        raise ValueError("unexpected failed-run status")
    failed_tests = args.failed_tests.read_text()
    if "Ran 31 tests" not in failed_tests or "skipped=16" not in failed_tests:
        raise ValueError("failed-run signature mismatch")
    failed_report = json.loads(args.failed_report.read_text())
    if failed_report.get("candidate_eligible") is not True:
        raise ValueError("failed diagnostic report eligibility changed")
    failed_job = recovery.get("failed_job", {})
    if failed_job.get("job_id") != 9262 or failed_job.get("formal_scientific_result_accepted") is not False:
        raise ValueError("failed-job recovery lineage mismatch")
    for field, path in {
        "authorization_sha256": args.failed_authorization,
        "tests_sha256": args.failed_tests,
        "exit_status_sha256": args.failed_exit_status,
        "held_job_record_sha256": args.failed_held_job_record,
        "diagnostic_report_sha256": args.failed_report,
    }.items():
        if failed_job.get(field) != sha256_file(path):
            raise ValueError(f"recovery failed-job SHA mismatch: {field}")

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
    if int(snapshot.get("cpu_free", -1)) < 16 or int(snapshot.get("mem_free_mib", -1)) < 196608:
        raise ValueError("selector snapshot lacks capacity")

    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-cdge-v4-1-composite-eligibility-code-v61-recovery",
        "created_utc": args.created_utc,
        "stage": "governance_composite_eligibility_audit",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
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
        "recovery_contract": _artifact(args.recovery_contract),
        "v4_contract": _artifact(args.v4_contract),
        "v3_contract": _artifact(args.v3_contract),
        "v4_checkpoint": _artifact(args.v4_checkpoint),
        "v4_fit_report": _artifact(args.v4_fit_report),
        "v4_fit_authorization": _artifact(args.v4_fit_authorization),
        "v3_checkpoint": _artifact(args.v3_checkpoint),
        "v3_fit_report": _artifact(args.v3_fit_report),
        "capture_manifest": _artifact(args.capture_manifest),
        "protected_capture_manifest": _artifact(args.protected_capture_manifest),
        **{name: _artifact(path) for name, path in failed_paths.items()},
        "engineering_recovery_of_job_id": 9262,
        "cann_runtime_environment_loaded": True,
        "torch_device_backend_autoload_disabled": False,
        "previous_diagnostic_report_accepted": False,
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
