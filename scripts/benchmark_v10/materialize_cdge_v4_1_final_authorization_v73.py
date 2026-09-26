#!/usr/bin/env python3
"""Materialize the one-time immutable C-DGE-V4.1 final authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v10.cdge_contract import (
    FINAL_EXPECTED_KEY_SHA256,
    FINAL_EXPECTED_ROWS,
    FINAL_STAGE,
    METHOD,
    RUNTIME_METHOD,
    _require,
)


WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")


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
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--pareto-lock", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--archive-helper", type=Path, required=True)
    parser.add_argument("--archive-coordinator", type=Path, required=True)
    parser.add_argument("--pull-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C-DGE final authorization: {args.output}")
    if args.code_root != WORK_ROOT / "code-v73":
        raise ValueError("C-DGE final authorization requires exact code-v73")
    bundle = args.code_root / "bundle.sha256"
    lock = json.loads(args.pareto_lock.read_text())
    fit = json.loads(args.fit_report.read_text())
    evaluation = json.loads(args.evaluation_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    model = json.loads(args.model_manifest.read_text())
    _require(
        lock,
        {
            "manifest_id": "context-mismatch-qwen3-8b-cdge-v4-1-lock-v1",
            "method": METHOD,
            "locked": True,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "candidate_may_be_locked": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "C-DGE Pareto lock",
    )
    _require(
        fit,
        {
            "method": RUNTIME_METHOD,
            "fit_complete": True,
            "fit_eligible": False,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "historical V4 fit report",
    )
    for name, path in {
        "runtime_checkpoint": args.checkpoint,
        "fit_report": args.fit_report,
        "editor_contract": args.editor_contract,
        "composite_contract": args.composite_contract,
        "evaluation_contract": args.evaluation_contract,
    }.items():
        if lock.get(name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"Pareto lock source mismatch: {name}")
    _require(
        evaluation.get("bound_artifacts", {}),
        {
            "v4_contract_sha256": sha256_file(args.editor_contract),
            "v4_1_composite_contract_sha256": sha256_file(args.composite_contract),
            "v4_checkpoint_sha256": sha256_file(args.checkpoint),
            "v4_fit_report_sha256": sha256_file(args.fit_report),
            "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
            "final_test_design_audit_sha256": sha256_file(args.design_audit),
            "model_manifest_sha256": sha256_file(args.model_manifest),
        },
        "post-eligibility final binding",
    )
    if design.get("stage") != "final_test" or design.get("audit", {}).get("success") is not True:
        raise ValueError("final-test design audit failed")
    if design.get("audit", {}).get("row_count") != FINAL_EXPECTED_ROWS:
        raise ValueError("final-test design row mismatch")
    expected_key = design.get("audit", {}).get("expected_key_sha256")
    if expected_key != FINAL_EXPECTED_KEY_SHA256:
        raise ValueError("final-test design key mismatch")
    if not (model.get("verified") or model.get("success")):
        raise ValueError("model manifest is unverified")
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    if not (
        len(args.selected_node) == 3
        and args.selected_node[0] == "a"
        and args.selected_node[1:].isdigit()
    ):
        raise ValueError("invalid selected node")
    expected_selection = {
        "allow_nodes": [],
        "cpus": 8,
        "exclude_nodes": [],
        "mem_mib": 131072,
        "npu_type": "910B3",
        "npus": 1,
        "partition": "a01",
    }
    if snapshot.get("selection_contract") != expected_selection:
        raise ValueError("selector resource contract mismatch")
    if int(snapshot.get("npu_free", -1)) < 1 or int(snapshot.get("cpu_free", -1)) < 8:
        raise ValueError("selector snapshot lacks compute capacity")
    if int(snapshot.get("mem_free_mib", -1)) < 131072:
        raise ValueError("selector snapshot lacks memory capacity")
    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-cdge-v4-1-final-code-v73-once",
        "created_utc": args.created_utc,
        "stage": FINAL_STAGE,
        "method": METHOD,
        "runtime_checkpoint_method": RUNTIME_METHOD,
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "resource_contract": {
            "partition": "a01",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "mem_mib": 131072,
            "npu_type": "910B3",
            "npus": 1,
            "time_limit": "12:00:00",
            "node": args.selected_node,
        },
        "node_selection_snapshot": _artifact(args.selector_snapshot),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "pareto_lock": _artifact(args.pareto_lock),
        "pareto_lock_sha256": sha256_file(args.pareto_lock),
        "editor_lock_sha256": sha256_file(args.pareto_lock),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "design_audit_sha256": sha256_file(args.design_audit),
        "checkpoint": _artifact(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report": _artifact(args.fit_report),
        "fit_report_sha256": sha256_file(args.fit_report),
        "model_manifest": _artifact(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "monitoring_contract": {
            "poll_seconds": 1,
            "continuous_watch_required": True,
            "sacct_required_for_recovery_only": False,
            "archive_helper": _artifact(args.archive_helper),
            "archive_coordinator": _artifact(args.archive_coordinator),
            "pull_helper": _artifact(args.pull_helper),
        },
        "expected_rows": FINAL_EXPECTED_ROWS,
        "expected_key_sha256": FINAL_EXPECTED_KEY_SHA256,
        "candidate_id": "C-DGE-V4.1-27:mlp",
        "composite_eligible": True,
        "candidate_eligible": True,
        "candidate_may_be_locked": True,
        "final_test_open": True,
        "final_test_open_count": 1,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
