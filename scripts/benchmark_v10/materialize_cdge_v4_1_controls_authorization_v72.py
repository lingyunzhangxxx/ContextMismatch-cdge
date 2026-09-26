#!/usr/bin/env python3
"""Materialize one immutable C-DGE-V4.1 protected-controls authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v10.cdge_contract import (
    CONTROLS_EXPECTED_KEY_SHA256,
    CONTROLS_EXPECTED_ROWS,
    CONTROLS_STAGE,
    METHOD,
    RUNTIME_METHOD,
    _require,
    validate_behavior_prerequisite,
)


WORK_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
FAMILIES = {
    "fresh_verification",
    "matched_verification",
    "matched_delegated_choice",
    "explicit_governance_reset",
    "supported_user_authority",
    "factual_boundary_memory",
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
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--candidate-behavior-analysis", type=Path, required=True)
    parser.add_argument("--candidate-behavior-receipt", type=Path, required=True)
    parser.add_argument("--v3-controls-analysis", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--archive-helper", type=Path, required=True)
    parser.add_argument("--archive-coordinator", type=Path, required=True)
    parser.add_argument("--pull-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C-DGE controls authorization: {args.output}")
    if args.code_root != WORK_ROOT / "code-v72":
        raise ValueError("C-DGE protected controls require exact code-v72")

    bundle = args.code_root / "bundle.sha256"
    editor = json.loads(args.editor_contract.read_text())
    composite = json.loads(args.composite_contract.read_text())
    evaluation = json.loads(args.evaluation_contract.read_text())
    behavior = json.loads(args.candidate_behavior_analysis.read_text())
    behavior_receipt = json.loads(args.candidate_behavior_receipt.read_text())
    v3 = json.loads(args.v3_controls_analysis.read_text())
    fit = json.loads(args.fit_report.read_text())
    model = json.loads(args.model_manifest.read_text())
    validate_behavior_prerequisite(behavior, behavior_receipt)
    if editor.get("method_short_name") != RUNTIME_METHOD:
        raise ValueError("unexpected runtime editor contract")
    if composite.get("method_short_name") != METHOD:
        raise ValueError("unexpected composite editor contract")
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
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 controls comparison is not audited")
    if not (model.get("verified") or model.get("success")):
        raise ValueError("model manifest is unverified")
    bound = evaluation.get("bound_artifacts", {})
    _require(
        bound,
        {
            "v4_contract_sha256": sha256_file(args.editor_contract),
            "v4_1_composite_contract_sha256": sha256_file(args.composite_contract),
            "v4_checkpoint_sha256": sha256_file(args.checkpoint),
            "v4_fit_report_sha256": sha256_file(args.fit_report),
            "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
            "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
            "model_manifest_sha256": sha256_file(args.model_manifest),
        },
        "evaluation contract binding",
    )
    controls_contract = evaluation.get("protected_controls", {})
    _require(
        controls_contract,
        {
            "stage": CONTROLS_STAGE,
            "expected_rows": CONTROLS_EXPECTED_ROWS,
            "expected_key_sha256": CONTROLS_EXPECTED_KEY_SHA256,
            "forced_directions": ["positive", "negative"],
            "application_gated_identity_required": True,
            "forced_direction_reference_gates_required": True,
            "all_rows_unique_and_finite_required": True,
        },
        "protected-controls evaluation contract",
    )
    if set(controls_contract.get("control_families", [])) != FAMILIES:
        raise ValueError("protected-controls family contract mismatch")
    archive_path = Path(behavior_receipt.get("cluster_shared_path", ""))
    if not archive_path.is_file() or sha256_file(archive_path) != behavior_receipt["archive_sha256"]:
        raise ValueError("cluster shared behavior archive is missing or changed")

    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    if not (len(args.selected_node) == 3 and args.selected_node[0] == "a" and args.selected_node[1:].isdigit()):
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
        "authorization_id": "qwen3-8b-cdge-v4-1-protected-controls-code-v72",
        "created_utc": args.created_utc,
        "stage": CONTROLS_STAGE,
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
            "time_limit": "10:00:00",
            "node": args.selected_node,
        },
        "node_selection_snapshot": _artifact(args.selector_snapshot),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "candidate_behavior_analysis": _artifact(args.candidate_behavior_analysis),
        "candidate_behavior_analysis_sha256": sha256_file(args.candidate_behavior_analysis),
        "candidate_behavior_receipt": _artifact(args.candidate_behavior_receipt),
        "candidate_behavior_receipt_sha256": sha256_file(args.candidate_behavior_receipt),
        "candidate_behavior_archive_sha256": behavior_receipt["archive_sha256"],
        "v3_controls_analysis": _artifact(args.v3_controls_analysis),
        "v3_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "controls_sha256": sha256_file(args.controls),
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
        "expected_rows": CONTROLS_EXPECTED_ROWS,
        "expected_key_sha256": CONTROLS_EXPECTED_KEY_SHA256,
        "forced_directions": ["positive", "negative"],
        "original_v4_fit_eligible": False,
        "composite_eligible": True,
        "candidate_eligible": True,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
