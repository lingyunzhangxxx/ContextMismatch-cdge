#!/usr/bin/env python3
"""Materialize one immutable C-DGE-V4.1 operator-dev authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v10.cdge_contract import (
    ARCHIVE_SHA256,
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    METHOD,
    STAGE,
    _require,
    validate_composite_prerequisite,
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
    parser.add_argument("--composite-report", type=Path, required=True)
    parser.add_argument("--composite-authorization", type=Path, required=True)
    parser.add_argument("--composite-receipt", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--v3-operator-dev-analysis", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--archive-helper", type=Path, required=True)
    parser.add_argument("--archive-coordinator", type=Path, required=True)
    parser.add_argument("--pull-helper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C-DGE authorization: {args.output}")
    if args.code_root != WORK_ROOT / "code-v70":
        raise ValueError("C-DGE behavior requires exact code-v70")

    bundle = args.code_root / "bundle.sha256"
    editor = json.loads(args.editor_contract.read_text())
    composite_contract = json.loads(args.composite_contract.read_text())
    evaluation = json.loads(args.evaluation_contract.read_text())
    composite_report = json.loads(args.composite_report.read_text())
    composite_authorization = json.loads(args.composite_authorization.read_text())
    composite_receipt = json.loads(args.composite_receipt.read_text())
    fit = json.loads(args.fit_report.read_text())
    v3 = json.loads(args.v3_operator_dev_analysis.read_text())
    design = json.loads(args.design_audit.read_text())
    model = json.loads(args.model_manifest.read_text())
    validate_composite_prerequisite(composite_report, composite_receipt)
    if editor.get("method_short_name") != "ADSGE-V4":
        raise ValueError("unexpected runtime editor contract")
    if composite_contract.get("method_short_name") != METHOD:
        raise ValueError("unexpected composite contract")
    _require(
        fit,
        {
            "method": "ADSGE-V4",
            "fit_complete": True,
            "fit_eligible": False,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "historical fit report",
    )
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 operator-dev analysis is not audited")
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev design audit failed")
    if not (model.get("verified") or model.get("success")):
        raise ValueError("model manifest is unverified")
    if composite_report.get("authorization_sha256") != sha256_file(args.composite_authorization):
        raise ValueError("composite report/authorization mismatch")
    if composite_authorization.get("stage") != "governance_composite_eligibility_audit":
        raise ValueError("unexpected composite authorization stage")
    if composite_authorization.get("final_test_open") is not False:
        raise ValueError("composite authorization opened final test")
    bound = evaluation.get("bound_artifacts", {})
    _require(
        bound,
        {
            "v4_contract_sha256": sha256_file(args.editor_contract),
            "v4_1_composite_contract_sha256": sha256_file(args.composite_contract),
            "v4_checkpoint_sha256": sha256_file(args.checkpoint),
            "v4_fit_report_sha256": sha256_file(args.fit_report),
            "composite_eligibility_report_sha256": sha256_file(args.composite_report),
            "composite_eligibility_authorization_sha256": sha256_file(args.composite_authorization),
            "composite_eligibility_receipt_sha256": sha256_file(args.composite_receipt),
            "composite_eligibility_archive_sha256": ARCHIVE_SHA256,
            "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
            "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
            "operator_dev_design_audit_sha256": sha256_file(args.design_audit),
            "v3_operator_dev_analysis_sha256": sha256_file(args.v3_operator_dev_analysis),
            "model_manifest_sha256": sha256_file(args.model_manifest),
        },
        "evaluation contract binding",
    )
    _require(
        evaluation.get("operator_dev", {}),
        {
            "stage": STAGE,
            "expected_rows": EXPECTED_ROWS,
            "expected_key_sha256": EXPECTED_KEY_SHA256,
        },
        "operator-dev evaluation",
    )
    archive_path = Path(composite_receipt.get("cluster_shared_path", ""))
    if not archive_path.is_file() or sha256_file(archive_path) != ARCHIVE_SHA256:
        raise ValueError("cluster shared composite archive is missing or changed")

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
        "authorization_id": "qwen3-8b-cdge-v4-1-behavior-code-v70",
        "created_utc": args.created_utc,
        "stage": STAGE,
        "method": METHOD,
        "runtime_checkpoint_method": "ADSGE-V4",
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
            "time_limit": "08:00:00",
            "node": args.selected_node,
        },
        "node_selection_snapshot": _artifact(args.selector_snapshot),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "composite_report": _artifact(args.composite_report),
        "composite_report_sha256": sha256_file(args.composite_report),
        "composite_authorization": _artifact(args.composite_authorization),
        "composite_authorization_sha256": sha256_file(args.composite_authorization),
        "composite_receipt": _artifact(args.composite_receipt),
        "composite_receipt_sha256": sha256_file(args.composite_receipt),
        "composite_archive_sha256": ARCHIVE_SHA256,
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "design_audit_sha256": sha256_file(args.design_audit),
        "checkpoint": _artifact(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report": _artifact(args.fit_report),
        "fit_report_sha256": sha256_file(args.fit_report),
        "v3_operator_dev_analysis": _artifact(args.v3_operator_dev_analysis),
        "v3_operator_dev_analysis_sha256": sha256_file(args.v3_operator_dev_analysis),
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
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
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
