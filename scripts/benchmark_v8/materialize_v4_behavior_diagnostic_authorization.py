#!/usr/bin/env python3
"""Create one immutable SHA-bound ADSGE-V4 operator_dev diagnostic authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


EXPECTED_ROWS = 3072
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
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
    parser.add_argument("--diagnostic-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--v3-operator-dev-analysis", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing V4 diagnostic authorization: {args.output}")
    if args.code_root != WORK_ROOT / "code-v57":
        raise ValueError("V4 behavior diagnostic requires exact code-v57")
    bundle = args.code_root / "bundle.sha256"
    editor = json.loads(args.editor_contract.read_text())
    diagnostic = json.loads(args.diagnostic_contract.read_text())
    fit = json.loads(args.fit_report.read_text())
    v3_analysis = json.loads(args.v3_operator_dev_analysis.read_text())
    design = json.loads(args.design_audit.read_text())
    model = json.loads(args.model_manifest.read_text())
    if editor.get("method_short_name") != "ADSGE-V4":
        raise ValueError("unexpected editor contract")
    if fit.get("method") != "ADSGE-V4" or fit.get("fit_eligible") is not False:
        raise ValueError("V4 fit failure is not preserved")
    if fit.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("V4 checkpoint/report mismatch")
    if v3_analysis.get("method") != "DSGE-V3":
        raise ValueError("unexpected V3 operator_dev analysis")
    if v3_analysis.get("audit", {}).get("success") is not True:
        raise ValueError("V3 operator_dev analysis audit failed")
    for field, expected in {
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if diagnostic.get(field) != expected:
            raise ValueError(f"diagnostic contract mismatch: {field}")
    evaluation = diagnostic.get("evaluation", {})
    if evaluation.get("expected_rows") != EXPECTED_ROWS:
        raise ValueError("diagnostic row contract mismatch")
    if evaluation.get("expected_key_sha256") != EXPECTED_KEY_SHA256:
        raise ValueError("diagnostic key contract mismatch")
    bound = diagnostic.get("bound_artifacts", {})
    expected_bound = {
        "v4_editor_contract_sha256": sha256_file(args.editor_contract),
        "v4_checkpoint_sha256": sha256_file(args.checkpoint),
        "v4_fit_report_sha256": sha256_file(args.fit_report),
        "v3_operator_dev_analysis_sha256": sha256_file(args.v3_operator_dev_analysis),
        "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "operator_dev_design_audit_sha256": sha256_file(args.design_audit),
    }
    for field, expected in expected_bound.items():
        if bound.get(field) != expected:
            raise ValueError(f"diagnostic bound artifact mismatch: {field}")
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator_dev design audit failed")
    if not (model.get("verified") or model.get("success")):
        raise ValueError("model manifest is unverified")
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    if not (len(args.selected_node) == 3 and args.selected_node[0] == "a" and args.selected_node[1:].isdigit()):
        raise ValueError("invalid selected node")
    if snapshot.get("selection_contract") != {
        "allow_nodes": [],
        "cpus": 8,
        "exclude_nodes": [],
        "mem_mib": 131072,
        "npu_type": "910B3",
        "npus": 1,
        "partition": "a01",
    }:
        raise ValueError("selector resource contract mismatch")
    if int(snapshot.get("npu_free", -1)) < 1 or int(snapshot.get("cpu_free", -1)) < 8:
        raise ValueError("selector snapshot lacks requested capacity")
    if int(snapshot.get("mem_free_mib", -1)) < 131072:
        raise ValueError("selector snapshot lacks requested memory")
    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-adsge-v4-post-fit-behavior-diagnostic-code-v57",
        "created_utc": args.created_utc,
        "stage": "governance_failed_fit_selection",
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
        "editor_contract_filename": args.editor_contract.name,
        "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
        "diagnostic_contract_filename": args.diagnostic_contract.name,
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "checkpoint": _artifact(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report": _artifact(args.fit_report),
        "fit_report_sha256": sha256_file(args.fit_report),
        "v3_operator_dev_analysis": _artifact(args.v3_operator_dev_analysis),
        "v3_operator_dev_analysis_sha256": sha256_file(args.v3_operator_dev_analysis),
        "model_manifest": _artifact(args.model_manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
