#!/usr/bin/env python3
"""Create one stage-specific SHA-bound Qwen3.5 C-DGE code-v88 authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v13.qwen35_eval_contract import (
    CONTROL_KEY,
    CONTROL_ROWS,
    FINAL_KEY,
    FINAL_ROWS,
    METHOD,
    OPERATOR_KEY,
    OPERATOR_ROWS,
    PERFORMANCE_CELLS,
    PERFORMANCE_ROWS,
    WORK_ROOT,
    require_fit,
    require_pareto_lock,
)


def artifact(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": sha256_file(path)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("behavior", "controls", "performance", "final"), required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--selector-snapshot", type=Path, required=True)
    p.add_argument("--selected-node", required=True)
    p.add_argument("--created-utc", required=True)
    common_names = (
        "evaluation_contract", "replication_protocol", "crossover_contract", "manifest",
        "manifest_report", "controls",
        "model_contract", "fit_report", "fit_receipt", "directional_checkpoint",
        "applicability_checkpoint", "directional_contract", "applicability_contract",
        "composite_contract", "composite_report", "archive_helper", "archive_coordinator",
        "pull_helper", "output",
    )
    for name in common_names:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--operator-dev-design-audit", type=Path)
    p.add_argument("--final-design-audit", type=Path)
    p.add_argument("--behavior-analysis", type=Path)
    p.add_argument("--behavior-receipt", type=Path)
    p.add_argument("--controls-analysis", type=Path)
    p.add_argument("--controls-receipt", type=Path)
    p.add_argument("--performance-analysis", type=Path)
    p.add_argument("--performance-receipt", type=Path)
    p.add_argument("--pareto-lock", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError("authorization output must be a fresh path")
    if args.code_root != WORK_ROOT / "code-v88":
        raise ValueError("authorization requires exact code-v88")
    require_fit(
        fit_report=args.fit_report, fit_receipt=args.fit_receipt,
        directional_checkpoint=args.directional_checkpoint,
        applicability_checkpoint=args.applicability_checkpoint,
        composite_report=args.composite_report,
        evaluation_contract=args.evaluation_contract,
    )
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("node-selection snapshot mismatch")
    selection = snapshot.get("selection_contract", {})
    if selection != {
        "allow_nodes": [], "cpus": 8, "exclude_nodes": [], "mem_mib": 131072,
        "npu_type": "910B3", "npus": 1, "partition": "a01",
    }:
        raise ValueError("node-selection resource contract mismatch")
    if int(snapshot.get("npu_free", -1)) < 1:
        raise ValueError("node-selection snapshot lacks capacity")
    if int(snapshot.get("cpu_free", -1)) < 8:
        raise ValueError("node-selection snapshot lacks CPU capacity")
    if int(snapshot.get("mem_free_mib", -1)) < 131072:
        raise ValueError("node-selection snapshot lacks memory capacity")
    common_paths = {
        name: getattr(args, name)
        for name in (
            "evaluation_contract", "replication_protocol", "crossover_contract", "manifest",
            "manifest_report", "controls", "model_contract", "fit_report", "fit_receipt",
            "directional_checkpoint", "applicability_checkpoint", "directional_contract",
            "applicability_contract", "composite_contract", "composite_report",
        )
    }
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-5-9b-cdge-v4-1-{args.stage}-code-v88-{args.created_utc}",
        "created_utc": args.created_utc,
        "stage": f"qwen35_cdge_{args.stage}",
        "method": METHOD,
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "execution_node": args.selected_node,
        "resource_contract": {
            "partition": "a01", "nodes": 1, "ntasks": 1, "cpus_per_task": 8,
            "mem_mib": 131072, "npu_type": "910B3", "npus": 1,
            "time_limit": "12:00:00", "node": args.selected_node,
        },
        "node_selection_snapshot": artifact(args.selector_snapshot),
        "bound_artifacts": {name: artifact(path) for name, path in common_paths.items()},
        "monitoring_contract": {
            "poll_seconds": 1,
            "continuous_watch_required": True,
            "archive_helper": artifact(args.archive_helper),
            "archive_coordinator": artifact(args.archive_coordinator),
            "pull_helper": artifact(args.pull_helper),
        },
        "final_test_open": args.stage == "final",
        "final_test_open_count": 1 if args.stage == "final" else 0,
        "production_rollout_approved": False,
    }
    if args.stage == "behavior":
        if args.operator_dev_design_audit is None:
            raise ValueError("behavior requires operator_dev_design_audit")
        value["bound_artifacts"]["operator_dev_design_audit"] = artifact(
            args.operator_dev_design_audit
        )
        value.update(expected_rows=OPERATOR_ROWS, expected_key_sha256=OPERATOR_KEY)
    elif args.stage == "controls":
        for name in ("behavior_analysis", "behavior_receipt"):
            path = getattr(args, name)
            if path is None: raise ValueError(f"controls requires {name}")
            value["bound_artifacts"][name] = artifact(path)
        value.update(expected_rows=CONTROL_ROWS, expected_key_sha256=CONTROL_KEY)
    elif args.stage == "performance":
        for name in ("behavior_analysis", "behavior_receipt", "controls_analysis", "controls_receipt"):
            path = getattr(args, name)
            if path is None: raise ValueError(f"performance requires {name}")
            value["bound_artifacts"][name] = artifact(path)
        if args.operator_dev_design_audit is None:
            raise ValueError("performance requires operator_dev_design_audit")
        value["bound_artifacts"]["operator_dev_design_audit"] = artifact(
            args.operator_dev_design_audit
        )
        value.update(expected_measurements=PERFORMANCE_ROWS, expected_cells=PERFORMANCE_CELLS)
    else:
        for name in (
            "behavior_analysis", "behavior_receipt", "controls_analysis", "controls_receipt",
            "performance_analysis", "performance_receipt", "pareto_lock",
        ):
            path = getattr(args, name)
            if path is None: raise ValueError(f"final requires {name}")
            value["bound_artifacts"][name] = artifact(path)
        if args.final_design_audit is None:
            raise ValueError("final requires final_design_audit")
        value["bound_artifacts"]["final_design_audit"] = artifact(args.final_design_audit)
        require_pareto_lock(
            args.pareto_lock,
            expected_sources={
                "behavior_analysis": args.behavior_analysis,
                "behavior_receipt": args.behavior_receipt,
                "controls_analysis": args.controls_analysis,
                "controls_receipt": args.controls_receipt,
                "performance_analysis": args.performance_analysis,
                "performance_receipt": args.performance_receipt,
                "fit_report": args.fit_report,
                "fit_receipt": args.fit_receipt,
                "directional_checkpoint": args.directional_checkpoint,
                "applicability_checkpoint": args.applicability_checkpoint,
                "composite_contract": args.composite_contract,
                "evaluation_contract": args.evaluation_contract,
                "applicability_contract": args.applicability_contract,
            },
        )
        value.update(expected_rows=FINAL_ROWS, expected_key_sha256=FINAL_KEY,
                     candidate_may_be_locked=True)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
