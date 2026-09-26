#!/usr/bin/env python3
"""Analyze one complete DSGE-V3 behavior run without cross-site averaging."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator


def _fit_audit_gate(report: dict, contract: dict, fit_eligible: bool) -> dict:
    gates = contract["direct_behavior_site_selection"]
    gap = report["matched_minus_mismatched_reduction_bootstrap"]
    normalized = report["normalized_gap_reduction_bootstrap"]
    directional = report["directional_recovery"]
    swaps = report["gap_reduction_by_label_swap"]
    benchmarks = report["gap_reduction_by_benchmark"]
    checks = {
        "fit_eligible": fit_eligible,
        "gap_reduction_lower_95": gap is not None
        and float(gap["ci95"][0]) >= float(gates["gap_reduction_lower_95_minimum"]),
        "normalized_gap_reduction_lower_95": normalized is not None
        and float(normalized["ci95"][0])
        >= float(gates["normalized_gap_reduction_lower_95_minimum"]),
        "both_mismatch_directions_positive": all(
            value["equal_weight_benchmark_mean"] is not None
            and float(value["equal_weight_benchmark_mean"]) > 0
            for value in directional.values()
        ),
        "both_label_swaps_positive": all(
            value["equal_weight_benchmark_mean"] is not None
            and float(value["equal_weight_benchmark_mean"]) > 0
            for value in swaps.values()
        ),
        "all_benchmarks_nonnegative": all(
            value["equal_weight_benchmark_mean"] is not None
            and float(value["equal_weight_benchmark_mean"]) >= 0
            for value in benchmarks.values()
        ),
        "matched_selected_logit_exact_identity": float(
            report["matched_selected_logit_max_error"]
        )
        <= float(gates["matched_selected_logit_max_error"]),
        "external_zero_gate_exact_identity": report["audit"]["zero_gate_max_error"] == 0.0,
    }
    return {
        "checks": checks,
        "behavior_would_pass_without_fit_gate": all(
            value for key, value in checks.items() if key != "fit_eligible"
        ),
        "candidate_eligible": all(checks.values()),
    }


def _operator_dev_gate(report: dict, contract: dict) -> dict:
    gates = contract["developmental_operator_dev_gate"]
    gap = report["matched_minus_mismatched_reduction_bootstrap"]
    normalized = report["normalized_gap_reduction_bootstrap"]
    directional = report["directional_recovery"]
    swaps = report["gap_reduction_by_label_swap"]
    benchmarks = report["gap_reduction_by_benchmark"]
    matched = report["matched_cell_collateral"]
    checks = {
        "gap_reduction_lower_95": gap is not None
        and float(gap["ci95"][0]) >= float(gates["gap_reduction_lower_95_minimum"]),
        "normalized_gap_reduction_lower_95": normalized is not None
        and float(normalized["ci95"][0])
        >= float(gates["normalized_gap_reduction_lower_95_minimum"]),
        "both_mismatch_directions_positive": all(
            float(value["equal_weight_benchmark_mean"]) > 0
            for value in directional.values()
        ),
        "both_label_swaps_positive": all(
            float(value["equal_weight_benchmark_mean"]) > 0 for value in swaps.values()
        ),
        "all_benchmarks_nonnegative": all(
            float(value["equal_weight_benchmark_mean"]) >= 0
            for value in benchmarks.values()
        ),
        "matched_margin_loss": all(
            float(value["margin_loss_upper"]["upper_one_sided_95"])
            <= float(gates["matched_cell_margin_loss_upper_95_maximum"])
            for value in matched.values()
        ),
        "matched_binary_kl": all(
            float(value["binary_kl_upper"]["upper_one_sided_95"])
            <= float(gates["matched_cell_binary_kl_upper_95_maximum"])
            for value in matched.values()
        ),
        "matched_selected_logit_exact_identity": report["matched_selected_logit_max_error"]
        == float(gates["matched_selected_logit_max_error"]),
        "external_zero_gate_exact_identity": report["audit"]["zero_gate_max_error"] == 0.0,
    }
    return {"checks": checks, "candidate_eligible": all(checks.values())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing directional analysis: {args.output}")
    original_argv = sys.argv
    try:
        sys.argv = [
            "analyze_operator_candidate",
            "--input",
            str(args.input),
            "--environment",
            str(args.environment),
            "--identity-report",
            str(args.identity_report),
            "--output",
            str(args.output),
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
        ]
        analyze_operator()
    finally:
        sys.argv = original_argv
    report = json.loads(args.output.read_text())
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.editor_contract.read_text())
    rows = load_jsonl(args.input)
    matched_errors = [
        float(row["selected_logit_max_error"])
        for row in rows
        if row.get("matched_context") is True
    ]
    if not matched_errors:
        raise ValueError("directional behavior output has no matched rows")
    report.update(
        {
            "method": "DSGE-V3",
            "evaluation_stage": environment["stage"],
            "checkpoint_sha256": environment["checkpoint_sha256"],
            "fit_report_sha256": environment["fit_report_sha256"],
            "editor_contract_sha256": sha256_file(args.editor_contract),
            "fit_eligible": bool(environment["fit_eligible"]),
            "matched_selected_logit_max_error": max(matched_errors),
            "teacher_reconstruction_is_not_used_as_behavior_evidence": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    if environment["stage"] == "governance_directional_fit_audit":
        gate = _fit_audit_gate(report, contract, bool(environment["fit_eligible"]))
        report["selection_gate"] = gate
        report["admissibility"] = {
            "fit_audit_complete": report["audit"]["success"],
            "fit_eligible": bool(environment["fit_eligible"]),
            "behavior_would_pass_without_fit_gate": gate[
                "behavior_would_pass_without_fit_gate"
            ],
            "candidate_eligible": gate["candidate_eligible"],
            "operator_dev_complete": False,
            "protected_controls_complete": False,
            "candidate_may_be_locked": False,
        }
    elif environment["stage"] == "governance_directional_operator_dev":
        gate = _operator_dev_gate(report, contract)
        report["selection_gate"] = gate
        report["admissibility"] = {
            "fit_audit_complete": True,
            "operator_dev_complete": report["audit"]["success"],
            "candidate_eligible": gate["candidate_eligible"],
            "protected_controls_complete": False,
            "candidate_may_be_locked": False,
        }
    else:
        raise ValueError("unexpected directional behavior stage")
    atomic_write_text(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        allow_overwrite=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
