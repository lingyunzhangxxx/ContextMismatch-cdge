#!/usr/bin/env python3
"""Audit AMSGE behavior and apply the frozen selection gates when appropriate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator


def _behavior_gate(report: dict, contract: dict) -> dict:
    gates = contract["behavior_selection"]["admissibility"]
    gap_bootstrap = report["matched_minus_mismatched_reduction_bootstrap"]
    normalized_bootstrap = report["normalized_gap_reduction_bootstrap"]
    directional = report["directional_recovery"]
    swaps = report["gap_reduction_by_label_swap"]
    benchmarks = report["gap_reduction_by_benchmark"]
    matched = report["matched_cell_collateral"]
    checks = {
        "gap_reduction_lower_95": gap_bootstrap is not None
        and float(gap_bootstrap["ci95"][0])
        >= float(gates["gap_reduction_lower_95_minimum"]),
        "normalized_gap_reduction_lower_95": normalized_bootstrap is not None
        and float(normalized_bootstrap["ci95"][0])
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
        "zero_gate_exact_identity": report["audit"]["zero_gate_max_error"] == 0.0,
    }
    return {"checks": checks, "behavior_eligible": all(checks.values())}


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
        raise FileExistsError(f"refusing existing analysis output: {args.output}")
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
    report["method"] = contract["method_short_name"]
    report["editor_contract_sha256"] = sha256_file(args.editor_contract)
    selection_stage = environment["stage"] in {
        "governance_behavior_selection",
        "governance_failed_fit_selection",
    }
    if selection_stage:
        gate = _behavior_gate(report, contract)
        report["selection_gate"] = gate
        if environment.get("post_failure_characterization"):
            report["post_failure_characterization"] = True
            report["fit_gates_passed"] = False
            report["candidate_eligible"] = False
            report["admissibility"] = {
                "behavioral_selection_complete": report["audit"]["success"],
                "behavior_gate_would_pass_in_isolation": gate["behavior_eligible"],
                "behavior_eligible": False,
                "protected_controls_complete": False,
                "candidate_may_be_locked": False,
                "reason": "V1 failed its frozen fit gates; behavior is diagnostic only.",
            }
        else:
            report["admissibility"] = {
                "behavioral_selection_complete": report["audit"]["success"],
                "behavior_eligible": gate["behavior_eligible"],
                "protected_controls_complete": False,
                "candidate_may_be_locked": False,
                "reason": "Protected gated and forced-on controls remain required.",
            }
    else:
        report["selection_gate"] = {
            "evaluated": False,
            "reason": "smoke is execution validation, not operator_dev selection",
        }
        if environment.get("post_failure_characterization"):
            report["post_failure_characterization"] = True
            report["fit_gates_passed"] = False
            report["candidate_eligible"] = False
            report["admissibility"] = {
                "behavioral_selection_complete": False,
                "behavior_eligible": False,
                "protected_controls_complete": False,
                "candidate_may_be_locked": False,
                "reason": "Failed-fit smoke is execution validation only.",
            }
    report["final_test_open_count"] = int(environment.get("final_test_open_count", 0))
    atomic_write_text(
        args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", allow_overwrite=True
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
