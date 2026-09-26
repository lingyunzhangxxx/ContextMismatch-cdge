#!/usr/bin/env python3
"""Audit ADSGE-V4 operator_dev behavior without overriding its failed fit gate."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator


def _active_summary(rows: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        matched = (
            row["history_condition"] == "obedience"
            and float(row["target_obedience"]) == 1.0
        ) or (
            row["history_condition"] == "verification"
            and float(row["target_obedience"]) == 0.0
        )
        groups["matched" if matched else "mismatched"].append(row)
    result = {}
    for name, group in sorted(groups.items()):
        diagnostics = [next(iter(row["controller_diagnostics"].values())) for row in group]
        active = [float(value.get("application_gate_active", 0.0)) for value in diagnostics]
        routed = [float(value.get("v4_gate_active", 0.0)) for value in diagnostics]
        logits = [float(value["application_logit"]) for value in diagnostics]
        result[name] = {
            "rows": len(group),
            "application_active_fraction": sum(active) / len(active),
            "v4_route_active_fraction": sum(routed) / len(routed),
            "application_logit_minimum": min(logits),
            "application_logit_maximum": max(logits),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--v3-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing V4 diagnostic analysis: {args.output}")
    temporary = args.output.with_name(f".{args.output.name}.base")
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
            str(temporary),
            "--bootstrap-replicates",
            str(args.bootstrap_replicates),
        ]
        analyze_operator()
    finally:
        sys.argv = original_argv
    report = json.loads(temporary.read_text())
    temporary.unlink()
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.editor_contract.read_text())
    diagnostic = json.loads(args.diagnostic_contract.read_text())
    fit = json.loads(args.fit_report.read_text())
    v3 = json.loads(args.v3_analysis.read_text())
    if contract.get("method_short_name") != "ADSGE-V4":
        raise ValueError("unexpected V4 editor contract")
    if fit.get("method") != "ADSGE-V4" or fit.get("fit_eligible") is not False:
        raise ValueError("V4 diagnostic requires the frozen failed fit report")
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 comparison analysis is not terminal audited evidence")
    if diagnostic.get("bound_artifacts", {}).get("v3_operator_dev_analysis_sha256") != sha256_file(args.v3_analysis):
        raise ValueError("V3 comparison analysis SHA is not frozen by the diagnostic contract")
    if v3.get("audit", {}).get("observed_key_sha256") != report.get("audit", {}).get("observed_key_sha256"):
        raise ValueError("V3/V4 same-identity key mismatch")

    def metric(value: dict, key: str) -> float:
        return float(value[key]["equal_weight_benchmark_mean"])

    comparison = {}
    for key in (
        "match_advantage_reduction",
        "normalized_gap_reduction",
        "matched_minus_mismatched_reduction",
    ):
        v4_value = metric(report, key)
        v3_value = metric(v3, key)
        comparison[key] = {
            "v3": v3_value,
            "v4": v4_value,
            "v4_minus_v3": v4_value - v3_value,
        }
    comparison["gap_reduction_by_benchmark"] = {
        name: {
            "v3": float(v3["gap_reduction_by_benchmark"][name]["mean"]),
            "v4": float(report["gap_reduction_by_benchmark"][name]["mean"]),
            "v4_minus_v3": float(report["gap_reduction_by_benchmark"][name]["mean"])
            - float(v3["gap_reduction_by_benchmark"][name]["mean"]),
        }
        for name in sorted(v3["gap_reduction_by_benchmark"])
    }
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
            raise ValueError(f"V4 diagnostic contract mismatch: {field}")
    report.update(
        {
            "method": "ADSGE-V4",
            "stage": "governance_failed_fit_selection",
            "evidence_class": "post_fit_failure_behavior_diagnostic",
            "post_failure_characterization": True,
            "fit_gates_passed": False,
            "fit_eligible": False,
            "candidate_eligible": False,
            "candidate_may_be_locked": False,
            "editor_contract_sha256": sha256_file(args.editor_contract),
            "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
            "fit_report_sha256": sha256_file(args.fit_report),
            "v3_operator_dev_analysis_sha256": sha256_file(args.v3_analysis),
            "comparison_to_v3_same_identity": comparison,
            "application_routing": _active_summary(load_jsonl(args.input)),
            "admissibility": {
                "behavioral_diagnostic_complete": report["audit"]["success"],
                "candidate_eligible": False,
                "candidate_may_be_locked": False,
                "reason": "ADSGE-V4 failed its frozen exact-abstention fit gate; this operator_dev run is diagnostic only.",
            },
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    if environment.get("fit_gates_passed") is not False:
        raise ValueError("V4 diagnostic environment changed the failed fit gate")
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
