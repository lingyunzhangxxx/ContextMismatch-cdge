#!/usr/bin/env python3
"""Audit Qwen3.5 C-DGE operator-dev or one-time final behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator
from scripts.benchmark_v8.analyze_v4_behavior_diagnostic import _active_summary
from scripts.benchmark_v13.qwen35_eval_contract import FINAL_KEY, FINAL_ROWS, METHOD, OPERATOR_KEY, OPERATOR_ROWS


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-split", choices=("operator_dev", "final_test"), required=True)
    for name in ("input", "environment", "identity_report", "evaluation_contract", "execution_authorization", "output"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = p.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    temporary = args.output.with_name("." + args.output.name + ".base")
    original = sys.argv
    try:
        sys.argv = ["analyze_operator_candidate", "--input", str(args.input),
                    "--environment", str(args.environment), "--identity-report", str(args.identity_report),
                    "--output", str(temporary), "--bootstrap-replicates", str(args.bootstrap_replicates)]
        analyze_operator()
    finally: sys.argv = original
    report = json.loads(temporary.read_text()); temporary.unlink()
    environment = json.loads(args.environment.read_text())
    identity = json.loads(args.identity_report.read_text())
    contract = json.loads(args.evaluation_contract.read_text())
    final = args.evaluation_split == "final_test"
    rows, key = (FINAL_ROWS, FINAL_KEY) if final else (OPERATOR_ROWS, OPERATOR_KEY)
    audit = report["audit"]
    if environment.get("stage") != ("qwen35_cdge_final_test" if final else "qwen35_cdge_operator_dev"):
        raise ValueError("Qwen3.5 behavior environment stage mismatch")
    if environment.get("method") != METHOD or identity.get("success") is not True or identity.get("max_error") != 0.0:
        raise ValueError("Qwen3.5 behavior identity mismatch")
    required_audit = {"success": True, "row_count": rows, "unique_job_keys": rows,
                      "expected_key_sha256": key, "observed_key_sha256": key,
                      "zero_gate_max_error": 0.0}
    for field, expected in required_audit.items():
        if audit.get(field) != expected: raise ValueError(f"behavior audit mismatch: {field}")
    gates = contract["developmental_gates"]
    checks = {
        "gap_lower_95": float(report["matched_minus_mismatched_reduction_bootstrap"]["ci95"][0]) >= float(gates["gap_reduction_lower_95_minimum"]),
        "normalized_gap_lower_95": float(report["normalized_gap_reduction_bootstrap"]["ci95"][0]) >= float(gates["normalized_gap_reduction_lower_95_minimum"]),
        "both_directions_positive": all(float(v["equal_weight_benchmark_mean"]) > 0 for v in report["directional_recovery"].values()),
        "both_label_swaps_positive": all(float(v["equal_weight_benchmark_mean"]) > 0 for v in report["gap_reduction_by_label_swap"].values()),
        "all_benchmarks_nonnegative": len(report["gap_reduction_by_benchmark"]) == 6 and all(float(v["equal_weight_benchmark_mean"]) >= 0 for v in report["gap_reduction_by_benchmark"].values()),
        "zero_gate_exact": audit["zero_gate_max_error"] == 0.0,
    }
    eligible = all(checks.values())
    report.update({
        "method": METHOD, "runtime_checkpoint_method": "ADSGE-V4",
        "stage": "qwen35_cdge_final_test" if final else "qwen35_cdge_operator_dev",
        "model": "Qwen3.5-9B", "evidence_class": "confirmatory" if final else "developmental",
        "confirmatory": final, "behavior_gate_checks": checks,
        "candidate_eligible": True if final else eligible,
        "candidate_may_be_locked": final,
        "behavior_evaluation_complete": audit["success"],
        "application_routing": _active_summary(load_jsonl(args.input)),
        "input_sha256": sha256_file(args.input), "environment_sha256": sha256_file(args.environment),
        "identity_report_sha256": sha256_file(args.identity_report),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "final_test_open": final, "final_test_open_count": 1 if final else 0,
        "production_rollout_approved": False,
    })
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__": main()
