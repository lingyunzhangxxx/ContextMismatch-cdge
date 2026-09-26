#!/usr/bin/env python3
"""Audit the one-time C-DGE-V4.1 final-test behavior."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator
from scripts.benchmark_v10.cdge_contract import (
    FINAL_EXPECTED_KEY_SHA256,
    FINAL_EXPECTED_ROWS,
    FINAL_STAGE,
    METHOD,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--pareto-lock", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C-DGE final analysis: {args.output}")
    temporary = args.output.with_name(f".{args.output.name}.base")
    original_argv = sys.argv
    try:
        sys.argv = [
            "analyze_operator_candidate",
            "--input", str(args.input),
            "--environment", str(args.environment),
            "--identity-report", str(args.identity_report),
            "--output", str(temporary),
            "--bootstrap-replicates", str(args.bootstrap_replicates),
        ]
        analyze_operator()
    finally:
        sys.argv = original_argv
    report = json.loads(temporary.read_text())
    temporary.unlink()
    environment = json.loads(args.environment.read_text())
    identity = json.loads(args.identity_report.read_text())
    lock = json.loads(args.pareto_lock.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    required_environment = {
        "stage": FINAL_STAGE,
        "method": METHOD,
        "candidate_id": "C-DGE-V4.1-27:mlp",
        "candidate_eligible": True,
        "candidate_may_be_locked": True,
        "expected_rows": FINAL_EXPECTED_ROWS,
        "planned_rows": FINAL_EXPECTED_ROWS,
        "expected_key_sha256": FINAL_EXPECTED_KEY_SHA256,
        "final_test_open": True,
        "final_test_open_count": 1,
        "production_rollout_approved": False,
    }
    for field, expected in required_environment.items():
        if environment.get(field) != expected:
            raise ValueError(f"C-DGE final environment mismatch: {field}")
    if identity.get("success") is not True or float(identity.get("max_error", 1.0)) != 0.0:
        raise ValueError("C-DGE final zero-gate identity failed")
    if lock.get("locked") is not True or lock.get("candidate_may_be_locked") is not True:
        raise ValueError("C-DGE final lock is not valid")
    if authorization.get("final_test_open") is not True or authorization.get(
        "final_test_open_count"
    ) != 1:
        raise ValueError("C-DGE final authorization boundary mismatch")
    audit = report.get("audit", {})
    required_audit = {
        "success": True,
        "row_count": FINAL_EXPECTED_ROWS,
        "unique_job_keys": FINAL_EXPECTED_ROWS,
        "expected_key_sha256": FINAL_EXPECTED_KEY_SHA256,
        "observed_key_sha256": FINAL_EXPECTED_KEY_SHA256,
        "zero_gate_max_error": 0.0,
    }
    for field, expected in required_audit.items():
        if audit.get(field) != expected:
            raise ValueError(f"C-DGE final audit mismatch: {field}")
    report.update(
        {
            "method": METHOD,
            "runtime_checkpoint_method": "ADSGE-V4",
            "stage": FINAL_STAGE,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "confirmatory": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": True,
            "pareto_lock_sha256": sha256_file(args.pareto_lock),
            "authorization_sha256": sha256_file(args.execution_authorization),
            "final_test_open": True,
            "final_test_open_count": 1,
            "production_rollout_approved": False,
        }
    )
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
