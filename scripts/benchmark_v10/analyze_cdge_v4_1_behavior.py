#!/usr/bin/env python3
"""Audit fresh C-DGE-V4.1 operator-dev behavior and compare same identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator
from scripts.benchmark_v8.analyze_v4_behavior_diagnostic import _active_summary


METHOD = "C-DGE-V4.1"
STAGE = "governance_composite_behavior_selection"
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"


def _metric(value: dict, key: str) -> float:
    return float(value[key]["equal_weight_benchmark_mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--composite-report", type=Path, required=True)
    parser.add_argument("--composite-authorization", type=Path, required=True)
    parser.add_argument("--composite-receipt", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--v3-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C-DGE analysis: {args.output}")
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
    fit = json.loads(args.fit_report.read_text())
    v3 = json.loads(args.v3_analysis.read_text())
    composite = json.loads(args.composite_report.read_text())
    receipt = json.loads(args.composite_receipt.read_text())
    if environment.get("stage") != STAGE or environment.get("method") != METHOD:
        raise ValueError("C-DGE environment identity mismatch")
    if environment.get("original_v4_fit_eligible") is not False:
        raise ValueError("historical V4 fit failure was not preserved")
    if environment.get("composite_eligible") is not True or environment.get("candidate_eligible") is not True:
        raise ValueError("terminal composite eligibility was not propagated")
    if fit.get("method") != "ADSGE-V4" or fit.get("fit_eligible") is not False:
        raise ValueError("unexpected historical fit report")
    if composite.get("candidate_eligible") is not True or composite.get("audit_complete") is not True:
        raise ValueError("composite report is not terminal eligible evidence")
    if receipt.get("archive_sha256") != "602ad58c15ae47b3b24cf4de18bc1ba1195edf694cce5656e26bb7a4aba4ccdc":
        raise ValueError("composite archive receipt mismatch")
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 comparison is not audited")
    audit = report.get("audit", {})
    if audit.get("observed_key_sha256") != EXPECTED_KEY_SHA256:
        raise ValueError("C-DGE operator-dev key mismatch")
    if v3.get("audit", {}).get("observed_key_sha256") != EXPECTED_KEY_SHA256:
        raise ValueError("V3/C-DGE same-identity key mismatch")
    if identity.get("success") is not True or identity.get("max_error") != 0.0:
        raise ValueError("C-DGE zero-gate identity failed")

    comparison = {}
    for key in (
        "match_advantage_reduction",
        "normalized_gap_reduction",
        "matched_minus_mismatched_reduction",
    ):
        current = _metric(report, key)
        prior = _metric(v3, key)
        comparison[key] = {"v3": prior, "cdge_v4_1": current, "cdge_v4_1_minus_v3": current - prior}
    comparison["gap_reduction_by_benchmark"] = {
        name: {
            "v3": float(v3["gap_reduction_by_benchmark"][name]["mean"]),
            "cdge_v4_1": float(report["gap_reduction_by_benchmark"][name]["mean"]),
            "cdge_v4_1_minus_v3": float(report["gap_reduction_by_benchmark"][name]["mean"])
            - float(v3["gap_reduction_by_benchmark"][name]["mean"]),
        }
        for name in sorted(v3["gap_reduction_by_benchmark"])
    }
    report.update(
        {
            "method": METHOD,
            "runtime_checkpoint_method": "ADSGE-V4",
            "stage": STAGE,
            "evidence_class": "developmental_post_eligibility_evaluation",
            "confirmatory": False,
            "original_v4_fit_eligible": False,
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "behavior_evaluation_complete": audit.get("success") is True,
            "editor_contract_sha256": sha256_file(args.editor_contract),
            "composite_contract_sha256": sha256_file(args.composite_contract),
            "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
            "composite_report_sha256": sha256_file(args.composite_report),
            "composite_authorization_sha256": sha256_file(args.composite_authorization),
            "composite_receipt_sha256": sha256_file(args.composite_receipt),
            "fit_report_sha256": sha256_file(args.fit_report),
            "v3_operator_dev_analysis_sha256": sha256_file(args.v3_analysis),
            "comparison_to_v3_same_identity": comparison,
            "application_routing": _active_summary(load_jsonl(args.input)),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report.get("audit", {}).get("success") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
