#!/usr/bin/env python3
"""Audit fresh formal C-DGE-V4.1 protected controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v8.analyze_v4_protected_controls_diagnostic import analyze


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--candidate-behavior-analysis", type=Path, required=True)
    parser.add_argument("--v3-controls-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing C-DGE controls analysis: {args.output}")
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.evaluation_contract.read_text())
    behavior = json.loads(args.candidate_behavior_analysis.read_text())
    v3 = json.loads(args.v3_controls_analysis.read_text())
    if environment.get("method") != "C-DGE-V4.1" or environment.get("composite_eligible") is not True:
        raise ValueError("formal C-DGE controls environment mismatch")
    if behavior.get("candidate_eligible") is not True or behavior.get("audit", {}).get("success") is not True:
        raise ValueError("fresh C-DGE behavior prerequisite failed")
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 comparison is not audited")
    report = analyze(rows, environment, contract, v3, replicates=args.bootstrap_replicates)
    report.update(
        {
            "method": "C-DGE-V4.1",
            "runtime_checkpoint_method": "ADSGE-V4",
            "evaluation_stage": "governance_composite_protected_controls",
            "evidence_class": "developmental_post_eligibility_evaluation",
            "confirmatory": False,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "candidate_behavior_analysis_sha256": sha256_file(args.candidate_behavior_analysis),
            "input_sha256": sha256_file(args.input),
            "environment_sha256": sha256_file(args.environment),
            "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
            "original_v4_fit_eligible": False,
            "post_failure_characterization": False,
            "composite_eligible": True,
            "fit_gates_passed": False,
            "controls_evaluation_complete": report["audit"]["success"],
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
