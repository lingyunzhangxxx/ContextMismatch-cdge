#!/usr/bin/env python3
"""Audit one external-baseline shard and emit paired method statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator
from scripts.benchmark_v19.run_external_baselines import GROUPS


def _baseline_digest(rows: list[dict]) -> str:
    payload = [canonical_json({"job_key": row["job_key"], "baseline": row["baseline"]}) for row in sorted(rows, key=lambda row: row["job_key"])]
    return hashlib.sha256((("\n".join(payload)) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing existing external baseline shard analysis")
    reports = {}
    baseline_digests = {}
    for method in GROUPS[args.group]:
        rows_path = args.input_dir / f"{method}.jsonl"
        environment = args.input_dir / f"{method}.environment.json"
        identity = args.input_dir / f"{method}.identity.json"
        analysis = args.input_dir / f"{method}.analysis.json"
        old = sys.argv
        try:
            sys.argv = [
                "analyze_operator_candidate", "--input", str(rows_path),
                "--environment", str(environment), "--identity-report", str(identity),
                "--output", str(analysis), "--bootstrap-replicates", str(args.bootstrap_replicates),
            ]
            analyze_operator()
        finally:
            sys.argv = old
        report = json.loads(analysis.read_text())
        if report.get("audit", {}).get("success") is not True:
            raise RuntimeError(f"external baseline scientific audit failed: {method}")
        rows = load_jsonl(rows_path)
        digest = _baseline_digest(rows)
        baseline_digests[method] = digest
        report.update({
            "method": method,
            "method_faithful_adapter": True,
            "unmodified_official_implementation": False,
            "baseline_logits_sha256": digest,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        })
        atomic_write_text(analysis, json.dumps(report, indent=2, sort_keys=True) + "\n", allow_overwrite=True)
        reports[method] = {
            "analysis_path": str(analysis),
            "analysis_sha256": sha256_file(analysis),
            "normalized_gap_reduction": report["normalized_gap_reduction_bootstrap"],
            "matched_minus_mismatched_reduction": report["matched_minus_mismatched_reduction_bootstrap"],
            "directional_recovery": report["directional_recovery"],
            "matched_cell_collateral": report["matched_cell_collateral"],
        }
    if len(set(baseline_digests.values())) != 1:
        raise RuntimeError("baseline logits differ within external baseline shard")
    summary = {
        "schema_version": 1,
        "stage": "qwen3_8b_external_baseline_evaluation",
        "group": args.group,
        "methods": reports,
        "rows_per_method": 3072,
        "baseline_logits_sha256": next(iter(baseline_digests.values())),
        "baseline_logits_bit_identical_within_shard": True,
        "scientific_audit_complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
