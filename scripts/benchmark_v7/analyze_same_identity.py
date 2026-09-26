#!/usr/bin/env python3
"""Audit and analyze one complete same-identity supplementary shard."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_candidate import main as analyze_operator
from scripts.benchmark_v7.run_same_identity import EXPECTED_KEY_SHA256, EXPECTED_ROWS, METHOD_SHARDS


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _analyze_method(root: Path, method: str, replicates: int) -> dict:
    output = root / "analysis" / f"{method}.json"
    original = sys.argv
    try:
        sys.argv = [
            "analyze_operator_candidate",
            "--input", str(root / "rows" / f"{method}.jsonl"),
            "--environment", str(root / "environment" / f"{method}.json"),
            "--identity-report", str(root / "identity" / f"{method}.json"),
            "--output", str(output),
            "--bootstrap-replicates", str(replicates),
        ]
        analyze_operator()
    finally:
        sys.argv = original
    report = json.loads(output.read_text())
    report.update(
        {
            "method": method,
            "same_identity_supplement": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    atomic_write_text(
        output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        allow_overwrite=True,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing shard audit: {args.output}")
    authorization = json.loads(args.execution_authorization.read_text())
    shard = authorization.get("method_shard")
    methods = tuple(authorization.get("methods", []))
    if shard not in METHOD_SHARDS or methods != METHOD_SHARDS[shard]:
        raise ValueError("authorization method shard changed")
    reports = {
        method: _analyze_method(args.output_dir, method, args.bootstrap_replicates)
        for method in methods
    }
    rows = {
        method: load_jsonl(args.output_dir / "rows" / f"{method}.jsonl")
        for method in methods
    }
    baseline_by_method = {
        method: {row["job_key"]: row["baseline"] for row in values}
        for method, values in rows.items()
    }
    first, second = methods
    common_keys = set(baseline_by_method[first]) & set(baseline_by_method[second])
    baseline_mismatch_keys = sorted(
        key
        for key in common_keys
        if baseline_by_method[first][key] != baseline_by_method[second][key]
    )
    systems_path = args.output_dir / "systems_metrics.json"
    systems = json.loads(systems_path.read_text())
    method_audits = {method: reports[method]["audit"] for method in methods}
    success = (
        len(common_keys) == EXPECTED_ROWS
        and not baseline_mismatch_keys
        and all(
            audit.get("success") is True
            and audit.get("row_count") == EXPECTED_ROWS
            and audit.get("observed_key_sha256") == EXPECTED_KEY_SHA256
            and audit.get("zero_gate_max_error") == 0.0
            for audit in method_audits.values()
        )
        and _finite(systems)
    )
    value = {
        "schema_version": 1,
        "stage": authorization["stage"],
        "method_shard": shard,
        "methods": list(methods),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "method_audits": method_audits,
        "paired_baseline_common_keys": len(common_keys),
        "paired_baseline_mismatch_keys": baseline_mismatch_keys[:20],
        "paired_baseline_logits_identical": not baseline_mismatch_keys,
        "systems_metrics_sha256": sha256_file(systems_path),
        "success": success,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))
    if not success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
