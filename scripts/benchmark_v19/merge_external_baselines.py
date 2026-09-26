#!/usr/bin/env python3
"""Merge three terminal external-baseline shards and verify paired identity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing existing merged external baseline report")
    shards = [json.loads(path.read_text()) for path in args.input]
    if {row.get("group") for row in shards} != {"input", "activation", "representation"}:
        raise ValueError("external baseline shard set is incomplete")
    for row in shards:
        if row.get("scientific_audit_complete") is not True or row.get("rows_per_method") != 3072:
            raise ValueError("external baseline shard is not terminal-audited")
        if row.get("final_test_open") is not False or row.get("production_rollout_approved") is not False:
            raise ValueError("external baseline safety boundary changed")
    baseline = {row["baseline_logits_sha256"] for row in shards}
    if len(baseline) != 1:
        raise RuntimeError("baseline logits are not bit-identical across all six methods")
    methods = {}
    for row in shards:
        overlap = set(methods) & set(row["methods"])
        if overlap:
            raise ValueError(f"duplicate external methods: {sorted(overlap)}")
        methods.update(row["methods"])
    if set(methods) != {"governance_reset_prompt", "session_isolation", "caa", "cast", "loreft", "reps"}:
        raise ValueError("external method set is incomplete")
    result = {
        "schema_version": 1,
        "stage": "qwen3_8b_external_baseline_comparison",
        "methods": methods,
        "method_count": len(methods),
        "rows_per_method": 3072,
        "total_method_rows": 3072 * len(methods),
        "baseline_logits_sha256": next(iter(baseline)),
        "baseline_logits_bit_identical_across_all_methods": True,
        "input_shards": [{"path": str(path), "sha256": sha256_file(path)} for path in args.input],
        "scientific_audit_complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
