#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .common import DEFAULT_CONTRACT, PARTITIONS, load_jsonl, sha256_file


def audit(contract: dict, rows: list[dict]) -> dict:
    required = {
        "benchmark",
        "native_id",
        "item_id",
        "question",
        "correct_answer",
        "foil_answer",
        "foil_rule",
        "partition",
        "history_realization",
        "dataset_revision",
    }
    missing = [(index, sorted(required - set(row))) for index, row in enumerate(rows) if required - set(row)]
    item_counts = Counter(row.get("item_id") for row in rows)
    duplicates = sorted(item_id for item_id, count in item_counts.items() if count != 1)
    equal_candidates = [row.get("item_id") for row in rows if row.get("correct_answer") == row.get("foil_answer")]
    expected_by_benchmark = {spec["id"]: spec["sample_count"] for spec in contract["benchmarks"]}
    observed_by_benchmark = Counter(row.get("benchmark") for row in rows)
    partition_counts = Counter((row.get("benchmark"), row.get("partition")) for row in rows)
    partition_errors = []
    for benchmark, total in expected_by_benchmark.items():
        expected = total // len(PARTITIONS)
        for partition in PARTITIONS:
            observed = partition_counts[(benchmark, partition)]
            if observed != expected:
                partition_errors.append(
                    {"benchmark": benchmark, "partition": partition, "expected": expected, "observed": observed}
                )
    revision_errors = []
    revisions = {spec["id"]: spec["revision"] for spec in contract["benchmarks"]}
    for row in rows:
        if row.get("dataset_revision") != revisions.get(row.get("benchmark")):
            revision_errors.append(row.get("item_id"))
    success = not any(
        [
            missing,
            duplicates,
            equal_candidates,
            partition_errors,
            revision_errors,
            dict(observed_by_benchmark) != expected_by_benchmark,
        ]
    )
    return {
        "success": success,
        "row_count": len(rows),
        "expected_row_count": sum(expected_by_benchmark.values()),
        "observed_by_benchmark": dict(sorted(observed_by_benchmark.items())),
        "expected_by_benchmark": expected_by_benchmark,
        "missing_field_rows": missing[:20],
        "duplicate_item_ids": duplicates[:20],
        "equal_candidate_item_ids": equal_candidates[:20],
        "partition_errors": partition_errors,
        "revision_error_item_ids": revision_errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    report = audit(contract, load_jsonl(args.manifest))
    report["contract_sha256"] = sha256_file(args.contract)
    report["manifest_sha256"] = sha256_file(args.manifest)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
