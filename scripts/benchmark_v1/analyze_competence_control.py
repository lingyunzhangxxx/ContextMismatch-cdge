#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .common import atomic_write_text, load_jsonl, sha256_file


def _summary(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "binary_accuracy": statistics.fmean(float(row["correct"]) for row in rows),
        "mean_correct_margin": statistics.fmean(float(row["correct_logit_margin"]) for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    keys = Counter(row["job_key"] for row in rows)
    observed_key_sha256 = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(keys)).encode("utf-8")
    ).hexdigest()
    suffixes = defaultdict(set)
    for row in rows:
        suffixes[(row["item_id"], row["label_swap"])].add(
            row["suffix_token_sha256_int32_le"]
        )
    nonfinite = [
        row["job_key"]
        for row in rows
        if not all(math.isfinite(float(row[key])) for key in ("logit_a", "logit_b", "correct_logit_margin"))
    ]
    audit = {
        "row_count": len(rows),
        "planned_rows": environment["planned_rows"],
        "duplicate_keys": [key for key, count in keys.items() if count != 1],
        "observed_job_key_set_sha256": observed_key_sha256,
        "expected_job_key_set_sha256": environment["expected_job_key_set_sha256"],
        "nonfinite_keys": nonfinite,
        "suffix_mismatch_keys": [key for key, values in suffixes.items() if len(values) != 1],
    }
    audit["success"] = (
        audit["row_count"] == audit["planned_rows"]
        and not audit["duplicate_keys"]
        and observed_key_sha256 == environment["expected_job_key_set_sha256"]
        and not nonfinite
        and not audit["suffix_mismatch_keys"]
    )
    by_benchmark = defaultdict(list)
    by_role = defaultdict(list)
    by_label = defaultdict(list)
    by_partition = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
        by_role[row["declared_role"]].append(row)
        by_label[str(row["label_swap"])].append(row)
        by_partition[row["partition"]].append(row)
    report = {
        "schema_version": 1,
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "audit": audit,
        "overall": _summary(rows),
        "benchmark": {key: _summary(value) for key, value in sorted(by_benchmark.items())},
        "declared_role": {key: _summary(value) for key, value in sorted(by_role.items())},
        "label_swap": {key: _summary(value) for key, value in sorted(by_label.items())},
        "partition": {key: _summary(value) for key, value in sorted(by_partition.items())},
        "interpretation_boundary": {
            "pressure_free_pairwise_not_official_leaderboard_accuracy": True,
            "context_mismatch_estimand_unchanged": True,
        },
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not audit["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
