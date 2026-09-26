#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

from .common import atomic_write_text, load_jsonl, sha256_file


def pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return math.nan
    left_mean, right_mean = statistics.fmean(left), statistics.fmean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - left_mean) ** 2 for a in left) * sum((b - right_mean) ** 2 for b in right)
    )
    return numerator / denominator if denominator else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scalar", type=Path, required=True)
    parser.add_argument("--batched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-logit-error", type=float, default=0.125)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    scalar_rows = {row["job_key"]: row for row in load_jsonl(args.scalar)}
    batched_rows = {row["job_key"]: row for row in load_jsonl(args.batched)}
    keys_match = set(scalar_rows) == set(batched_rows)
    common = sorted(set(scalar_rows) & set(batched_rows))
    logit_errors = []
    margin_errors = []
    scalar_margins = []
    batched_margins = []
    decision_matches = []
    suffix_matches = []
    for key in common:
        scalar, batched = scalar_rows[key], batched_rows[key]
        logit_errors.extend(
            [abs(float(scalar["logit_a"]) - float(batched["logit_a"])), abs(float(scalar["logit_b"]) - float(batched["logit_b"]))]
        )
        scalar_margin = float(scalar["correct_logit_margin"])
        batched_margin = float(batched["correct_logit_margin"])
        margin_errors.append(abs(scalar_margin - batched_margin))
        scalar_margins.append(scalar_margin)
        batched_margins.append(batched_margin)
        decision_matches.append(scalar["answer"] == batched["answer"])
        suffix_matches.append(scalar["suffix_token_sha256_int32_le"] == batched["suffix_token_sha256_int32_le"])
    max_logit_error = max(logit_errors, default=math.inf)
    report = {
        "success": bool(
            keys_match
            and common
            and all(decision_matches)
            and all(suffix_matches)
            and max_logit_error <= args.max_logit_error
        ),
        "scalar_sha256": sha256_file(args.scalar),
        "batched_sha256": sha256_file(args.batched),
        "scalar_rows": len(scalar_rows),
        "batched_rows": len(batched_rows),
        "keys_match": keys_match,
        "common_rows": len(common),
        "max_logit_error": max_logit_error,
        "mean_abs_logit_error": statistics.fmean(logit_errors) if logit_errors else math.nan,
        "max_margin_error": max(margin_errors, default=math.inf),
        "mean_abs_margin_error": statistics.fmean(margin_errors) if margin_errors else math.nan,
        "margin_pearson": pearson(scalar_margins, batched_margins),
        "decision_match_fraction": statistics.fmean(decision_matches) if decision_matches else 0.0,
        "suffix_match_fraction": statistics.fmean(suffix_matches) if suffix_matches else 0.0,
        "max_logit_error_threshold": args.max_logit_error,
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
