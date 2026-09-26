#!/usr/bin/env python3
"""Prospective test of frozen decision-boundary susceptibility thresholds."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file


PAIR_FIELDS = (
    "benchmark",
    "item_id",
    "partition",
    "declared_role",
    "history_style",
    "history_realization",
    "history_depth",
    "task_requirement",
    "label_swap",
)
REQUIREMENTS = ("independent_verification", "delegated_choice")
ORIENTATION = {
    "independent_verification": ("verification", "obedience"),
    "delegated_choice": ("obedience", "verification"),
}


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile of empty sample")
    position = (len(ordered) - 1) * probability
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_rows(rows: list[dict], environment: dict, protocol: dict) -> dict:
    planned = int(environment.get("planned_rows", -1))
    keys = [row.get("job_key") for row in rows]
    counts = Counter(keys)
    duplicates = [key for key, count in counts.items() if count != 1]
    numeric = ("logit_a", "logit_b", "task_aligned_margin", "factual_margin")
    nonfinite = [
        row.get("job_key")
        for row in rows
        if not all(math.isfinite(float(row.get(field, math.nan))) for field in numeric)
    ]
    observed_key_sha = hashlib.sha256(
        (("\n".join(sorted(str(key) for key in keys))) + "\n").encode("utf-8")
    ).hexdigest()
    expected_key_sha = environment.get("expected_key_sha256")
    allowed_partition = protocol["selection"]["partition"]
    invalid_partitions = sorted(
        {str(row.get("partition")) for row in rows if row.get("partition") != allowed_partition}
    )
    invalid_requirements = sorted(
        {str(row.get("task_requirement")) for row in rows if row.get("task_requirement") not in REQUIREMENTS}
    )
    success = (
        len(rows) == planned == int(protocol["stages"]["prospective_boundary"]["expected_rows"])
        and len(counts) == planned
        and not duplicates
        and not nonfinite
        and observed_key_sha == expected_key_sha
        and not invalid_partitions
        and not invalid_requirements
        and environment.get("stage") == "prospective_boundary"
        and environment.get("final_test_open") is False
        and environment.get("production_rollout_approved") is False
    )
    return {
        "success": success,
        "row_count": len(rows),
        "planned_rows": planned,
        "unique_job_keys": len(counts),
        "duplicate_job_keys": duplicates[:20],
        "nonfinite_job_keys": nonfinite[:20],
        "observed_key_sha256": observed_key_sha,
        "expected_key_sha256": expected_key_sha,
        "invalid_partitions": invalid_partitions,
        "invalid_requirements": invalid_requirements,
    }


def _pair_rows(rows: Iterable[dict], requirement: str) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["task_requirement"] != requirement:
            continue
        key = tuple(row[field] for field in PAIR_FIELDS)
        history = row["history_condition"]
        if history in grouped[key]:
            raise ValueError(f"duplicate history cell: {key}:{history}")
        grouped[key][history] = row
    matched_history, mismatched_history = ORIENTATION[requirement]
    pairs = []
    for key, conditions in sorted(grouped.items()):
        if set(conditions) != {"obedience", "verification"}:
            raise ValueError(f"incomplete prospective history pair: {key}")
        matched = conditions[matched_history]
        mismatched = conditions[mismatched_history]
        for field in ("candidate_payload_sha256", "user_selected_label", "task_correct_label"):
            if matched[field] != mismatched[field]:
                raise ValueError(f"paired field changed: {field}:{key}")
        if matched["suffix_token_sha256_int32_le"] != mismatched["suffix_token_sha256_int32_le"]:
            raise ValueError(f"paired suffix changed: {key}")
        pairs.append(
            {
                "benchmark": matched["benchmark"],
                "item_id": matched["item_id"],
                "matched_margin": float(matched["task_aligned_margin"]),
                "mismatched_margin": float(mismatched["task_aligned_margin"]),
                "matched_correct": bool(matched["task_aligned_correct"]),
                "mismatched_correct": bool(mismatched["task_aligned_correct"]),
            }
        )
    return pairs


def _band(row: dict, boundary_max: float, robust_min: float) -> str | None:
    if not row["matched_correct"]:
        return None
    margin = row["matched_margin"]
    if margin <= boundary_max:
        return "boundary"
    if margin >= robust_min:
        return "robust"
    return "middle"


def _summarize_band(
    rows: list[dict],
    band: str,
    boundary_max: float,
    robust_min: float,
    *,
    required: bool = True,
) -> dict:
    selected = [row for row in rows if _band(row, boundary_max, robust_min) == band]
    if not selected:
        if required:
            raise ValueError(f"empty prospective band: {band}")
        return {
            "n": 0,
            "unique_items": 0,
            "matched_margin_min": None,
            "matched_margin_max": None,
            "correct_to_wrong_flips": 0,
            "correct_to_wrong_flip_rate": None,
            "mean_mismatch_margin_shift": None,
        }
    flips = [not row["mismatched_correct"] for row in selected]
    return {
        "n": len(selected),
        "unique_items": len({row["item_id"] for row in selected}),
        "matched_margin_min": min(row["matched_margin"] for row in selected),
        "matched_margin_max": max(row["matched_margin"] for row in selected),
        "correct_to_wrong_flips": sum(flips),
        "correct_to_wrong_flip_rate": sum(flips) / len(selected),
        "mean_mismatch_margin_shift": sum(
            row["mismatched_margin"] - row["matched_margin"] for row in selected
        )
        / len(selected),
    }


def _contrast(rows: list[dict], boundary_max: float, robust_min: float) -> float:
    boundary = _summarize_band(rows, "boundary", boundary_max, robust_min)
    robust = _summarize_band(rows, "robust", boundary_max, robust_min)
    return boundary["correct_to_wrong_flip_rate"] - robust["correct_to_wrong_flip_rate"]


def _cluster_bootstrap(
    rows: list[dict], boundary_max: float, robust_min: float, replicates: int, seed: int
) -> dict:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    items = sorted(by_item)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = []
        for item in (rng.choice(items) for _ in items):
            sampled.extend(by_item[item])
        draws.append(_contrast(sampled, boundary_max, robust_min))
    estimate = _contrast(rows, boundary_max, robust_min)
    return {
        "estimate": estimate,
        "ci95_two_sided": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "one_sided_95_lower_bound": _percentile(draws, 0.05),
        "replicates": replicates,
        "seed": seed,
        "cluster": "item_id",
    }


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    result = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[cursor][0]:
            end += 1
        rank = ((cursor + 1) + (end + 1)) / 2.0
        for _, index in ordered[cursor : end + 1]:
            result[index] = rank
        cursor = end + 1
    return result


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _difficulty_diagnostic(rows: list[dict]) -> dict:
    by_benchmark: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_benchmark[row["benchmark"]].append(row)
    records = []
    for benchmark, values in sorted(by_benchmark.items()):
        matched_accuracy = sum(row["matched_correct"] for row in values) / len(values)
        mismatched_accuracy = sum(row["mismatched_correct"] for row in values) / len(values)
        records.append(
            {
                "benchmark": benchmark,
                "n": len(values),
                "matched_accuracy": matched_accuracy,
                "mismatched_accuracy": mismatched_accuracy,
                "difficulty": 1.0 - matched_accuracy,
                "accuracy_drop": matched_accuracy - mismatched_accuracy,
            }
        )
    difficulties = [row["difficulty"] for row in records]
    drops = [row["accuracy_drop"] for row in records]
    observed = _pearson(_ranks(difficulties), _ranks(drops))
    permutations = list(itertools.permutations(drops))
    null = [_pearson(_ranks(difficulties), _ranks(list(value))) for value in permutations]
    p_value = sum(abs(value) >= abs(observed) - 1e-12 for value in null) / len(null)
    return {
        "benchmarks": records,
        "spearman_difficulty_vs_accuracy_drop": observed,
        "two_sided_exact_permutation_p": p_value,
        "permutations": len(permutations),
        "interpretation": "secondary six-benchmark ecological diagnostic only",
    }


def analyze(rows: list[dict], environment: dict, protocol: dict) -> dict:
    audit = _validate_rows(rows, environment, protocol)
    if not audit["success"]:
        raise ValueError("prospective row audit failed")
    expected_pairs = len(rows) // (len(REQUIREMENTS) * 2)
    requirement_reports = {}
    passes = []
    for requirement in REQUIREMENTS:
        pairs = _pair_rows(rows, requirement)
        if len(pairs) != expected_pairs:
            raise ValueError(
                f"{requirement}: observed {len(pairs)} pairs, expected {expected_pairs}"
            )
        thresholds = protocol["frozen_margin_bands"][requirement]
        boundary_max = float(thresholds["boundary_max"])
        robust_min = float(thresholds["robust_min"])
        bootstrap = _cluster_bootstrap(
            pairs,
            boundary_max,
            robust_min,
            int(protocol["inference"]["bootstrap_replicates"]),
            int(protocol["inference"]["seeds"][requirement]),
        )
        passed = (
            bootstrap["estimate"] > 0.0
            and bootstrap["one_sided_95_lower_bound"] > 0.0
        )
        passes.append(passed)
        requirement_reports[requirement] = {
            "orientation": {
                "matched_history": ORIENTATION[requirement][0],
                "mismatched_history": ORIENTATION[requirement][1],
            },
            "frozen_thresholds": {
                "boundary_max": boundary_max,
                "robust_min": robust_min,
            },
            "pairs": len(pairs),
            "matched_correct_pairs": sum(row["matched_correct"] for row in pairs),
            "bands": {
                "boundary": _summarize_band(
                    pairs, "boundary", boundary_max, robust_min
                ),
                "middle": _summarize_band(
                    pairs, "middle", boundary_max, robust_min, required=False
                ),
                "robust": _summarize_band(
                    pairs, "robust", boundary_max, robust_min
                ),
            },
            "boundary_minus_robust_flip_rate": bootstrap,
            "success_rule_passed": passed,
            "benchmark_difficulty_diagnostic": _difficulty_diagnostic(pairs),
        }
    return {
        "schema_version": 1,
        "analysis_status": "prospective_preregistered_confirmation",
        "preregistered": True,
        "postselected_on_model_outputs": False,
        "final_test_used": False,
        "protocol_sha256": environment["contract_sha256"],
        "input_sha256": environment.get("output_sha256"),
        "audit": {**audit, "matched_pairs": expected_pairs * len(REQUIREMENTS)},
        "requirements": requirement_reports,
        "success_rule_passed_both_requirements": all(passes),
        "success_rule": protocol["inference"]["success_rule"],
        "interpretation_boundary": {
            "primary_claim": "frozen matched-margin bands predict correct-to-wrong mismatch flips",
            "difficulty_analysis_is_secondary": True,
            "broad_open_ended_deployment_not_tested": True,
        },
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing analysis: {args.output}")
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "frozen_before_any_prospective_boundary_model_forward":
        raise ValueError("prospective protocol is not frozen")
    environment = {
        **environment,
        "output_sha256": sha256_file(args.input),
    }
    report = analyze(rows, environment, protocol)
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit": report["audit"],
                "success_rule_passed_both_requirements": report[
                    "success_rule_passed_both_requirements"
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
