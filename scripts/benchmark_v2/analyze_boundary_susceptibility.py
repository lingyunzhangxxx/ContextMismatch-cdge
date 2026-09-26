#!/usr/bin/env python3
"""Cross-split validation of decision-boundary susceptibility to context mismatch."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


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


def _read_rows(path: Path, expected_rows: int) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if len(rows) != expected_rows:
        raise ValueError(f"{path}: observed {len(rows)} rows, expected {expected_rows}")
    keys = [row.get("job_key") for row in rows]
    if len(set(keys)) != expected_rows or any(not isinstance(key, str) for key in keys):
        raise ValueError(f"{path}: job keys are missing or non-unique")
    for row in rows:
        if row.get("task_requirement") not in REQUIREMENTS:
            raise ValueError(f"{path}: invalid task requirement")
        if row.get("history_condition") not in {"obedience", "verification"}:
            raise ValueError(f"{path}: invalid history condition")
        if not math.isfinite(float(row.get("task_aligned_margin", math.nan))):
            raise ValueError(f"{path}: non-finite task-aligned margin")
        if not isinstance(row.get("task_aligned_correct"), bool):
            raise ValueError(f"{path}: invalid task-aligned correctness")
    return rows


def _pair_rows(rows: Iterable[dict], requirement: str) -> list[dict]:
    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if row["task_requirement"] != requirement:
            continue
        grouped[tuple(row[field] for field in PAIR_FIELDS)][row["history_condition"]] = row
    matched_condition, mismatched_condition = ORIENTATION[requirement]
    pairs = []
    for key, conditions in grouped.items():
        if set(conditions) != {"obedience", "verification"}:
            raise ValueError(f"incomplete history pair: {key}")
        matched = conditions[matched_condition]
        mismatched = conditions[mismatched_condition]
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
    if len(pairs) != 1536:
        raise ValueError(f"{requirement}: observed {len(pairs)} pairs, expected 1536")
    return pairs


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _band(row: dict, q25: float, q75: float) -> str | None:
    if not row["matched_correct"]:
        return None
    margin = row["matched_margin"]
    if margin <= q25:
        return "boundary"
    if margin >= q75:
        return "robust"
    return "middle"


def _band_summary(rows: list[dict], q25: float, q75: float) -> dict[str, dict]:
    output = {}
    for name in ("boundary", "middle", "robust"):
        selected = [row for row in rows if _band(row, q25, q75) == name]
        if not selected:
            raise ValueError(f"empty confidence band: {name}")
        flips = [row["matched_correct"] and not row["mismatched_correct"] for row in selected]
        output[name] = {
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
    return output


def _percentile(values: list[float], probability: float) -> float:
    return _quantile(values, probability)


def _boundary_contrast(rows: list[dict], q25: float, q75: float) -> float:
    summaries = _band_summary(rows, q25, q75)
    return (
        summaries["boundary"]["correct_to_wrong_flip_rate"]
        - summaries["robust"]["correct_to_wrong_flip_rate"]
    )


def _cluster_bootstrap(
    rows: list[dict], q25: float, q75: float, replicates: int, seed: int
) -> dict:
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_item[row["item_id"]].append(row)
    item_ids = sorted(by_item)
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        sampled = []
        for item_id in (rng.choice(item_ids) for _ in item_ids):
            sampled.extend(by_item[item_id])
        samples.append(_boundary_contrast(sampled, q25, q75))
    return {
        "estimate": _boundary_contrast(rows, q25, q75),
        "ci95": [_percentile(samples, 0.025), _percentile(samples, 0.975)],
        "replicates": replicates,
        "seed": seed,
        "cluster": "item_id",
    }


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered) and ordered[end + 1][0] == ordered[start][0]:
            end += 1
        average = ((start + 1) + (end + 1)) / 2.0
        for _, index in ordered[start : end + 1]:
            ranks[index] = average
        start = end + 1
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right))
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator else 0.0


def _benchmark_difficulty_test(rows: list[dict]) -> dict:
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
    difficulty = [row["difficulty"] for row in records]
    drops = [row["accuracy_drop"] for row in records]
    observed = _pearson(_ranks(difficulty), _ranks(drops))
    permutations = list(itertools.permutations(drops))
    null = [_pearson(_ranks(difficulty), _ranks(list(permutation))) for permutation in permutations]
    p_value = sum(abs(value) >= abs(observed) - 1e-12 for value in null) / len(null)
    return {
        "benchmarks": records,
        "spearman_difficulty_vs_accuracy_drop": observed,
        "two_sided_exact_permutation_p": p_value,
        "permutations": len(permutations),
        "interpretation": "six-benchmark ecological diagnostic; not an item-level difficulty test",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--discovery", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=6144)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=58123)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    discovery_rows = _read_rows(args.discovery, args.expected_rows)
    replication_rows = _read_rows(args.replication, args.expected_rows)
    results = {}
    validation_passes = []
    for offset, requirement in enumerate(REQUIREMENTS):
        discovery_pairs = _pair_rows(discovery_rows, requirement)
        replication_pairs = _pair_rows(replication_rows, requirement)
        discovery_correct_margins = [
            row["matched_margin"] for row in discovery_pairs if row["matched_correct"]
        ]
        q25 = _quantile(discovery_correct_margins, 0.25)
        q75 = _quantile(discovery_correct_margins, 0.75)
        bootstrap = _cluster_bootstrap(
            replication_pairs,
            q25,
            q75,
            args.bootstrap_replicates,
            args.bootstrap_seed + offset,
        )
        validation_passes.append(bootstrap["ci95"][0] > 0.0)
        results[requirement] = {
            "orientation": {
                "matched_history": ORIENTATION[requirement][0],
                "mismatched_history": ORIENTATION[requirement][1],
            },
            "discovery_derived_thresholds": {
                "matched_correct_margin_q25": q25,
                "matched_correct_margin_q75": q75,
                "matched_correct_n": len(discovery_correct_margins),
            },
            "discovery": _band_summary(discovery_pairs, q25, q75),
            "replication": _band_summary(replication_pairs, q25, q75),
            "replication_boundary_minus_robust_flip_rate": bootstrap,
            "benchmark_difficulty_diagnostic": {
                "discovery": _benchmark_difficulty_test(discovery_pairs),
                "replication": _benchmark_difficulty_test(replication_pairs),
            },
        }
    output = {
        "schema_version": 1,
        "analysis_status": "retrospective_cross_split_validation",
        "not_preregistered": True,
        "final_test_used": False,
        "hypothesis": (
            "decision errors concentrate near the matched decision boundary rather than "
            "increasing monotonically with benchmark difficulty"
        ),
        "discovery_input": str(args.discovery),
        "discovery_input_sha256": sha256_file(args.discovery),
        "replication_input": str(args.replication),
        "replication_input_sha256": sha256_file(args.replication),
        "audit": {
            "success": True,
            "discovery_rows": len(discovery_rows),
            "replication_rows": len(replication_rows),
            "unique_discovery_job_keys": len({row["job_key"] for row in discovery_rows}),
            "unique_replication_job_keys": len({row["job_key"] for row in replication_rows}),
        },
        "requirements": results,
        "replication_boundary_validation_passed_both_requirements": all(validation_passes),
        "interpretation_boundary": {
            "difficulty_proxy": "matched task-aligned accuracy at benchmark level",
            "boundary_proxy": "matched-correct task-aligned margin",
            "causal_scope": "history-condition effects within the existing randomized paired design",
            "prospective_confirmation_still_required": True,
        },
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audit": output["audit"],
                "passed_both_requirements": output[
                    "replication_boundary_validation_passed_both_requirements"
                ],
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
