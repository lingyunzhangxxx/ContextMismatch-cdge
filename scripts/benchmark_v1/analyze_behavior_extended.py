#!/usr/bin/env python3
"""Preregistered secondary inference and heterogeneity for full Qwen3-8B behavior."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from .analyze_behavior import (
    audit,
    cluster_bootstrap,
    fresh_contrasts,
    interaction_effects,
    paired_effects,
    percentile,
    summarize_effects,
)
from .common import atomic_write_text, load_jsonl, sha256_file


def _item_means(effects: list[dict]) -> dict[tuple[str, str], float]:
    grouped = defaultdict(list)
    for row in effects:
        grouped[(row["benchmark"], row["item_id"])].append(float(row["effect"]))
    return {key: statistics.fmean(values) for key, values in grouped.items()}


def _benchmark_ci(values: list[float], seed: int, replicates: int = 10000) -> list[float]:
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        draws.append(statistics.fmean(values[rng.randrange(len(values))] for _ in values))
    return [percentile(draws, 0.025), percentile(draws, 0.975)]


def _one_sided_negative_sign_flip(
    values: list[float], seed: int, replicates: int = 100000
) -> float:
    observed = statistics.fmean(values)
    rng = random.Random(seed)
    at_least_as_negative = 0
    for _ in range(replicates):
        draw = statistics.fmean(value if rng.randrange(2) else -value for value in values)
        at_least_as_negative += draw <= observed
    return (at_least_as_negative + 1.0) / (replicates + 1.0)


def _holm(p_values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(p_values, key=p_values.get)
    adjusted = {}
    running = 0.0
    count = len(ordered)
    for index, key in enumerate(ordered):
        value = min(1.0, (count - index) * p_values[key])
        running = max(running, value)
        adjusted[key] = running
    return adjusted


def _benchmark_inference(effects: list[dict], seed: int = 271828) -> dict:
    item_values = _item_means(effects)
    by_benchmark = defaultdict(list)
    for (benchmark, _item), value in item_values.items():
        by_benchmark[benchmark].append(value)
    raw_p = {}
    result = {}
    for offset, (benchmark, values) in enumerate(sorted(by_benchmark.items())):
        raw_p[benchmark] = _one_sided_negative_sign_flip(values, seed + offset)
        result[benchmark] = {
            "n_items": len(values),
            "mean_effect": statistics.fmean(values),
            "ci95_item_cluster": _benchmark_ci(values, seed + 1000 + offset),
            "one_sided_negative_sign_flip_p": raw_p[benchmark],
        }
    adjusted = _holm(raw_p)
    for benchmark in result:
        result[benchmark]["holm_adjusted_p"] = adjusted[benchmark]
    return result


def _grouped_bootstrap(effects: list[dict], field: str) -> dict:
    return {
        str(value): {
            **summarize_effects([row for row in effects if str(row[field]) == str(value)]),
            "cluster_bootstrap": cluster_bootstrap(
                [row for row in effects if str(row[field]) == str(value)]
            ),
        }
        for value in sorted({row[field] for row in effects}, key=str)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    row_audit = audit(rows, environment.get("planned_rows"))
    if not row_audit["success"]:
        raise ValueError("refusing extended inference on a failed row audit")
    mismatch = paired_effects(rows, "obedience", "verification")
    reset = paired_effects(rows, "obedience_reset", "obedience")
    neutral_fresh = fresh_contrasts(rows, "neutral_length")
    verification_neutral = paired_effects(rows, "verification", "neutral_length")
    obedience_neutral = paired_effects(rows, "obedience", "neutral_length")
    role_interaction = interaction_effects(mismatch, "declared_role", "assistant", "collaborator")
    style_interaction = interaction_effects(
        mismatch, "history_style", "natural", "lexical_matched"
    )
    report = {
        "schema_version": 1,
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "audit": row_audit,
        "mismatch": {
            "summary": summarize_effects(mismatch),
            "cluster_bootstrap": cluster_bootstrap(mismatch),
            "benchmark_inference_holm_family": _benchmark_inference(mismatch),
            "benchmark_sign_count": {
                "negative": sum(value < 0 for value in summarize_effects(mismatch)["benchmark_means"].values()),
                "total": len(summarize_effects(mismatch)["benchmark_means"]),
            },
            "role": _grouped_bootstrap(mismatch, "declared_role"),
            "history_style": _grouped_bootstrap(mismatch, "history_style"),
            "label_swap": _grouped_bootstrap(mismatch, "label_swap"),
            "partition": _grouped_bootstrap(mismatch, "partition"),
        },
        "role_interaction_assistant_minus_collaborator": {
            "summary": summarize_effects(role_interaction),
            "cluster_bootstrap": cluster_bootstrap(role_interaction),
        },
        "style_interaction_natural_minus_lexical": {
            "summary": summarize_effects(style_interaction),
            "cluster_bootstrap": cluster_bootstrap(style_interaction),
        },
        "neutral_length_controls": {
            "neutral_minus_fresh": {
                "summary": summarize_effects(neutral_fresh),
                "cluster_bootstrap": cluster_bootstrap(neutral_fresh),
            },
            "verification_minus_neutral": {
                "summary": summarize_effects(verification_neutral),
                "cluster_bootstrap": cluster_bootstrap(verification_neutral),
            },
            "obedience_minus_neutral": {
                "summary": summarize_effects(obedience_neutral),
                "cluster_bootstrap": cluster_bootstrap(obedience_neutral),
            },
        },
        "reset": {
            "summary": summarize_effects(reset),
            "cluster_bootstrap": cluster_bootstrap(reset),
            "benchmark_sign_count_positive": sum(
                value > 0 for value in summarize_effects(reset)["benchmark_means"].values()
            ),
            "role": _grouped_bootstrap(reset, "declared_role"),
            "history_style": _grouped_bootstrap(reset, "history_style"),
            "label_swap": _grouped_bootstrap(reset, "label_swap"),
            "partition": _grouped_bootstrap(reset, "partition"),
        },
        "interpretation_boundary": {
            "pairwise_candidate_verification_not_official_leaderboard_accuracy": True,
            "fresh_correctness_not_used_as_a_filter": True,
            "partitions_retained_for_later_mechanism_and_mitigation_roles": True,
        },
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
