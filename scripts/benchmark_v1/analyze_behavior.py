#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .common import atomic_write_text, load_jsonl, sha256_file


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_effects(rows: list[dict], left: str, right: str) -> list[dict]:
    lookup = {}
    for row in rows:
        if row["condition"] not in (left, right):
            continue
        if row["history_style"] == "none":
            continue
        key = (
            row["benchmark"],
            row["item_id"],
            row["partition"],
            row["declared_role"],
            row["history_style"],
            row["history_depth"],
            row["label_swap"],
        )
        lookup[(key, row["condition"])] = row
    result = []
    keys = sorted({key for key, _ in lookup})
    for key in keys:
        if (key, left) not in lookup or (key, right) not in lookup:
            continue
        left_row, right_row = lookup[(key, left)], lookup[(key, right)]
        result.append(
            {
                "benchmark": key[0],
                "item_id": key[1],
                "partition": key[2],
                "declared_role": key[3],
                "history_style": key[4],
                "history_depth": key[5],
                "label_swap": key[6],
                "effect": left_row["correct_logit_margin"] - right_row["correct_logit_margin"],
            }
        )
    return result


def summarize_effects(effects: list[dict]) -> dict:
    if not effects:
        return {
            "n_pairs": 0,
            "mean_effect": None,
            "equal_weight_benchmark_mean": None,
            "fraction_predicted_negative": None,
            "benchmark_means": {},
        }
    groups = defaultdict(list)
    for row in effects:
        groups[row["benchmark"]].append(float(row["effect"]))
    benchmark_means = {key: mean(values) for key, values in sorted(groups.items())}
    return {
        "n_pairs": len(effects),
        "mean_effect": mean([float(row["effect"]) for row in effects]),
        "equal_weight_benchmark_mean": mean(list(benchmark_means.values())),
        "fraction_predicted_negative": mean([float(row["effect"] < 0) for row in effects]),
        "benchmark_means": benchmark_means,
    }


def stratified_effects(effects: list[dict], fields: tuple[str, ...]) -> dict:
    """Expose preregistered heterogeneity checks without changing the pooled estimand."""
    report = {}
    for field in fields:
        values = sorted({str(row[field]) for row in effects})
        report[field] = {
            value: summarize_effects([row for row in effects if str(row[field]) == value])
            for value in values
        }
    return report


def interaction_effects(
    effects: list[dict], field: str, left_value: str, right_value: str
) -> list[dict]:
    lookup = {}
    key_fields = (
        "benchmark",
        "item_id",
        "partition",
        "declared_role",
        "history_style",
        "history_depth",
        "label_swap",
    )
    for row in effects:
        key = tuple(row[name] for name in key_fields if name != field)
        lookup[(key, str(row[field]))] = row
    result = []
    for key in sorted({key for key, _ in lookup}):
        if (key, left_value) not in lookup or (key, right_value) not in lookup:
            continue
        left, right = lookup[(key, left_value)], lookup[(key, right_value)]
        result.append(
            {
                **{name: left[name] for name in key_fields},
                "effect": float(left["effect"]) - float(right["effect"]),
            }
        )
    return result


def fresh_contrasts(rows: list[dict], condition: str) -> list[dict]:
    """Pair each styled history condition with its one style-free fresh row."""
    fresh = {}
    for row in rows:
        if row["condition"] != "fresh":
            continue
        key = (
            row["benchmark"],
            row["item_id"],
            row["partition"],
            row["declared_role"],
            row["history_depth"],
            row["label_swap"],
        )
        fresh[key] = row
    result = []
    for row in rows:
        if row["condition"] != condition or row["history_style"] == "none":
            continue
        key = (
            row["benchmark"],
            row["item_id"],
            row["partition"],
            row["declared_role"],
            row["history_depth"],
            row["label_swap"],
        )
        if key not in fresh:
            continue
        result.append(
            {
                "benchmark": row["benchmark"],
                "item_id": row["item_id"],
                "partition": row["partition"],
                "declared_role": row["declared_role"],
                "history_style": row["history_style"],
                "history_depth": row["history_depth"],
                "label_swap": row["label_swap"],
                "effect": float(row["correct_logit_margin"])
                - float(fresh[key]["correct_logit_margin"]),
            }
        )
    return result


def cluster_bootstrap(effects: list[dict], replicates: int = 10_000, seed: int = 314159) -> dict:
    by_benchmark_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in effects:
        by_benchmark_item[row["benchmark"]][row["item_id"]].append(float(row["effect"]))
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        benchmark_draws = []
        for benchmark in sorted(by_benchmark_item):
            items = sorted(by_benchmark_item[benchmark])
            sampled = [items[rng.randrange(len(items))] for _ in items]
            benchmark_draws.append(mean([value for item in sampled for value in by_benchmark_item[benchmark][item]]))
        draws.append(mean(benchmark_draws))
    return {
        "replicates": replicates,
        "seed": seed,
        "estimate": mean(draws),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
    }


def audit(rows: list[dict], planned_rows: int | None) -> dict:
    keys = Counter(row.get("job_key") for row in rows)
    duplicate_keys = [key for key, count in keys.items() if count != 1]
    nonfinite = []
    for row in rows:
        if not all(math.isfinite(float(row[name])) for name in ("logit_a", "logit_b", "correct_logit_margin")):
            nonfinite.append(row.get("job_key"))
    suffixes: dict[tuple, set[str]] = defaultdict(set)
    for row in rows:
        suffixes[(row["item_id"], row["label_swap"])].add(row["suffix_token_sha256_int32_le"])
    suffix_mismatch = [key for key, hashes in suffixes.items() if len(hashes) != 1]
    return {
        "row_count": len(rows),
        "planned_rows": planned_rows,
        "complete": planned_rows is None or planned_rows == len(rows),
        "duplicate_job_keys": duplicate_keys[:20],
        "nonfinite_job_keys": nonfinite[:20],
        "paired_suffix_mismatch_keys": suffix_mismatch[:20],
        "success": not duplicate_keys and not nonfinite and not suffix_mismatch and (planned_rows is None or planned_rows == len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text()) if args.environment else {}
    planned = environment.get("planned_rows")
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        selected = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "n": len(selected),
            "mean_margin": mean([float(row["correct_logit_margin"]) for row in selected]),
            "binary_accuracy": mean([float(row["correct"]) for row in selected]),
            "unsupported_preference_following": mean([float(row["followed_unsupported_preference"]) for row in selected]),
            "benchmark_binary_accuracy": {
                benchmark: mean(
                    [float(row["correct"]) for row in selected if row["benchmark"] == benchmark]
                )
                for benchmark in sorted({row["benchmark"] for row in selected})
            },
        }
    mismatch = paired_effects(rows, "obedience", "verification")
    reset = paired_effects(rows, "obedience_reset", "obedience")
    role_interaction = interaction_effects(mismatch, "declared_role", "assistant", "collaborator")
    style_interaction = interaction_effects(mismatch, "history_style", "natural", "lexical_matched")
    neutral_fresh = fresh_contrasts(rows, "neutral_length")
    verification_neutral = paired_effects(rows, "verification", "neutral_length")
    obedience_neutral = paired_effects(rows, "obedience", "neutral_length")
    report = {
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "audit": audit(rows, planned),
        "condition_summary": by_condition,
        "mismatch_obedience_minus_verification": summarize_effects(mismatch),
        "mismatch_cluster_bootstrap": cluster_bootstrap(mismatch) if mismatch else None,
        "mismatch_strata": stratified_effects(
            mismatch,
            ("benchmark", "declared_role", "history_style", "history_depth", "label_swap", "partition"),
        ),
        "mismatch_depth_cluster_bootstrap": {
            str(depth): cluster_bootstrap(
                [row for row in mismatch if str(row["history_depth"]) == str(depth)]
            )
            for depth in sorted({row["history_depth"] for row in mismatch})
        },
        "role_interaction_assistant_minus_collaborator": summarize_effects(role_interaction),
        "role_interaction_cluster_bootstrap": cluster_bootstrap(role_interaction)
        if role_interaction
        else None,
        "style_interaction_natural_minus_lexical": summarize_effects(style_interaction),
        "style_interaction_cluster_bootstrap": cluster_bootstrap(style_interaction)
        if style_interaction
        else None,
        "neutral_length_controls": {
            "neutral_minus_fresh": summarize_effects(neutral_fresh),
            "verification_minus_neutral": summarize_effects(verification_neutral),
            "obedience_minus_neutral": summarize_effects(obedience_neutral),
        },
        "reset_minus_obedience": summarize_effects(reset),
        "reset_cluster_bootstrap": cluster_bootstrap(reset) if reset else None,
        "reset_strata": stratified_effects(
            reset,
            ("benchmark", "declared_role", "history_style", "history_depth", "label_swap", "partition"),
        ),
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
