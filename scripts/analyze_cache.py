#!/usr/bin/env python3
"""Analyze recurrent-state versus full-attention KV interventions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def exact_cluster_sign_p(values):
    values = list(values)
    observed = abs(mean(values))
    null = [
        abs(mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/mechanistic"))
    parser.add_argument("--input-name", default="cache_interventions.jsonl")
    parser.add_argument("--output-prefix", default="cache")
    args = parser.parse_args()
    path = args.input_dir / args.input_name
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    cells = defaultdict(list)
    for row in rows:
        cells[(row["regime"], row["history_depth"], row["intervention"])].append(row)
    summaries = []
    for (regime, depth, intervention), cell in sorted(cells.items()):
        summaries.append({
            "regime": regime,
            "history_depth": depth,
            "intervention": intervention,
            "n": len(cell),
            "accuracy": mean([row["correct"] for row in cell]),
            "correct_logit_margin_mean": mean(
                [row["correct_logit_margin"] for row in cell]
            ),
            "binary_correct_probability_mean": mean(
                [row["correct_probability_binary"] for row in cell]
            ),
        })
    write_csv(args.input_dir / f"{args.output_prefix}_summary.csv", summaries)

    index = {
        (
            row["regime"], row["history_depth"], row["history_realization"],
            row["task_id"], row["label_swap"], row["evidence_order_swap"],
            row["intervention"],
        ): row
        for row in rows
    }
    interventions = sorted({row["intervention"] for row in rows if row["intervention"] != "intact"})
    depths = sorted({row["history_depth"] for row in rows})
    keys = sorted({
        (
            row["history_depth"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"],
        )
        for row in rows
    })
    contrasts = []
    for depth in depths:
        for intervention in interventions:
            for regime in ("obedience", "verification"):
                by_task = defaultdict(list)
                by_label = defaultdict(list)
                by_order = defaultdict(list)
                effects = []
                for key in [key for key in keys if key[0] == depth]:
                    lookup = (regime,) + key
                    effect = (
                        index[lookup + (intervention,)]["correct_logit_margin"]
                        - index[lookup + ("intact",)]["correct_logit_margin"]
                    )
                    effects.append(effect)
                    by_task[key[2]].append(effect)
                    by_label[key[3]].append(effect)
                    by_order[key[4]].append(effect)
                task_effects = {task: mean(values) for task, values in by_task.items()}
                contrasts.append({
                    "history_depth": depth,
                    "intervention": intervention,
                    "regime": regime,
                    "effect_vs_intact": mean([
                        value for values in by_task.values() for value in values
                    ]),
                    "fraction_positive": mean([value > 0 for value in effects]),
                    "task_cluster_sign_p": exact_cluster_sign_p(task_effects.values()),
                    "task_effects_json": json.dumps(task_effects, sort_keys=True),
                    "label_swap_effects_json": json.dumps({
                        str(key): mean(values) for key, values in by_label.items()
                    }, sort_keys=True),
                    "order_swap_effects_json": json.dumps({
                        str(key): mean(values) for key, values in by_order.items()
                    }, sort_keys=True),
                })
    write_csv(args.input_dir / f"{args.output_prefix}_contrasts.csv", contrasts)

    gap_rows = []
    all_interventions = ["intact"] + interventions
    for depth in depths:
        for intervention in all_interventions:
            by_task = defaultdict(list)
            for key in [key for key in keys if key[0] == depth]:
                obey = index[("obedience",) + key + (intervention,)]["correct_logit_margin"]
                verify = index[("verification",) + key + (intervention,)]["correct_logit_margin"]
                by_task[key[2]].append(verify - obey)
            task_gaps = {task: mean(values) for task, values in by_task.items()}
            gap_rows.append({
                "history_depth": depth,
                "intervention": intervention,
                "verification_minus_obedience_margin_gap": mean([
                    value for values in by_task.values() for value in values
                ]),
                "task_cluster_sign_p": exact_cluster_sign_p(task_gaps.values()),
                "task_gaps_json": json.dumps(task_gaps, sort_keys=True),
            })
    write_csv(args.input_dir / f"{args.output_prefix}_mismatch_gap.csv", gap_rows)
    regimes = {row["regime"] for row in rows}
    realizations = {row["history_realization"] for row in rows}
    tasks = {row["task_id"] for row in rows}
    label_swaps = {row["label_swap"] for row in rows}
    order_swaps = {row["evidence_order_swap"] for row in rows}
    expected_factorial_rows = (
        len(regimes) * len(depths) * len(realizations) * len(tasks)
        * len(label_swaps) * len(order_swaps) * len(all_interventions)
    )
    cell_counts = defaultdict(int)
    for row in rows:
        cell_counts[(
            row["regime"], row["history_depth"], row["intervention"]
        )] += 1
    audit = {
        "rows": len(rows),
        "unique_keys": len(index),
        "duplicate_rows": len(rows) - len(index),
        "expected_observed_factorial_rows": expected_factorial_rows,
        "complete_observed_factorial": len(index) == expected_factorial_rows,
        "cell_count_min": min(cell_counts.values()),
        "cell_count_max": max(cell_counts.values()),
        "finite_margins": sum(math.isfinite(row["correct_logit_margin"]) for row in rows),
        "finite_binary_probabilities": sum(
            math.isfinite(row["correct_probability_binary"]) for row in rows
        ),
    }
    if "paired_all" in all_interventions:
        endpoint_errors = []
        for key in keys:
            for regime in ("obedience", "verification"):
                opposite = "verification" if regime == "obedience" else "obedience"
                endpoint_errors.append(abs(
                    index[(regime,) + key + ("paired_all",)]["correct_logit_margin"]
                    - index[(opposite,) + key + ("intact",)]["correct_logit_margin"]
                ))
        audit["paired_all_vs_target_intact_max_abs"] = max(endpoint_errors)
    (args.input_dir / f"{args.output_prefix}_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
