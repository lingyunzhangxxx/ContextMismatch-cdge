#!/usr/bin/env python3
"""Summarize bidirectional governance-direction interventions."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    index = {
        (
            row["regime"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"], row["layer"], row["alpha"],
        ): row
        for row in rows
    }
    summaries = []
    for regime in ("obedience", "verification"):
        for layer in sorted({row["layer"] for row in rows}):
            for alpha in sorted({row["alpha"] for row in rows if row["alpha"] > 0}):
                effects = []
                by_task = defaultdict(list)
                edited_rows = []
                keys = sorted({
                    (
                        row["history_realization"], row["task_id"], row["label_swap"],
                        row["evidence_order_swap"],
                    )
                    for row in rows
                    if row["regime"] == regime and row["layer"] == layer
                })
                for realization, task_id, label_swap, order_swap in keys:
                    base = index[(regime, realization, task_id, label_swap, order_swap, layer, 0.0)]
                    edited = index[(regime, realization, task_id, label_swap, order_swap, layer, alpha)]
                    effect = edited["correct_logit_margin"] - base["correct_logit_margin"]
                    effects.append(effect)
                    by_task[task_id].append(effect)
                    edited_rows.append(edited)
                expected_positive = regime == "obedience"
                task_effects = {task: mean(values) for task, values in by_task.items()}
                summaries.append({
                    "history_style": rows[0].get("history_style", "natural"),
                    "history_depth": rows[0]["history_depth"],
                    "regime": regime,
                    "causal_test": "necessity_recovery" if expected_positive else "reverse_sufficiency",
                    "layer": layer,
                    "alpha": alpha,
                    "n": len(effects),
                    "edited_accuracy": mean([row["correct"] for row in edited_rows]),
                    "margin_effect_vs_alpha0": mean(effects),
                    "effect_min": min(effects),
                    "effect_max": max(effects),
                    "fraction_expected_direction": mean([
                        effect > 0 if expected_positive else effect < 0
                        for effect in effects
                    ]),
                    "task_cluster_sign_p": exact_cluster_sign_p(task_effects.values()),
                    "task_effects_json": json.dumps(task_effects, sort_keys=True),
                })
    output = args.output or args.input.with_suffix(".summary.csv")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
