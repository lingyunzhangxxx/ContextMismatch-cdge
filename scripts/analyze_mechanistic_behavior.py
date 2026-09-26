#!/usr/bin/env python3
"""Paired behavioral summaries for mechanistic activation collections."""

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


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    path = args.input_dir / "activation_metadata.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    index = {
        (
            row["regime"], row["history_depth"], row["history_realization"],
            row["task_id"], row["label_swap"], row["evidence_order_swap"],
        ): row
        for row in rows
    }
    pair_keys = sorted({key[1:] for key in index})
    summaries = []
    strata = []
    for depth in sorted({key[0] for key in pair_keys}):
        depth_keys = [key for key in pair_keys if key[0] == depth]
        effects = []
        by_task = defaultdict(list)
        by_stratum = defaultdict(list)
        for key in depth_keys:
            verification = index[("verification",) + key]
            obedience = index[("obedience",) + key]
            effect = (
                obedience["correct_logit_margin"]
                - verification["correct_logit_margin"]
            )
            effects.append(effect)
            by_task[key[2]].append(effect)
            by_stratum[(key[3], key[4])].append(effect)
        task_effects = {task: mean(values) for task, values in by_task.items()}
        verification_rows = [index[("verification",) + key] for key in depth_keys]
        obedience_rows = [index[("obedience",) + key] for key in depth_keys]
        summaries.append({
            "history_style": rows[0].get("history_style", "natural"),
            "history_depth": depth,
            "n_pairs": len(effects),
            "verification_accuracy": mean([row["correct"] for row in verification_rows]),
            "obedience_accuracy": mean([row["correct"] for row in obedience_rows]),
            "verification_margin_mean": mean([row["correct_logit_margin"] for row in verification_rows]),
            "obedience_margin_mean": mean([row["correct_logit_margin"] for row in obedience_rows]),
            "obedience_minus_verification_margin": mean(effects),
            "paired_effect_min": min(effects),
            "paired_effect_max": max(effects),
            "paired_fraction_negative": mean([effect < 0 for effect in effects]),
            "task_cluster_sign_p": exact_cluster_sign_p(task_effects.values()),
            "task_effects_json": json.dumps(task_effects, sort_keys=True),
        })
        for (label_swap, order_swap), values in sorted(by_stratum.items()):
            strata.append({
                "history_depth": depth,
                "label_swap": label_swap,
                "evidence_order_swap": order_swap,
                "n": len(values),
                "effect_mean": mean(values),
                "effect_min": min(values),
                "effect_max": max(values),
                "all_negative": all(value < 0 for value in values),
            })
    write_csv(args.input_dir / "behavior_summary.csv", summaries)
    write_csv(args.input_dir / "behavior_strata.csv", strata)
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
