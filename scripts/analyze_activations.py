#!/usr/bin/env python3
"""Leave-one-task-domain-out analysis of final-token governance activations."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else float("nan")


def cosine(x, y):
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom else float("nan")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("results/mechanistic"))
    args = parser.parse_args()
    metadata_path = args.input_dir / "activation_metadata.jsonl"
    rows = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["activation"] = np.load(
            args.input_dir / row["activation_file"], allow_pickle=False
        ).astype(np.float32)
    output = []
    for depth in sorted({row["history_depth"] for row in rows}):
        depth_rows = [row for row in rows if row["history_depth"] == depth]
        layer_count = depth_rows[0]["activation"].shape[0]
        task_ids = sorted({row["task_id"] for row in depth_rows})
        for layer in range(layer_count):
            predictions = []
            labels = []
            projections = []
            margins = []
            fold_effects = []
            task_directions = {}
            for task_id in task_ids:
                train = [row for row in depth_rows if row["task_id"] != task_id]
                test = [row for row in depth_rows if row["task_id"] == task_id]
                obey_train = np.stack([
                    row["activation"][layer] for row in train if row["regime"] == "obedience"
                ])
                verify_train = np.stack([
                    row["activation"][layer] for row in train if row["regime"] == "verification"
                ])
                direction = obey_train.mean(0) - verify_train.mean(0)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction /= norm
                midpoint = 0.5 * (
                    float(obey_train.mean(0) @ direction)
                    + float(verify_train.mean(0) @ direction)
                )
                test_scores = [float(row["activation"][layer] @ direction) for row in test]
                test_labels = [1 if row["regime"] == "obedience" else 0 for row in test]
                predictions.extend([int(score > midpoint) for score in test_scores])
                labels.extend(test_labels)
                projections.extend(test_scores)
                margins.extend([row["correct_logit_margin"] for row in test])
                fold_effects.append(
                    mean([s for s, y in zip(test_scores, test_labels) if y == 1])
                    - mean([s for s, y in zip(test_scores, test_labels) if y == 0])
                )
                task_rows = [row for row in depth_rows if row["task_id"] == task_id]
                task_directions[task_id] = (
                    np.stack([
                        row["activation"][layer]
                        for row in task_rows if row["regime"] == "obedience"
                    ]).mean(0)
                    - np.stack([
                        row["activation"][layer]
                        for row in task_rows if row["regime"] == "verification"
                    ]).mean(0)
                )
            direction_cosines = [
                cosine(task_directions[a], task_directions[b])
                for index, a in enumerate(task_ids)
                for b in task_ids[index + 1 :]
            ]
            output.append({
                "history_depth": depth,
                "layer": layer,
                "layer_label": "final_norm" if layer == layer_count - 1 else f"decoder_{layer}",
                "lodo_accuracy": mean([p == y for p, y in zip(predictions, labels)]),
                "lodo_projection_effect_mean": mean(fold_effects),
                "lodo_projection_effect_min": min(fold_effects),
                "regime_projection_correlation": pearson(projections, labels),
                "decision_margin_correlation": pearson(projections, margins),
                "cross_task_direction_cosine_mean": mean(direction_cosines),
                "cross_task_direction_cosine_min": min(direction_cosines),
                "n_examples": len(labels),
                "n_tasks": len(task_ids),
            })
    write_csv(args.input_dir / "probe_layerwise.csv", output)
    ranking = []
    for depth in sorted({row["history_depth"] for row in output}):
        depth_ranking = sorted(
            [row for row in output if row["history_depth"] == depth],
            key=lambda row: (
                row["lodo_accuracy"],
                row["cross_task_direction_cosine_mean"],
                abs(row["decision_margin_correlation"]),
            ),
            reverse=True,
        )
        ranking.extend(depth_ranking[:12])
    (args.input_dir / "probe_top_layers.json").write_text(
        json.dumps(ranking, indent=2) + "\n"
    )
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
