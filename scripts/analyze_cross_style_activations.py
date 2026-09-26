#!/usr/bin/env python3
"""Cross-history-style, leave-one-domain-out governance-direction transfer."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else float("nan")


def cosine(x, y):
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denominator) if denominator else float("nan")


def load_rows(root):
    metadata = root / "activation_metadata.jsonl"
    rows = [json.loads(line) for line in metadata.read_text().splitlines() if line.strip()]
    for row in rows:
        row["activation"] = np.load(
            root / row["activation_file"], allow_pickle=False
        ).astype(np.float32)
    return rows


def regime_direction(rows, layer):
    obedience = np.stack([
        row["activation"][layer] for row in rows if row["regime"] == "obedience"
    ])
    verification = np.stack([
        row["activation"][layer] for row in rows if row["regime"] == "verification"
    ])
    return obedience.mean(0) - verification.mean(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_rows(args.source_dir)
    target = load_rows(args.target_dir)
    depths = sorted(
        {row["history_depth"] for row in source}
        & {row["history_depth"] for row in target}
    )
    output = []
    for depth in depths:
        source_depth = [row for row in source if row["history_depth"] == depth]
        target_depth = [row for row in target if row["history_depth"] == depth]
        task_ids = sorted({row["task_id"] for row in target_depth})
        layer_count = source_depth[0]["activation"].shape[0]
        for layer in range(layer_count):
            predictions = []
            labels = []
            projections = []
            margins = []
            fold_effects = []
            direction_cosines = []
            for task_id in task_ids:
                train = [row for row in source_depth if row["task_id"] != task_id]
                test = [row for row in target_depth if row["task_id"] == task_id]
                direction = regime_direction(train, layer)
                norm = np.linalg.norm(direction)
                if norm == 0:
                    continue
                direction /= norm
                obedience_train = np.stack([
                    row["activation"][layer]
                    for row in train if row["regime"] == "obedience"
                ])
                verification_train = np.stack([
                    row["activation"][layer]
                    for row in train if row["regime"] == "verification"
                ])
                midpoint = 0.5 * (
                    float(obedience_train.mean(0) @ direction)
                    + float(verification_train.mean(0) @ direction)
                )
                scores = [float(row["activation"][layer] @ direction) for row in test]
                test_labels = [int(row["regime"] == "obedience") for row in test]
                predictions.extend([int(score > midpoint) for score in scores])
                labels.extend(test_labels)
                projections.extend(scores)
                margins.extend([row["correct_logit_margin"] for row in test])
                fold_effects.append(
                    mean([score for score, label in zip(scores, test_labels) if label == 1])
                    - mean([score for score, label in zip(scores, test_labels) if label == 0])
                )
                target_direction = regime_direction(test, layer)
                direction_cosines.append(cosine(direction, target_direction))
            output.append({
                "source_style": source[0].get("history_style", "natural"),
                "target_style": target[0].get("history_style", "natural"),
                "history_depth": depth,
                "layer": layer,
                "layer_label": "final_norm" if layer == layer_count - 1 else f"decoder_{layer}",
                "cross_style_lodo_accuracy": mean([p == y for p, y in zip(predictions, labels)]),
                "target_projection_effect_mean": mean(fold_effects),
                "target_projection_effect_min": min(fold_effects),
                "target_fraction_task_effect_positive": mean([effect > 0 for effect in fold_effects]),
                "target_regime_projection_correlation": pearson(projections, labels),
                "target_decision_margin_correlation": pearson(projections, margins),
                "source_target_direction_cosine_mean": mean(direction_cosines),
                "source_target_direction_cosine_min": min(direction_cosines),
                "n_examples": len(labels),
                "n_tasks": len(task_ids),
            })
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output[0].keys())
        writer.writeheader()
        writer.writerows(output)
    ranking = sorted(
        output,
        key=lambda row: (
            row["history_depth"],
            row["target_fraction_task_effect_positive"],
            row["source_target_direction_cosine_mean"],
            row["cross_style_lodo_accuracy"],
        ),
        reverse=True,
    )
    print(json.dumps(ranking[:20], indent=2))


if __name__ == "__main__":
    main()
