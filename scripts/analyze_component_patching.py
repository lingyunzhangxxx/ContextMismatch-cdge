#!/usr/bin/env python3
"""Summarize exact paired mixer/MLP output patching."""

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
    rows = [
        json.loads(line) for line in args.input.read_text().splitlines()
        if line.strip()
    ]
    index = {
        (
            row["regime"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"], row["layer"],
            row["component"], row["patch_coefficient"],
        ): row
        for row in rows
    }
    summaries = []
    layers = sorted({row["layer"] for row in rows})
    components = sorted({row["component"] for row in rows})
    coefficients = sorted({
        row["patch_coefficient"] for row in rows
        if row["patch_coefficient"] > 0
    })
    for regime in ("obedience", "verification"):
        expected_positive = regime == "obedience"
        for layer in layers:
            for component in components:
                keys = sorted({
                    (
                        row["history_realization"], row["task_id"],
                        row["label_swap"], row["evidence_order_swap"],
                    )
                    for row in rows
                    if row["regime"] == regime and row["layer"] == layer
                    and row["component"] == component
                })
                for coefficient in coefficients:
                    effects = []
                    gaps = []
                    by_task = defaultdict(list)
                    by_label = defaultdict(list)
                    by_order = defaultdict(list)
                    for realization, task_id, label_swap, order_swap in keys:
                        lookup = (
                            regime, realization, task_id, label_swap, order_swap,
                            layer, component,
                        )
                        base = index[lookup + (0.0,)]
                        edited = index[lookup + (coefficient,)]
                        opposite = (
                            "verification" if regime == "obedience" else "obedience"
                        )
                        target = index[(
                            opposite, realization, task_id, label_swap, order_swap,
                            layer, component, 0.0,
                        )]
                        effect = (
                            edited["correct_logit_margin"]
                            - base["correct_logit_margin"]
                        )
                        effects.append(effect)
                        gaps.append(
                            target["correct_logit_margin"]
                            - base["correct_logit_margin"]
                        )
                        by_task[task_id].append(effect)
                        by_label[label_swap].append(effect)
                        by_order[order_swap].append(effect)
                    task_effects = {
                        task: mean(values) for task, values in by_task.items()
                    }
                    summaries.append({
                        "regime": regime,
                        "causal_test": (
                            "verification_to_obedience_rescue"
                            if expected_positive
                            else "obedience_to_verification_reverse_induction"
                        ),
                        "layer": layer,
                        "component": component,
                        "patch_coefficient": coefficient,
                        "n": len(effects),
                        "margin_effect_vs_unpatched": mean(effects),
                        "opposite_minus_source_baseline_gap": mean(gaps),
                        "mediated_fraction_of_gap": (
                            mean(effects) / mean(gaps) if mean(gaps) else float("nan")
                        ),
                        "fraction_expected_direction": mean([
                            effect > 0 if expected_positive else effect < 0
                            for effect in effects
                        ]),
                        "task_cluster_sign_p": exact_cluster_sign_p(
                            task_effects.values()
                        ),
                        "task_effects_json": json.dumps(
                            task_effects, sort_keys=True
                        ),
                        "label_swap_effects_json": json.dumps({
                            str(key): mean(values) for key, values in by_label.items()
                        }, sort_keys=True),
                        "order_swap_effects_json": json.dumps({
                            str(key): mean(values) for key, values in by_order.items()
                        }, sort_keys=True),
                    })

    output = args.output or args.input.with_suffix(".summary.csv")
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    zero_rows = [row for row in rows if row["patch_coefficient"] == 0.0]
    zero_by_item = defaultdict(list)
    for row in zero_rows:
        zero_by_item[(
            row["regime"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"],
        )].append(row["correct_logit_margin"])
    audit = {
        "rows": len(rows),
        "unique_rows": len(index),
        "coefficient_zero_rows": len(zero_rows),
        "coefficient_zero_items": len(zero_by_item),
        "coefficient_zero_component_layer_invariance_max_abs": max(
            (max(values) - min(values) for values in zero_by_item.values()),
            default=float("nan"),
        ),
    }
    output.with_suffix(".audit.json").write_text(
        json.dumps(audit, indent=2) + "\n"
    )
    print(json.dumps({"audit": audit, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
