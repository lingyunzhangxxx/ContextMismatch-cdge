#!/usr/bin/env python3
"""Summarize paired residual-patching mediation effects."""

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
    parser.add_argument(
        "--activation-metadata",
        type=Path,
        help="optional full-forward metadata used to audit coefficient-0 equivalence",
    )
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    index = {
        (
            row["regime"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"], row["layer"],
            row["patch_coefficient"],
        ): row
        for row in rows
    }
    summaries = []
    for regime in ("obedience", "verification"):
        expected_positive = regime == "obedience"
        for layer in sorted({row["layer"] for row in rows}):
            for coefficient in sorted({
                row["patch_coefficient"] for row in rows
                if row["patch_coefficient"] > 0
            }):
                effects = []
                gaps = []
                by_task = defaultdict(list)
                by_label_swap = defaultdict(list)
                by_order_swap = defaultdict(list)
                keys = sorted({
                    (
                        row["history_realization"], row["task_id"],
                        row["label_swap"], row["evidence_order_swap"],
                    )
                    for row in rows
                    if row["regime"] == regime and row["layer"] == layer
                })
                for realization, task_id, label_swap, order_swap in keys:
                    base = index[(regime, realization, task_id, label_swap, order_swap, layer, 0.0)]
                    edited = index[(regime, realization, task_id, label_swap, order_swap, layer, coefficient)]
                    opposite = "verification" if regime == "obedience" else "obedience"
                    opposite_base = index[(opposite, realization, task_id, label_swap, order_swap, layer, 0.0)]
                    effect = edited["correct_logit_margin"] - base["correct_logit_margin"]
                    effects.append(effect)
                    gaps.append(
                        opposite_base["correct_logit_margin"] - base["correct_logit_margin"]
                    )
                    by_task[task_id].append(effect)
                    by_label_swap[label_swap].append(effect)
                    by_order_swap[order_swap].append(effect)
                task_effects = {task: mean(values) for task, values in by_task.items()}
                label_swap_effects = {
                    str(key): mean(values) for key, values in by_label_swap.items()
                }
                order_swap_effects = {
                    str(key): mean(values) for key, values in by_order_swap.items()
                }
                summaries.append({
                    "regime": regime,
                    "causal_test": (
                        "verification_to_obedience_rescue"
                        if expected_positive
                        else "obedience_to_verification_reverse_induction"
                    ),
                    "layer": layer,
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
                    "task_cluster_sign_p": exact_cluster_sign_p(task_effects.values()),
                    "task_effects_json": json.dumps(task_effects, sort_keys=True),
                    "label_swap_effects_json": json.dumps(
                        label_swap_effects, sort_keys=True
                    ),
                    "order_swap_effects_json": json.dumps(
                        order_swap_effects, sort_keys=True
                    ),
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
    layer_invariance_errors = [
        max(values) - min(values) for values in zero_by_item.values()
    ]
    audit = {
        "rows": len(rows),
        "unique_rows": len(index),
        "coefficient_zero_rows": len(zero_rows),
        "coefficient_zero_items": len(zero_by_item),
        "coefficient_zero_layer_invariance_max_abs": max(
            layer_invariance_errors, default=float("nan")
        ),
    }
    endpoint_layer = max(row["layer"] for row in rows)
    endpoint_rows = [
        row for row in rows
        if row["layer"] == endpoint_layer and row["patch_coefficient"] == 1.0
    ]
    endpoint_errors = []
    for row in endpoint_rows:
        opposite = "verification" if row["regime"] == "obedience" else "obedience"
        target_base = index[(
            opposite, row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"], endpoint_layer, 0.0,
        )]
        endpoint_errors.append(abs(
            row["correct_logit_margin"]
            - target_base["correct_logit_margin"]
        ))
    audit.update({
        "coefficient_one_endpoint_layer": endpoint_layer,
        "coefficient_one_endpoint_comparisons": len(endpoint_errors),
        "coefficient_one_endpoint_vs_cached_target_max_abs": max(
            endpoint_errors, default=float("nan")
        ),
    })
    if args.activation_metadata:
        full_rows = [
            json.loads(line)
            for line in args.activation_metadata.read_text().splitlines()
            if line.strip()
        ]
        full_index = {
            (
                row["regime"], row["history_depth"],
                row["history_realization"], row["task_id"],
                row["label_swap"], row["evidence_order_swap"],
            ): row
            for row in full_rows
        }
        full_forward_errors = []
        missing = []
        for row in zero_rows:
            key = (
                row["regime"], row["history_depth"],
                row["history_realization"], row["task_id"],
                row["label_swap"], row["evidence_order_swap"],
            )
            if key not in full_index:
                missing.append(key)
                continue
            full_forward_errors.append(abs(
                row["correct_logit_margin"]
                - full_index[key]["correct_logit_margin"]
            ))
        audit.update({
            "full_forward_comparisons": len(full_forward_errors),
            "full_forward_missing": len(missing),
            "coefficient_zero_vs_full_forward_max_abs": max(
                full_forward_errors, default=float("nan")
            ),
        })
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"audit": audit, "summaries": summaries}, indent=2))


if __name__ == "__main__":
    main()
