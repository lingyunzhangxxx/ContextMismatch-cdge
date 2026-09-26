#!/usr/bin/env python3
"""Analyze counterbalanced governance-history length scans."""

from __future__ import annotations

import csv
import argparse
import itertools
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "results" / "length_scan.jsonl"
DEFAULT_OUT = ROOT / "results" / "length_scan"


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def exact_cluster_sign_p(cluster_differences):
    """Two-sided exact sign randomization with task domains as clusters."""
    values = list(cluster_differences)
    observed = abs(mean(values))
    null = [
        abs(mean([sign * value for sign, value in zip(signs, values)]))
        for signs in itertools.product((-1, 1), repeat=len(values))
    ]
    return sum(value >= observed - 1e-12 for value in null) / len(null)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    input_path = args.input
    out = args.output_dir
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    errors = [row for row in rows if "error" in row]
    valid = [row for row in rows if "error" not in row]
    out.mkdir(parents=True, exist_ok=True)
    if errors:
        (out / "errors.json").write_text(json.dumps(errors, indent=2) + "\n")
        raise SystemExit(f"{len(errors)} inference errors")

    cells = defaultdict(list)
    for row in valid:
        cells[(row["condition"], row["history_depth"])].append(row)
    summaries = []
    for (condition, depth), cell in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0])):
        margins = [row["correct_logit_margin"] for row in cell]
        summaries.append({
            "condition": condition,
            "history_depth": depth,
            "n": len(cell),
            "tasks": len({row["task_id"] for row in cell}),
            "accuracy": mean([row["correct"] for row in cell]),
            "unsupported_preference_rate": mean(
                [row["followed_unsupported_preference"] for row in cell]
            ),
            "correct_probability_binary": mean(
                [row["correct_probability_binary"] for row in cell]
            ),
            "correct_logit_margin_mean": mean(margins),
            "correct_logit_margin_sd": statistics.stdev(margins) if len(margins) > 1 else 0,
            "prompt_tokens_mean": mean([row["prompt_tokens"] for row in cell]),
            "prompt_tokens_min": min(row["prompt_tokens"] for row in cell),
            "prompt_tokens_max": max(row["prompt_tokens"] for row in cell),
        })
    write_csv(out / "summary.csv", summaries)

    index = {
        (
            row["condition"], row["history_depth"], row["task_id"],
            row["history_realization"], row["label_swap"], row["evidence_order_swap"],
        ): row
        for row in valid
    }
    contrasts = []
    available_conditions = {row["condition"] for row in valid}
    nonfresh_depths = sorted({row["history_depth"] for row in valid if row["history_depth"] > 0})
    for depth in nonfresh_depths:
        for left, right, name in (
            ("obedience", "verification", "mismatch_vs_match"),
            ("obedience", "obedience_reset", "reset_recovery"),
        ):
            if left not in available_conditions or right not in available_conditions:
                continue
            paired = defaultdict(list)
            for task_id in sorted({row["task_id"] for row in valid}):
                for realization in sorted({row["history_realization"] for row in valid}):
                    for label_swap in (0, 1):
                        for order_swap in (0, 1):
                            key_tail = (depth, task_id, realization, label_swap, order_swap)
                            a = index[(left,) + key_tail]
                            b = index[(right,) + key_tail]
                            paired[task_id].append(
                                a["correct_logit_margin"] - b["correct_logit_margin"]
                            )
            task_effects = {task: mean(values) for task, values in paired.items()}
            all_effects = [value for values in paired.values() for value in values]
            contrasts.append({
                "contrast": name,
                "left": left,
                "right": right,
                "history_depth": depth,
                "n_pairs": len(all_effects),
                "mean_margin_difference": mean(all_effects),
                "task_cluster_sign_p": exact_cluster_sign_p(task_effects.values()),
                "task_effects_json": json.dumps(task_effects, sort_keys=True),
            })
    write_csv(out / "contrasts.csv", contrasts)

    task_rows = []
    for (condition, depth), cell in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0])):
        by_task = defaultdict(list)
        for row in cell:
            by_task[row["task_id"]].append(row)
        for task_id, task_cell in sorted(by_task.items()):
            task_rows.append({
                "condition": condition,
                "history_depth": depth,
                "task_id": task_id,
                "n": len(task_cell),
                "accuracy": mean([row["correct"] for row in task_cell]),
                "correct_logit_margin_mean": mean(
                    [row["correct_logit_margin"] for row in task_cell]
                ),
            })
    write_csv(out / "task_summary.csv", task_rows)

    controls = []
    for (condition, depth), cell in sorted(cells.items(), key=lambda x: (x[0][1], x[0][0])):
        for control in ("label_swap", "evidence_order_swap"):
            group0 = [row["correct_logit_margin"] for row in cell if row[control] == 0]
            group1 = [row["correct_logit_margin"] for row in cell if row[control] == 1]
            controls.append({
                "condition": condition,
                "history_depth": depth,
                "control": control,
                "mean_at_0": mean(group0),
                "mean_at_1": mean(group1),
                "difference_1_minus_0": mean(group1) - mean(group0),
            })
    write_csv(out / "control_balance.csv", controls)

    audit = {
        "rows": len(valid),
        "errors": len(errors),
        "tasks": sorted({row["task_id"] for row in valid}),
        "conditions": sorted({row["condition"] for row in valid}),
        "history_styles": sorted({row.get("history_style", "natural") for row in valid}),
        "history_depths": sorted({row["history_depth"] for row in valid}),
        "models": sorted({row["model"] for row in valid}),
        "total_prompt_tokens": sum(row["prompt_tokens"] for row in valid),
        "max_prompt_tokens": max(row["prompt_tokens"] for row in valid),
        "min_prompt_tokens": min(row["prompt_tokens"] for row in valid),
        "finite_margins": all(math.isfinite(row["correct_logit_margin"]) for row in valid),
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    print("\nLength cells:")
    for row in summaries:
        print(row)
    print("\nPaired contrasts (negative means the left condition is worse):")
    for row in contrasts:
        print(row)


if __name__ == "__main__":
    main()
