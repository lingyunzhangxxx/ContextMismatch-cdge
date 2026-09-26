#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file


VERIFY = "independent_verification"
DEFER = "delegated_choice"


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
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _base_key(row: dict) -> tuple:
    return (
        row["benchmark"],
        row["item_id"],
        row["partition"],
        row["declared_role"],
        row["history_style"],
        row["history_depth"],
        row["history_realization"],
        row["label_swap"],
    )


def build_interactions(rows: list[dict]) -> tuple[list[dict], list[str]]:
    lookup = {(_base_key(row), row["history_condition"], row["task_requirement"]): row for row in rows}
    problems = []
    interactions = []
    for key in sorted({_base_key(row) for row in rows}):
        cells = {}
        for history in ("verification", "obedience"):
            for requirement in (VERIFY, DEFER):
                cell = lookup.get((key, history, requirement))
                if cell is None:
                    problems.append(f"missing cell {key}:{history}:{requirement}")
                else:
                    cells[(history, requirement)] = cell
        if len(cells) != 4:
            continue
        payloads = {cell["candidate_payload_sha256"] for cell in cells.values()}
        user_labels = {cell["user_selected_label"] for cell in cells.values()}
        if len(payloads) != 1:
            problems.append(f"candidate payload mismatch: {key}")
        if len(user_labels) != 1:
            problems.append(f"user-label mismatch: {key}")
        verify_suffixes = {
            cells[(history, VERIFY)]["suffix_token_sha256_int32_le"]
            for history in ("verification", "obedience")
        }
        defer_suffixes = {
            cells[(history, DEFER)]["suffix_token_sha256_int32_le"]
            for history in ("verification", "obedience")
        }
        if len(verify_suffixes) != 1 or len(defer_suffixes) != 1:
            problems.append(f"history-paired suffix mismatch: {key}")
        aligned_verify_effect = (
            float(cells[("obedience", VERIFY)]["task_aligned_margin"])
            - float(cells[("verification", VERIFY)]["task_aligned_margin"])
        )
        aligned_defer_effect = (
            float(cells[("obedience", DEFER)]["task_aligned_margin"])
            - float(cells[("verification", DEFER)]["task_aligned_margin"])
        )
        raw_verify_effect = (
            float(cells[("obedience", VERIFY)]["user_choice_margin"])
            - float(cells[("verification", VERIFY)]["user_choice_margin"])
        )
        raw_defer_effect = (
            float(cells[("obedience", DEFER)]["user_choice_margin"])
            - float(cells[("verification", DEFER)]["user_choice_margin"])
        )
        match_advantage = aligned_defer_effect - aligned_verify_effect
        matched_minus_mismatched = 0.5 * (
            float(cells[("verification", VERIFY)]["task_aligned_margin"])
            + float(cells[("obedience", DEFER)]["task_aligned_margin"])
            - float(cells[("obedience", VERIFY)]["task_aligned_margin"])
            - float(cells[("verification", DEFER)]["task_aligned_margin"])
        )
        interactions.append(
            {
                "benchmark": key[0],
                "item_id": key[1],
                "partition": key[2],
                "declared_role": key[3],
                "history_style": key[4],
                "history_depth": key[5],
                "history_realization": key[6],
                "label_swap": key[7],
                "verification_history_effect": aligned_verify_effect,
                "deference_history_effect": aligned_defer_effect,
                "match_advantage_interaction": match_advantage,
                "matched_minus_mismatched": matched_minus_mismatched,
                "raw_user_carryover_verification_task": raw_verify_effect,
                "raw_user_carryover_deference_task": raw_defer_effect,
                "task_modulation_of_raw_carryover": raw_defer_effect - raw_verify_effect,
                "both_directional_predictions_met": (
                    aligned_verify_effect < 0 and aligned_defer_effect > 0
                ),
            }
        )
    return interactions, problems


def summarize(rows: list[dict], field: str) -> dict:
    if not rows:
        return {"n": 0, "mean": None, "equal_weight_benchmark_mean": None, "benchmark_means": {}}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(float(row[field]))
    benchmark_means = {key: mean(values) for key, values in sorted(grouped.items())}
    return {
        "n": len(rows),
        "mean": mean([float(row[field]) for row in rows]),
        "equal_weight_benchmark_mean": mean(list(benchmark_means.values())),
        "benchmark_means": benchmark_means,
    }


def cluster_bootstrap(
    rows: list[dict], field: str, replicates: int, seed: int
) -> dict:
    by_benchmark_item: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_benchmark_item[row["benchmark"]][row["item_id"]].append(float(row[field]))
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        benchmark_values = []
        for benchmark in sorted(by_benchmark_item):
            items = sorted(by_benchmark_item[benchmark])
            sampled = [items[rng.randrange(len(items))] for _ in items]
            benchmark_values.append(
                mean(
                    [
                        value
                        for item in sampled
                        for value in by_benchmark_item[benchmark][item]
                    ]
                )
            )
        draws.append(mean(benchmark_values))
    return {
        "replicates": replicates,
        "seed": seed,
        "estimate": mean(draws),
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
    }


def audit(rows: list[dict], environment: dict, interaction_problems: list[str]) -> dict:
    key_counts = Counter(row.get("job_key") for row in rows)
    duplicates = [key for key, count in key_counts.items() if count != 1]
    numeric = ("logit_a", "logit_b", "task_aligned_margin", "factual_margin", "user_choice_margin")
    nonfinite = [
        row.get("job_key")
        for row in rows
        if not all(math.isfinite(float(row[name])) for name in numeric)
    ]
    observed_key_hash = hashlib.sha256(
        (("\n".join(sorted(str(row["job_key"]) for row in rows))) + "\n").encode("utf-8")
    ).hexdigest()
    planned = int(environment.get("planned_rows", -1))
    expected_hash = environment.get("expected_key_sha256")
    success = (
        len(rows) == planned
        and not duplicates
        and not nonfinite
        and not interaction_problems
        and observed_key_hash == expected_hash
    )
    return {
        "row_count": len(rows),
        "planned_rows": planned,
        "unique_job_keys": len(key_counts),
        "duplicate_job_keys": duplicates[:20],
        "nonfinite_job_keys": nonfinite[:20],
        "interaction_problems": interaction_problems[:20],
        "observed_key_sha256": observed_key_hash,
        "expected_key_sha256": expected_hash,
        "success": success,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.contract.read_text())
    interactions, problems = build_interactions(rows)
    replicates = int(
        args.bootstrap_replicates
        if args.bootstrap_replicates is not None
        else contract["analysis"]["cluster_bootstrap_replicates"]
    )
    seed = int(contract["analysis"]["cluster_bootstrap_seed"])
    fields = (
        "verification_history_effect",
        "deference_history_effect",
        "match_advantage_interaction",
        "matched_minus_mismatched",
        "raw_user_carryover_verification_task",
        "raw_user_carryover_deference_task",
        "task_modulation_of_raw_carryover",
    )
    condition_summary = {}
    for history in ("verification", "obedience"):
        for requirement in (VERIFY, DEFER):
            selected = [
                row
                for row in rows
                if row["history_condition"] == history
                and row["task_requirement"] == requirement
            ]
            condition_summary[f"{history}__{requirement}"] = {
                "n": len(selected),
                "mean_task_aligned_margin": mean(
                    [float(row["task_aligned_margin"]) for row in selected]
                ),
                "task_aligned_accuracy": mean(
                    [float(row["task_aligned_correct"]) for row in selected]
                ),
                "factual_accuracy": mean([float(row["factual_correct"]) for row in selected]),
                "followed_user_selection": mean(
                    [float(row["followed_user_selection"]) for row in selected]
                ),
            }
    report = {
        "schema_version": 1,
        "input": str(args.input),
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "contract_sha256": sha256_file(args.contract),
        "audit": audit(rows, environment, problems),
        "interaction_pairs": len(interactions),
        "condition_summary": condition_summary,
        "estimands": {field: summarize(interactions, field) for field in fields},
        "cluster_bootstrap": {
            field: cluster_bootstrap(interactions, field, replicates, seed + index)
            for index, field in enumerate(fields)
        }
        if interactions
        else {},
        "fraction_both_directional_predictions_met": mean(
            [float(row["both_directional_predictions_met"]) for row in interactions]
        ),
        "strata": {
            field: {
                value: summarize(
                    [row for row in interactions if str(row[field]) == value],
                    "match_advantage_interaction",
                )
                for value in sorted({str(row[field]) for row in interactions})
            }
            for field in ("benchmark", "declared_role", "history_style", "label_swap", "partition")
        },
        "interpretation_boundary": {
            "task_aligned_interaction_is_not_raw_state_interaction": True,
            "raw_user_choice_carryover_reported_separately": True,
            "deference_task_does_not_relabel_factual_truth": True,
            "production_rollout_approved": False,
        },
    }
    atomic_write_text(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        args.allow_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
