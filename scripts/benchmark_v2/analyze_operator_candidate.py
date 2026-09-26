#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_crossover import (
    DEFER,
    VERIFY,
    build_interactions,
    cluster_bootstrap,
    mean,
    percentile,
    summarize,
)


def _view(rows: list[dict], field: str) -> list[dict]:
    result = []
    for row in rows:
        values = row[field]
        result.append(
            {
                **{key: value for key, value in row.items() if key not in {"baseline", "edited"}},
                "task_aligned_margin": values["task_aligned_margin"],
                "factual_margin": values["factual_margin"],
                "user_choice_margin": values["user_choice_margin"],
            }
        )
    return result


def _cell_delta(rows: list[dict], history: str, requirement: str) -> list[dict]:
    return [
        {
            "benchmark": row["benchmark"],
            "item_id": row["item_id"],
            "effect": float(row["delta_task_aligned_margin"]),
            "binary_kl": float(row["task_aligned_binary_kl"]),
        }
        for row in rows
        if row["history_condition"] == history and row["task_requirement"] == requirement
    ]


def _summarize_effect(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "mean": None, "equal_weight_benchmark_mean": None, "benchmark_means": {}}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["benchmark"]].append(float(row["effect"]))
    benchmark_means = {key: mean(values) for key, values in sorted(grouped.items())}
    return {
        "n": len(rows),
        "mean": mean([float(row["effect"]) for row in rows]),
        "equal_weight_benchmark_mean": mean(list(benchmark_means.values())),
        "benchmark_means": benchmark_means,
    }


def _bootstrap_upper(rows: list[dict], field: str, replicates: int, seed: int) -> dict:
    by_benchmark_item = defaultdict(lambda: defaultdict(list))
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
        "estimate": mean([float(row[field]) for row in rows]),
        "upper_one_sided_95": percentile(draws, 0.95),
        "replicates": replicates,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    identity = json.loads(args.identity_report.read_text())
    keys = Counter(row["job_key"] for row in rows)
    observed_key_hash = hashlib.sha256(
        (("\n".join(sorted(keys))) + "\n").encode("utf-8")
    ).hexdigest()
    nonfinite = []
    for row in rows:
        values = [
            *row["baseline"].values(),
            *row["edited"].values(),
            row["delta_task_aligned_margin"],
            row["delta_factual_margin"],
            row["delta_user_choice_margin"],
            row["task_aligned_binary_kl"],
        ]
        if not all(math.isfinite(float(value)) for value in values):
            nonfinite.append(row["job_key"])
    baseline_interactions, baseline_problems = build_interactions(_view(rows, "baseline"))
    edited_interactions, edited_problems = build_interactions(_view(rows, "edited"))
    edited_by_key = {
        (
            row["benchmark"],
            row["item_id"],
            row["partition"],
            row["declared_role"],
            row["history_style"],
            row["history_depth"],
            row["history_realization"],
            row["label_swap"],
        ): row
        for row in edited_interactions
    }
    improvements = []
    for baseline in baseline_interactions:
        key = (
            baseline["benchmark"],
            baseline["item_id"],
            baseline["partition"],
            baseline["declared_role"],
            baseline["history_style"],
            baseline["history_depth"],
            baseline["history_realization"],
            baseline["label_swap"],
        )
        if key not in edited_by_key:
            continue
        edited = edited_by_key[key]
        improvements.append(
            {
                **{field: baseline[field] for field in (
                    "benchmark", "item_id", "partition", "declared_role", "history_style",
                    "history_depth", "history_realization", "label_swap"
                )},
                "baseline_match_advantage": baseline["match_advantage_interaction"],
                "edited_match_advantage": edited["match_advantage_interaction"],
                "match_advantage_reduction": baseline["match_advantage_interaction"]
                - edited["match_advantage_interaction"],
                "matched_minus_mismatched_reduction": baseline["matched_minus_mismatched"]
                - edited["matched_minus_mismatched"],
                "normalized_gap_reduction": (
                    baseline["matched_minus_mismatched"]
                    - edited["matched_minus_mismatched"]
                ) / max(abs(baseline["matched_minus_mismatched"]), 1.0),
                "edited_minus_baseline_gap_change": edited["matched_minus_mismatched"]
                - baseline["matched_minus_mismatched"],
                "verification_history_effect_change": edited["verification_history_effect"]
                - baseline["verification_history_effect"],
                "deference_history_effect_change": edited["deference_history_effect"]
                - baseline["deference_history_effect"],
            }
        )
    mismatch_verify = _cell_delta(rows, "obedience", VERIFY)
    mismatch_defer = _cell_delta(rows, "verification", DEFER)
    matched_verify = _cell_delta(rows, "verification", VERIFY)
    matched_defer = _cell_delta(rows, "obedience", DEFER)
    audit_success = (
        len(rows) == int(environment.get("planned_rows", -1))
        and len(keys) == len(rows)
        and observed_key_hash == environment.get("expected_key_sha256")
        and not nonfinite
        and not baseline_problems
        and not edited_problems
        and identity.get("success") is True
        and float(identity.get("max_error", math.inf)) == 0.0
    )
    per_row_intervention = []
    per_site_intervention = defaultdict(list)
    for row in rows:
        site_values = []
        for site, diagnostics in row["intervention_diagnostics"].items():
            value = float(diagnostics["mean_relative_intervention_norm"])
            per_site_intervention[site].append(value)
            site_values.append(value)
        per_row_intervention.append(sum(site_values))
    report = {
        "schema_version": 1,
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "identity_report_sha256": sha256_file(args.identity_report),
        "candidate_id": environment["candidate_id"],
        "evaluation_split": environment["stage"],
        "audit": {
            "row_count": len(rows),
            "planned_rows": environment.get("planned_rows"),
            "unique_job_keys": len(keys),
            "observed_key_sha256": observed_key_hash,
            "expected_key_sha256": environment.get("expected_key_sha256"),
            "nonfinite_job_keys": nonfinite[:20],
            "baseline_interaction_problems": baseline_problems[:20],
            "edited_interaction_problems": edited_problems[:20],
            "zero_gate_max_error": identity.get("max_error"),
            "success": audit_success,
        },
        "baseline_match_advantage": summarize(
            baseline_interactions, "match_advantage_interaction"
        ),
        "edited_match_advantage": summarize(edited_interactions, "match_advantage_interaction"),
        "match_advantage_reduction": summarize(
            improvements, "match_advantage_reduction"
        ),
        "match_advantage_reduction_bootstrap": cluster_bootstrap(
            improvements,
            "match_advantage_reduction",
            args.bootstrap_replicates,
            42017,
        )
        if improvements
        else None,
        "matched_minus_mismatched_reduction": summarize(
            improvements, "matched_minus_mismatched_reduction"
        ),
        "matched_minus_mismatched_reduction_bootstrap": cluster_bootstrap(
            improvements,
            "matched_minus_mismatched_reduction",
            args.bootstrap_replicates,
            42019,
        )
        if improvements
        else None,
        "normalized_gap_reduction": summarize(improvements, "normalized_gap_reduction"),
        "normalized_gap_reduction_bootstrap": cluster_bootstrap(
            improvements,
            "normalized_gap_reduction",
            args.bootstrap_replicates,
            42023,
        )
        if improvements
        else None,
        "directional_recovery": {
            "obedience_history_on_verification_task": _summarize_effect(mismatch_verify),
            "verification_history_on_delegated_task": _summarize_effect(mismatch_defer),
        },
        "matched_cell_collateral": {
            "verification_history_on_verification_task": {
                **_summarize_effect(matched_verify),
                "margin_loss_upper": _bootstrap_upper(
                    [{**row, "margin_loss": -float(row["effect"])} for row in matched_verify],
                    "margin_loss",
                    args.bootstrap_replicates,
                    43001,
                ),
                "binary_kl_upper": _bootstrap_upper(
                    matched_verify, "binary_kl", args.bootstrap_replicates, 43003
                ),
            },
            "obedience_history_on_delegated_task": {
                **_summarize_effect(matched_defer),
                "margin_loss_upper": _bootstrap_upper(
                    [{**row, "margin_loss": -float(row["effect"])} for row in matched_defer],
                    "margin_loss",
                    args.bootstrap_replicates,
                    43007,
                ),
                "binary_kl_upper": _bootstrap_upper(
                    matched_defer, "binary_kl", args.bootstrap_replicates, 43009
                ),
            },
        },
        "gap_reduction_by_label_swap": {
            str(value): summarize(
                [row for row in improvements if int(row["label_swap"]) == value],
                "matched_minus_mismatched_reduction",
            )
            for value in (0, 1)
        },
        "gap_reduction_by_benchmark": {
            value: summarize(
                [row for row in improvements if row["benchmark"] == value],
                "matched_minus_mismatched_reduction",
            )
            for value in sorted({row["benchmark"] for row in improvements})
        },
        "intervention_norm": {
            "mean_sum_site_relative_norm": mean(per_row_intervention),
            "max_sum_site_relative_norm": max(per_row_intervention),
            "mean_by_site": {
                site: mean(values) for site, values in sorted(per_site_intervention.items())
            },
        },
        "admissibility": {
            "behavioral_dev_half_complete": audit_success,
            "protected_controls_complete": False,
            "candidate_may_be_locked": False,
            "reason": "Protected authority, reset, fresh, and factual-memory behavior must be evaluated before selection.",
        },
        "final_test_open": environment.get("stage") == "operator_final_test",
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        args.allow_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not audit_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
