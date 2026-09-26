#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analyze_behavior import audit, cluster_bootstrap, paired_effects, summarize_effects
from .common import atomic_write_text, load_jsonl, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--criteria", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    criteria = json.loads(args.criteria.read_text())
    row_audit = audit(rows, environment.get("planned_rows"))
    mismatch = paired_effects(rows, "obedience", "verification")
    depth_reports = {}
    all_depths_pass = True
    for depth in criteria["required_depths"]:
        selected = [row for row in mismatch if int(row["history_depth"]) == int(depth)]
        summary = summarize_effects(selected)
        bootstrap = cluster_bootstrap(
            selected,
            replicates=int(criteria["bootstrap_replicates"]),
            seed=int(criteria["bootstrap_seed"]),
        )
        negative_benchmarks = sum(value < 0 for value in summary["benchmark_means"].values())

        def negative_strata(field: str) -> dict[str, float]:
            return {
                str(value): summarize_effects(
                    [row for row in selected if str(row[field]) == str(value)]
                )["equal_weight_benchmark_mean"]
                for value in sorted({row[field] for row in selected}, key=str)
            }

        label_swaps = negative_strata("label_swap")
        roles = negative_strata("declared_role")
        styles = negative_strata("history_style")
        checks = {
            "pair_count": summary["n_pairs"] == criteria["required_pairs_per_depth"],
            "bootstrap_ci_upper_negative": bootstrap["ci95"][1]
            < criteria["max_bootstrap_ci95_upper"],
            "benchmark_signs": negative_benchmarks
            >= criteria["min_negative_benchmark_means_per_depth"],
            "label_swap_signs": sum(value < 0 for value in label_swaps.values())
            >= criteria["required_negative_label_swap_strata_per_depth"],
            "role_signs": sum(value < 0 for value in roles.values())
            >= criteria["required_negative_role_strata_per_depth"],
            "history_style_signs": sum(value < 0 for value in styles.values())
            >= criteria["required_negative_history_style_strata_per_depth"],
        }
        depth_pass = all(checks.values())
        all_depths_pass = all_depths_pass and depth_pass
        depth_reports[str(depth)] = {
            "pass": depth_pass,
            "checks": checks,
            "summary": summary,
            "bootstrap": bootstrap,
            "negative_benchmark_count": negative_benchmarks,
            "label_swap_means": label_swaps,
            "role_means": roles,
            "history_style_means": styles,
        }

    audit_pass = bool(
        row_audit["success"]
        and row_audit["row_count"] == criteria["required_row_count"]
        and environment.get("batch_size") == 1
    )
    report = {
        "success": bool(audit_pass and all_depths_pass),
        "audit_pass": audit_pass,
        "row_audit": row_audit,
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "criteria_sha256": sha256_file(args.criteria),
        "depths": depth_reports,
        "full_behavior_allowed": bool(audit_pass and all_depths_pass),
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        args.allow_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
