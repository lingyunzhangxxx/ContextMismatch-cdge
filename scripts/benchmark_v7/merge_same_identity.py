#!/usr/bin/env python3
"""Merge the three audited V3 supplement shards with the frozen full DGE run.

The merge is deliberately fail-closed.  It requires the exact frozen operator-dev
identity, successful per-method audits, and bit-for-bit identical baseline logits
for every method before computing any head-to-head statistic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_crossover import (
    build_interactions,
    cluster_bootstrap,
    summarize,
)
FULL_DGE_METHOD = "full_dge"
# Frozen same-identity contract.  Keep the merge path analysis-only: importing
# run_same_identity would also import torch and the NPU runtime even though the
# merge uses none of them.  These values are SHA-bound in the archived shards
# and are rechecked below for every method.
EXPECTED_ROWS = 3072
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
METHOD_SHARDS = {
    "operators": ("fixed_negative_vector", "symmetric_rank_one"),
    "experts": ("shared_single_expert", "positive_only_expert"),
    "routing": ("negative_only_expert", "dge_without_structural_routing"),
}
SUPPLEMENT_SHARDS = ("operators", "experts", "routing")
EXPECTED_METHODS = tuple(
    method for shard in SUPPLEMENT_SHARDS for method in METHOD_SHARDS[shard]
)
DISPLAY_NAMES = {
    "fixed_negative_vector": "Fixed negative vector",
    "symmetric_rank_one": "Symmetric rank-one",
    "shared_single_expert": "Shared single expert",
    "positive_only_expert": "Positive-only expert",
    "negative_only_expert": "Negative-only expert",
    "dge_without_structural_routing": "DGE without structural routing",
    FULL_DGE_METHOD: "Directional Governance Editing (DGE)",
}


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _key_sha(rows: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(str(row["job_key"]) for row in rows)) + "\n").encode("utf-8")
    ).hexdigest()


def _audit_rows(rows: list[dict], method: str) -> None:
    counts = Counter(str(row.get("job_key", "")) for row in rows)
    if len(rows) != EXPECTED_ROWS or len(counts) != EXPECTED_ROWS:
        raise ValueError(f"{method}: expected {EXPECTED_ROWS} unique rows")
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"{method}: duplicate job keys")
    if _key_sha(rows) != EXPECTED_KEY_SHA256:
        raise ValueError(f"{method}: frozen key SHA mismatch")
    for row in rows:
        if not _finite(row.get("baseline")) or not _finite(row.get("edited")):
            raise FloatingPointError(f"{method}: non-finite logits")


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


def _interaction_key(row: dict) -> tuple:
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


def _improvements(rows: list[dict]) -> list[dict]:
    baseline, baseline_problems = build_interactions(_view(rows, "baseline"))
    edited, edited_problems = build_interactions(_view(rows, "edited"))
    if baseline_problems or edited_problems:
        raise ValueError(
            f"incomplete interaction cells: baseline={baseline_problems[:3]}, "
            f"edited={edited_problems[:3]}"
        )
    edited_by_key = {_interaction_key(row): row for row in edited}
    result = []
    for base in baseline:
        key = _interaction_key(base)
        changed = edited_by_key.get(key)
        if changed is None:
            raise ValueError(f"missing edited interaction: {key}")
        baseline_gap = float(base["matched_minus_mismatched"])
        edited_gap = float(changed["matched_minus_mismatched"])
        reduction = baseline_gap - edited_gap
        result.append(
            {
                **{field: base[field] for field in (
                    "benchmark",
                    "item_id",
                    "partition",
                    "declared_role",
                    "history_style",
                    "history_depth",
                    "history_realization",
                    "label_swap",
                )},
                "baseline_gap": baseline_gap,
                "edited_gap": edited_gap,
                "match_advantage_reduction": float(base["match_advantage_interaction"])
                - float(changed["match_advantage_interaction"]),
                "matched_minus_mismatched_reduction": reduction,
                "normalized_gap_reduction": reduction / max(abs(baseline_gap), 1.0),
            }
        )
    return result


def _method_report(rows: list[dict], replicates: int, seed: int) -> tuple[dict, list[dict]]:
    improvements = _improvements(rows)
    return (
        {
            "display_name": DISPLAY_NAMES[str(rows[0].get("method", FULL_DGE_METHOD))],
            "rows": len(rows),
            "interactions": len(improvements),
            "match_advantage_reduction": summarize(improvements, "match_advantage_reduction"),
            "match_advantage_reduction_bootstrap": cluster_bootstrap(
                improvements, "match_advantage_reduction", replicates, seed
            ),
            "matched_minus_mismatched_reduction": summarize(
                improvements, "matched_minus_mismatched_reduction"
            ),
            "matched_minus_mismatched_reduction_bootstrap": cluster_bootstrap(
                improvements, "matched_minus_mismatched_reduction", replicates, seed + 1
            ),
            "normalized_gap_reduction": summarize(improvements, "normalized_gap_reduction"),
            "normalized_gap_reduction_bootstrap": cluster_bootstrap(
                improvements, "normalized_gap_reduction", replicates, seed + 2
            ),
        },
        improvements,
    )


def _paired_full_dge_comparison(
    full: list[dict], other: list[dict], replicates: int, seed: int
) -> dict:
    full_by_key = {_interaction_key(row): row for row in full}
    other_by_key = {_interaction_key(row): row for row in other}
    if set(full_by_key) != set(other_by_key):
        raise ValueError("full DGE/comparator interaction identity mismatch")
    paired = []
    for key in sorted(full_by_key):
        dge = full_by_key[key]
        comparator = other_by_key[key]
        paired.append(
            {
                "benchmark": dge["benchmark"],
                "item_id": dge["item_id"],
                "full_dge_minus_method_reduction": float(
                    dge["matched_minus_mismatched_reduction"]
                )
                - float(comparator["matched_minus_mismatched_reduction"]),
                "full_dge_minus_method_normalized_reduction": float(
                    dge["normalized_gap_reduction"]
                )
                - float(comparator["normalized_gap_reduction"]),
            }
        )
    return {
        "n": len(paired),
        "positive_favors_full_dge": True,
        "gap_reduction_difference": summarize(paired, "full_dge_minus_method_reduction"),
        "gap_reduction_difference_bootstrap": cluster_bootstrap(
            paired, "full_dge_minus_method_reduction", replicates, seed
        ),
        "normalized_reduction_difference": summarize(
            paired, "full_dge_minus_method_normalized_reduction"
        ),
        "normalized_reduction_difference_bootstrap": cluster_bootstrap(
            paired, "full_dge_minus_method_normalized_reduction", replicates, seed + 1
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, action="append", required=True)
    parser.add_argument("--full-dge-rows", type=Path, required=True)
    parser.add_argument("--full-dge-analysis", type=Path, required=True)
    parser.add_argument("--dge-systems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing merge output: {args.output}")
    if len(args.shard_root) != len(SUPPLEMENT_SHARDS):
        raise ValueError("exactly three shard roots are required")

    rows_by_method: dict[str, list[dict]] = {}
    systems_by_method: dict[str, dict] = {}
    shard_records = []
    observed_shards = set()
    for root in args.shard_root:
        audit_path = root / "shard_audit.json"
        systems_path = root / "systems_metrics.json"
        audit = json.loads(audit_path.read_text())
        systems = json.loads(systems_path.read_text())
        shard = str(audit.get("method_shard", ""))
        methods = tuple(audit.get("methods", []))
        if shard not in METHOD_SHARDS or methods != METHOD_SHARDS[shard]:
            raise ValueError(f"unexpected shard/method binding: {root}")
        if shard in observed_shards or audit.get("success") is not True:
            raise ValueError(f"duplicate or failed shard: {shard}")
        if audit.get("paired_baseline_logits_identical") is not True:
            raise ValueError(f"within-shard baseline mismatch: {shard}")
        observed_shards.add(shard)
        for method in methods:
            analysis_path = root / "analysis" / f"{method}.json"
            analysis = json.loads(analysis_path.read_text())
            if analysis.get("audit", {}).get("success") is not True:
                raise ValueError(f"failed method audit: {method}")
            rows = load_jsonl(root / "rows" / f"{method}.jsonl")
            _audit_rows(rows, method)
            if any(row.get("method") != method for row in rows):
                raise ValueError(f"row method mismatch: {method}")
            rows_by_method[method] = rows
            systems_by_method[method] = systems["methods_systems"][method]
        shard_records.append(
            {
                "method_shard": shard,
                "root": str(root),
                "audit_sha256": sha256_file(audit_path),
                "systems_metrics_sha256": sha256_file(systems_path),
            }
        )
    if observed_shards != set(SUPPLEMENT_SHARDS) or tuple(rows_by_method) != EXPECTED_METHODS:
        raise ValueError("supplement shard coverage is incomplete or reordered")

    full_rows = load_jsonl(args.full_dge_rows)
    _audit_rows(full_rows, FULL_DGE_METHOD)
    full_analysis = json.loads(args.full_dge_analysis.read_text())
    if full_analysis.get("audit", {}).get("success") is not True:
        raise ValueError("frozen full DGE analysis did not pass its audit")
    if any(row.get("method") != "DSGE-V3" for row in full_rows):
        raise ValueError("unexpected full DGE row identity")
    for row in full_rows:
        row["method"] = FULL_DGE_METHOD
    rows_by_method[FULL_DGE_METHOD] = full_rows

    canonical_method = FULL_DGE_METHOD
    canonical_baseline = {
        row["job_key"]: row["baseline"] for row in rows_by_method[canonical_method]
    }
    baseline_mismatches: dict[str, list[str]] = {}
    for method, rows in rows_by_method.items():
        observed = {row["job_key"]: row["baseline"] for row in rows}
        bad = sorted(key for key in canonical_baseline if observed.get(key) != canonical_baseline[key])
        if bad:
            baseline_mismatches[method] = bad[:20]
    if baseline_mismatches:
        raise ValueError(f"cross-shard baseline logits differ: {baseline_mismatches}")

    method_reports = {}
    improvement_rows = {}
    for index, method in enumerate((*EXPECTED_METHODS, FULL_DGE_METHOD)):
        report, improvements = _method_report(
            rows_by_method[method], args.bootstrap_replicates, 47000 + index * 20
        )
        method_reports[method] = report
        improvement_rows[method] = improvements
    paired = {
        method: _paired_full_dge_comparison(
            improvement_rows[FULL_DGE_METHOD],
            improvement_rows[method],
            args.bootstrap_replicates,
            49000 + index * 10,
        )
        for index, method in enumerate(EXPECTED_METHODS)
    }
    dge_systems = json.loads(args.dge_systems.read_text())
    value = {
        "schema_version": 1,
        "stage": "dge_same_identity_supplement_merged",
        "expected_rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "methods": list((*EXPECTED_METHODS, FULL_DGE_METHOD)),
        "display_names": DISPLAY_NAMES,
        "shards": sorted(shard_records, key=lambda item: item["method_shard"]),
        "full_dge_sources": {
            "rows_sha256": sha256_file(args.full_dge_rows),
            "analysis_sha256": sha256_file(args.full_dge_analysis),
            "systems_sha256": sha256_file(args.dge_systems),
        },
        "all_cross_shard_baseline_logits_identical": True,
        "baseline_mismatch_keys": {},
        "method_reports": method_reports,
        "full_dge_paired_comparisons": paired,
        "systems": {
            "same_identity_methods": systems_by_method,
            "full_dge": dge_systems,
        },
        "bootstrap_replicates": args.bootstrap_replicates,
        "success": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not _finite(value):
        raise FloatingPointError("merged report contains non-finite values")
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
