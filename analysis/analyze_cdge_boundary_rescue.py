#!/usr/bin/env python3
"""Describe where C-DGE changes decisions on the confirmatory final split.

This analysis separates mismatch-induced flips from matched-condition failures.
Frozen margin thresholds come from the preregistered prospective boundary
protocol; applying them to the C-DGE final rows is descriptive, not a new
confirmatory test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


PAIR_FIELDS = (
    "benchmark",
    "item_id",
    "partition",
    "declared_role",
    "history_style",
    "history_depth",
    "history_realization",
    "label_swap",
    "task_requirement",
)
ORIENTATION = {
    "independent_verification": ("verification", "obedience"),
    "delegated_choice": ("obedience", "verification"),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def is_correct(values: dict) -> bool:
    """Use the paper's strict task-success rule; zero-margin ties do not pass."""
    return float(values["task_aligned_margin"]) > 0.0


def classify_band(matched_correct: bool, margin: float, thresholds: dict) -> str:
    if not matched_correct:
        return "matched_failure"
    if margin <= float(thresholds["boundary_max"]):
        return "boundary"
    if margin >= float(thresholds["robust_min"]):
        return "robust"
    return "middle"


def summarize(pairs: list[dict]) -> dict:
    count = len(pairs)
    matched_correct = sum(row["matched_correct"] for row in pairs)
    raw_correct = sum(row["raw_mismatch_correct"] for row in pairs)
    edited_correct = sum(row["edited_mismatch_correct"] for row in pairs)
    flips = [
        row
        for row in pairs
        if row["matched_correct"] and not row["raw_mismatch_correct"]
    ]
    rescued = sum(row["edited_mismatch_correct"] for row in flips)
    harmed = sum(
        row["raw_mismatch_correct"] and not row["edited_mismatch_correct"]
        for row in pairs
    )
    return {
        "pairs": count,
        "matched_correct": matched_correct,
        "matched_accuracy": matched_correct / count if count else None,
        "raw_mismatch_correct": raw_correct,
        "raw_mismatch_accuracy": raw_correct / count if count else None,
        "edited_mismatch_correct": edited_correct,
        "edited_mismatch_accuracy": edited_correct / count if count else None,
        "accuracy_gain_pp": 100.0 * (edited_correct - raw_correct) / count if count else None,
        "mismatch_induced_flips": len(flips),
        "rescued_flips": rescued,
        "conditional_rescue_rate": rescued / len(flips) if flips else None,
        "raw_correct_harmed": harmed,
        "net_correct_decisions": edited_correct - raw_correct,
    }


def cluster_bootstrap_rescue(
    pairs: list[dict], replicates: int, seed: int
) -> dict | None:
    target = [
        row
        for row in pairs
        if row["matched_correct"] and not row["raw_mismatch_correct"]
    ]
    if not target:
        return None
    by_item: dict[str, list[dict]] = defaultdict(list)
    for row in target:
        by_item[row["item_id"]].append(row)
    items = sorted(by_item)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [
            row
            for item in (rng.choice(items) for _ in items)
            for row in by_item[item]
        ]
        draws.append(
            sum(row["edited_mismatch_correct"] for row in sampled) / len(sampled)
        )
    estimate = sum(row["edited_mismatch_correct"] for row in target) / len(target)
    return {
        "estimate": estimate,
        "ci95": [percentile(draws, 0.025), percentile(draws, 0.975)],
        "replicates": replicates,
        "seed": seed,
        "cluster": "item_id",
    }


def analyze(rows: list[dict], boundary_report: dict, replicates: int) -> dict:
    keys = [str(row["job_key"]) for row in rows]
    if len(rows) % 4 or len(set(keys)) != len(rows):
        raise ValueError("expected unique rows forming complete four-cell pairs")
    numeric_fields = ("logit_a", "logit_b", "task_aligned_margin")
    if not all(
        math.isfinite(float(row[view][field]))
        for row in rows
        for view in ("baseline", "edited")
        for field in numeric_fields
    ):
        raise ValueError("non-finite final-test logits or margins")

    grouped: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in PAIR_FIELDS)
        history = row["history_condition"]
        if history in grouped[key]:
            raise ValueError(f"duplicate history cell: {key}:{history}")
        grouped[key][history] = row

    all_pairs = []
    requirements = {}
    for requirement, (matched_history, mismatched_history) in ORIENTATION.items():
        thresholds = boundary_report["requirements"][requirement]["frozen_thresholds"]
        pairs = []
        for key, conditions in sorted(grouped.items()):
            if key[-1] != requirement:
                continue
            if set(conditions) != {"obedience", "verification"}:
                raise ValueError(f"incomplete history pair: {key}")
            matched = conditions[matched_history]
            mismatched = conditions[mismatched_history]
            for field in ("candidate_payload_sha256", "task_correct_label"):
                if matched[field] != mismatched[field]:
                    raise ValueError(f"paired field changed: {field}:{key}")
            matched_correct = is_correct(matched["baseline"])
            raw_correct = is_correct(mismatched["baseline"])
            edited_correct = is_correct(mismatched["edited"])
            matched_margin = float(matched["baseline"]["task_aligned_margin"])
            pair = {
                "benchmark": matched["benchmark"],
                "item_id": matched["item_id"],
                "requirement": requirement,
                "matched_margin": matched_margin,
                "raw_mismatch_margin": float(mismatched["baseline"]["task_aligned_margin"]),
                "edited_mismatch_margin": float(mismatched["edited"]["task_aligned_margin"]),
                "matched_correct": matched_correct,
                "raw_mismatch_correct": raw_correct,
                "edited_mismatch_correct": edited_correct,
                "band": classify_band(matched_correct, matched_margin, thresholds),
            }
            pairs.append(pair)
            all_pairs.append(pair)
        expected_pairs = len(rows) // 4
        if len(pairs) != expected_pairs:
            raise ValueError(
                f"{requirement}: expected {expected_pairs:,} pairs, got {len(pairs):,}"
            )
        report = {"overall": summarize(pairs), "bands": {}}
        report["overall"]["conditional_rescue_bootstrap"] = cluster_bootstrap_rescue(
            pairs, replicates, 64201 if requirement == "independent_verification" else 64202
        )
        for band in ("matched_failure", "boundary", "middle", "robust"):
            selected = [row for row in pairs if row["band"] == band]
            report["bands"][band] = summarize(selected)
            report["bands"][band]["conditional_rescue_bootstrap"] = (
                cluster_bootstrap_rescue(
                    selected,
                    replicates,
                    64300
                    + list(("matched_failure", "boundary", "middle", "robust")).index(band)
                    + (0 if requirement == "independent_verification" else 10),
                )
            )
        requirements[requirement] = report

    overall = summarize(all_pairs)
    overall["conditional_rescue_bootstrap"] = cluster_bootstrap_rescue(
        all_pairs, replicates, 64200
    )
    return {
        "schema_version": 1,
        "analysis_status": "posthoc_descriptive_boundary_rescue_analysis",
        "method": sorted({str(row.get("method")) for row in rows}),
        "evaluation_split": sorted(
            {str(row.get("evaluation_split", row.get("partition"))) for row in rows}
        ),
        "estimand": "conditional rescue among matched-correct, raw-mismatch-wrong pairs",
        "decision_rule": "task-aligned margin > 0; zero-margin ties count as failures",
        "frozen_band_source": "prospective boundary report; thresholds were frozen from discovery",
        "interpretation": {
            "margin_band_analysis_is_descriptive": True,
            "matched_failure_is_floor_population": True,
            "conditional_rescue_is_not_overall_accuracy_gain": True,
        },
        "audit": {
            "rows": len(rows),
            "unique_job_keys": len(set(keys)),
            "key_sha256": hashlib.sha256(
                (("\n".join(sorted(keys))) + "\n").encode("utf-8")
            ).hexdigest(),
            "all_finite": True,
        },
        "overall": overall,
        "requirements": requirements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--boundary-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    boundary_report = json.loads(args.boundary_report.read_text())
    result = analyze(rows, boundary_report, args.bootstrap_replicates)
    result["input_sha256"] = sha256_file(args.input)
    result["boundary_report_sha256"] = sha256_file(args.boundary_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
