#!/usr/bin/env python3
"""Audit and summarize the frozen C-DGE V4.1 performance benchmark."""

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
from scripts.benchmark_v10.cdge_performance_design import (
    EXPECTED_CELLS,
    EXPECTED_CELL_KEY_SHA256,
    EXPECTED_MEASUREMENTS,
    METHODS,
    TIMED_REPEATS,
    WARMUP_REPEATS,
    measurement_key,
)


METRICS = (
    "paired_latency_seconds",
    "examples_per_second",
    "suffix_tokens_per_second",
    "peak_npu_memory_bytes",
    "incremental_peak_npu_memory_bytes",
)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty quantile")
    index = (len(ordered) - 1) * probability
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": _quantile(values, 0.95),
        "minimum": min(values),
        "maximum": max(values),
    }


def _cluster_bootstrap(cell_values: dict[str, list[float]], replicates: int, seed: int) -> dict:
    cells = sorted(cell_values)
    estimates = [statistics.fmean(cell_values[cell]) for cell in cells]
    estimate = statistics.fmean(estimates)
    generator = random.Random(seed)
    samples = []
    for _ in range(replicates):
        selected = [cells[generator.randrange(len(cells))] for _ in cells]
        samples.append(statistics.fmean(statistics.fmean(cell_values[cell]) for cell in selected))
    return {
        "estimate": estimate,
        "cluster_unit": "performance_cell",
        "clusters": len(cells),
        "bootstrap_replicates": replicates,
        "two_sided_95": [_quantile(samples, 0.025), _quantile(samples, 0.975)],
    }


def analyze(rows: list[dict], environment: dict, contract: dict, *, replicates: int) -> dict:
    performance = contract.get("performance", {})
    expected_contract = {
        "stage": "governance_composite_performance",
        "methods": list(METHODS),
        "expected_cells": EXPECTED_CELLS,
        "batch_size": 1,
        "warmup_repeats": WARMUP_REPEATS,
        "timed_repeats": TIMED_REPEATS,
        "attn_implementation": "eager",
        "peak_memory_reset_after_model_load_required": True,
        "finite_measurements_required": True,
    }
    for field, expected in expected_contract.items():
        if performance.get(field) != expected:
            raise ValueError(f"performance contract mismatch: {field}")
    for field, expected in {
        "stage": "governance_composite_performance",
        "method": "C-DGE-V4.1",
        "methods": list(METHODS),
        "expected_cells": EXPECTED_CELLS,
        "expected_cell_key_sha256": EXPECTED_CELL_KEY_SHA256,
        "expected_measurements": EXPECTED_MEASUREMENTS,
        "batch_size": 1,
        "warmup_repeats": WARMUP_REPEATS,
        "timed_repeats": TIMED_REPEATS,
        "attn_implementation": "eager",
        "peak_memory_reset_after_model_load": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if environment.get(field) != expected:
            raise ValueError(f"performance environment mismatch: {field}")

    keys = Counter(row.get("measurement_key") for row in rows)
    cells = sorted({str(row.get("cell_key")) for row in rows})
    observed_cell_sha = hashlib.sha256((("\n".join(cells)) + "\n").encode()).hexdigest()
    nonfinite = []
    invalid = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    paired: dict[tuple[str, int], dict[str, dict]] = defaultdict(dict)
    for row in rows:
        method = row.get("method")
        repeat = row.get("repeat")
        cell = row.get("cell_key")
        try:
            expected_key = measurement_key(str(cell), str(method), int(repeat))
        except (TypeError, ValueError):
            invalid.append(str(row.get("measurement_key")))
            continue
        if row.get("measurement_key") != expected_key:
            invalid.append(str(row.get("measurement_key")))
        numeric = [float(row[name]) for name in METRICS]
        numeric.extend(float(value) for value in row.get("selected_label_logits", {}).values())
        if not all(math.isfinite(value) for value in numeric):
            nonfinite.append(expected_key)
        if float(row["paired_latency_seconds"]) <= 0 or int(row["peak_npu_memory_bytes"]) <= 0:
            invalid.append(expected_key)
        if not math.isclose(float(row["examples_per_second"]), 1.0 / float(row["paired_latency_seconds"]), rel_tol=1e-12):
            invalid.append(expected_key)
        if not math.isclose(float(row["suffix_tokens_per_second"]), int(row["suffix_tokens"]) / float(row["paired_latency_seconds"]), rel_tol=1e-12):
            invalid.append(expected_key)
        grouped[(str(cell), str(method))].append(row)
        paired[(str(cell), int(repeat))][str(method)] = row
    expected_group_counts = all(
        len(grouped.get((cell, method), [])) == TIMED_REPEATS
        for cell in cells for method in METHODS
    )
    token_identity = True
    for cell in cells:
        current = [row for row in rows if row.get("cell_key") == cell]
        signatures = {
            (
                row.get("operator_dev_job_key"), row.get("prefix_tokens"), row.get("suffix_tokens"),
                row.get("prefix_token_sha256_int32_le"), row.get("suffix_token_sha256_int32_le"),
            )
            for row in current
        }
        token_identity = token_identity and len(signatures) == 1
    pair_complete = len(paired) == EXPECTED_CELLS * TIMED_REPEATS and all(
        set(value) == set(METHODS) for value in paired.values()
    )
    audit_success = (
        len(rows) == EXPECTED_MEASUREMENTS
        and len(keys) == EXPECTED_MEASUREMENTS
        and all(count == 1 for count in keys.values())
        and len(cells) == EXPECTED_CELLS
        and observed_cell_sha == EXPECTED_CELL_KEY_SHA256
        and not nonfinite
        and not invalid
        and expected_group_counts
        and token_identity
        and pair_complete
    )

    method_reports = {}
    for method in METHODS:
        selected = [row for row in rows if row.get("method") == method]
        method_reports[method] = {
            "measurements": len(selected),
            **{metric: _summary([float(row[metric]) for row in selected]) for metric in METRICS},
        }
    paired_reports = {}
    for method_index, method in enumerate(METHODS[1:]):
        latency_ratio: dict[str, list[float]] = defaultdict(list)
        throughput_ratio: dict[str, list[float]] = defaultdict(list)
        memory_delta: dict[str, list[float]] = defaultdict(list)
        for (cell, _repeat), values in paired.items():
            if set(values) != set(METHODS):
                continue
            baseline = values["baseline"]
            current = values[method]
            latency_ratio[cell].append(
                float(current["paired_latency_seconds"]) / float(baseline["paired_latency_seconds"])
            )
            throughput_ratio[cell].append(
                float(current["examples_per_second"]) / float(baseline["examples_per_second"])
            )
            memory_delta[cell].append(
                float(current["peak_npu_memory_bytes"]) - float(baseline["peak_npu_memory_bytes"])
            )
        latency = _cluster_bootstrap(latency_ratio, replicates, 93001 + method_index * 10)
        throughput = _cluster_bootstrap(throughput_ratio, replicates, 93002 + method_index * 10)
        memory = _cluster_bootstrap(memory_delta, replicates, 93003 + method_index * 10)
        paired_reports[method] = {
            "latency_ratio_to_baseline": latency,
            "latency_overhead_percent": {
                **latency,
                "estimate": 100.0 * (latency["estimate"] - 1.0),
                "two_sided_95": [100.0 * (value - 1.0) for value in latency["two_sided_95"]],
            },
            "throughput_ratio_to_baseline": throughput,
            "peak_npu_memory_delta_bytes": memory,
        }
    return {
        "schema_version": 1,
        "stage": "governance_composite_performance",
        "method": "C-DGE-V4.1",
        "evidence_class": "developmental_post_eligibility_evaluation",
        "confirmatory": False,
        "audit": {
            "success": audit_success,
            "measurement_rows": len(rows),
            "unique_measurement_keys": len(keys),
            "cells": len(cells),
            "observed_cell_key_sha256": observed_cell_sha,
            "expected_cell_key_sha256": EXPECTED_CELL_KEY_SHA256,
            "nonfinite_measurement_keys": nonfinite[:20],
            "invalid_measurement_keys": invalid[:20],
            "all_method_repeat_groups_complete": expected_group_counts,
            "paired_repeats_complete": pair_complete,
            "same_prompt_and_tokens_within_cells": token_identity,
        },
        "method_reports": method_reports,
        "paired_reports": paired_reports,
        "performance_evaluation_complete": audit_success,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing performance analysis: {args.output}")
    report = analyze(
        load_jsonl(args.input),
        json.loads(args.environment.read_text()),
        json.loads(args.evaluation_contract.read_text()),
        replicates=args.bootstrap_replicates,
    )
    report.update({
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
    })
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
