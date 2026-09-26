#!/usr/bin/env python3
"""Audit and summarize exact paired Qwen3-8B mechanism interventions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from .common import atomic_write_text, load_jsonl, sha256_file


def _group(rows: list[dict], fields: tuple[str, ...]) -> dict[tuple, list[dict]]:
    result = defaultdict(list)
    for row in rows:
        result[tuple(row[field] for field in fields)].append(row)
    return result


def _cluster_bootstrap(rows: list[dict], seed: int = 314159, replicates: int = 10000) -> dict:
    item_values = defaultdict(list)
    for row in rows:
        item_values[(row["benchmark"], row["item_id"])].append(float(row["patch_effect"]))
    by_benchmark = defaultdict(list)
    for (benchmark, _item_id), values in item_values.items():
        by_benchmark[benchmark].append(float(np.mean(values)))
    observed_benchmark = {
        benchmark: float(np.mean(values)) for benchmark, values in sorted(by_benchmark.items())
    }
    observed = float(np.mean(list(observed_benchmark.values()))) if observed_benchmark else math.nan
    if not by_benchmark:
        return {"estimate": observed, "ci95": [math.nan, math.nan], "replicates": replicates}
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    arrays = {benchmark: np.asarray(values, dtype=np.float64) for benchmark, values in by_benchmark.items()}
    for index in range(replicates):
        benchmark_draws = []
        for values in arrays.values():
            sampled = rng.integers(0, len(values), size=len(values))
            benchmark_draws.append(float(values[sampled].mean()))
        draws[index] = float(np.mean(benchmark_draws))
    return {
        "estimate": observed,
        "ci95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
        "replicates": replicates,
        "seed": seed,
    }


def _summary(rows: list[dict], with_bootstrap: bool = True) -> dict:
    effects = [float(row["patch_effect"]) for row in rows]
    gaps = [float(row["target_gap"]) for row in rows]
    benchmark_means = {
        key[0]: {
            "n_rows": len(values),
            "mean_patch_effect": float(np.mean([row["patch_effect"] for row in values])),
            "mean_target_gap": float(np.mean([row["target_gap"] for row in values])),
        }
        for key, values in sorted(_group(rows, ("benchmark",)).items())
    }
    result = {
        "n_rows": len(rows),
        "n_items": len({(row["benchmark"], row["item_id"]) for row in rows}),
        "mean_patch_effect": float(np.mean(effects)) if effects else math.nan,
        "mean_target_gap": float(np.mean(gaps)) if gaps else math.nan,
        "benchmark_means": benchmark_means,
        "equal_weight_benchmark_mean": (
            float(np.mean([value["mean_patch_effect"] for value in benchmark_means.values()]))
            if benchmark_means
            else math.nan
        ),
    }
    if with_bootstrap:
        result["cluster_bootstrap"] = _cluster_bootstrap(rows)
    return result


def _strata(rows: list[dict]) -> list[dict]:
    fields = ("benchmark", "declared_role", "history_style", "label_swap", "source_regime")
    result = []
    for key, values in sorted(_group(rows, fields).items()):
        record = dict(zip(fields, key))
        record.update(_summary(values, with_bootstrap=False))
        result.append(record)
    return result


def _nominate_layers(rows: list[dict], contract: dict) -> tuple[list[int], list[dict]]:
    rule = contract["residual_scan"]["component_nomination_rule"]
    eligible = set(rule["eligible_layers"])
    primary = [
        row
        for row in rows
        if row["patch_coefficient"] == 1.0
        and row["history_style"] == "lexical_matched"
        and row["layer"] in eligible
    ]
    scores = []
    for layer in sorted(eligible):
        layer_rows = [row for row in primary if row["layer"] == layer]
        cells = []
        for (benchmark, source_regime), values in sorted(
            _group(layer_rows, ("benchmark", "source_regime")).items()
        ):
            effect = float(np.mean([row["patch_effect"] for row in values]))
            gap = float(np.mean([row["target_gap"] for row in values]))
            ratio = abs(effect / gap) if gap != 0 else 0.0
            cells.append(
                {
                    "benchmark": benchmark,
                    "source_regime": source_regime,
                    "mean_patch_effect": effect,
                    "mean_target_gap": gap,
                    "absolute_mediation_ratio": ratio,
                    "ranking_ratio_clipped_0_1": min(1.0, max(0.0, ratio)),
                }
            )
        score = float(np.mean([cell["ranking_ratio_clipped_0_1"] for cell in cells]))
        scores.append({"layer": layer, "ranking_score": score, "cells": cells})
    ranked = sorted(scores, key=lambda row: (-row["ranking_score"], row["layer"]))
    nominated = [row["layer"] for row in ranked[: rule["select_top_k"]]]
    return nominated, ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--mechanism-erratum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nomination-output", type=Path)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.mechanism_contract.read_text())
    mechanism_erratum_sha256 = sha256_file(args.mechanism_erratum)
    if environment.get("mechanism_contract_erratum_sha256") != mechanism_erratum_sha256:
        raise ValueError("environment mechanism contract erratum SHA mismatch")
    keys = [row["job_key"] for row in rows]
    observed_key_sha256 = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(set(keys))).encode("utf-8")
    ).hexdigest()
    finite_fields = (
        "logit_a",
        "logit_b",
        "correct_logit_margin",
        "source_cached_margin",
        "target_cached_margin",
        "patch_effect",
        "target_gap",
    )
    nonfinite = [
        row["job_key"]
        for row in rows
        if not all(math.isfinite(float(row[field])) for field in finite_fields)
    ]
    coefficient_zero = [row for row in rows if float(row["patch_coefficient"]) == 0.0]
    coefficient_zero_max_error = max(
        (abs(float(row["correct_logit_margin"]) - float(row["source_cached_margin"])) for row in coefficient_zero),
        default=math.nan,
    )
    endpoint = [
        row
        for row in rows
        if row["mode"] == "residual"
        and row["layer"] == contract["residual_scan"]["last_layer_endpoint_positive_control"]
        and float(row["patch_coefficient"]) == 1.0
    ]
    endpoint_max_error = (
        max(
            abs(float(row["correct_logit_margin"]) - float(row["target_cached_margin"]))
            for row in endpoint
        )
        if endpoint
        else None
    )
    audit = {
        "row_count": len(rows),
        "planned_rows": environment["planned_rows"],
        "unique_job_keys": len(set(keys)),
        "observed_job_key_set_sha256": observed_key_sha256,
        "expected_job_key_set_sha256": environment["expected_job_key_set_sha256"],
        "duplicate_job_keys": len(keys) - len(set(keys)),
        "nonfinite_rows": len(nonfinite),
        "coefficient_zero_rows": len(coefficient_zero),
        "coefficient_zero_max_absolute_margin_error": coefficient_zero_max_error,
        "residual_endpoint_rows": len(endpoint),
        "residual_endpoint_max_absolute_target_margin_error": endpoint_max_error,
    }
    audit["success"] = (
        audit["row_count"] == audit["planned_rows"]
        and audit["duplicate_job_keys"] == 0
        and observed_key_sha256 == environment["expected_job_key_set_sha256"]
        and audit["nonfinite_rows"] == 0
        and coefficient_zero_max_error == contract["numerical_gates"]["coefficient_zero_max_absolute_margin_error"]
        and (
            environment["mode"] != "residual"
            or endpoint_max_error
            == contract["numerical_gates"]["residual_layer_35_coefficient_one_max_target_margin_error"]
        )
    )

    coefficient_one = [row for row in rows if float(row["patch_coefficient"]) == 1.0]
    cell_fields = ("layer", "component", "source_regime")
    cells = []
    for key, values in sorted(_group(coefficient_one, cell_fields).items()):
        record = dict(zip(cell_fields, key))
        record["direction"] = (
            "obedience_to_verification_rescue"
            if record["source_regime"] == "obedience"
            else "verification_to_obedience_reverse_induction"
        )
        record.update(_summary(values))
        record["strata"] = _strata(values)
        cells.append(record)

    report = {
        "schema_version": 1,
        "mode": environment["mode"],
        "stage": environment["stage"],
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "mechanism_contract_sha256": sha256_file(args.mechanism_contract),
        "mechanism_contract_erratum_sha256": mechanism_erratum_sha256,
        "audit": audit,
        "coefficient_one_cells": cells,
        "interpretation_boundary": {
            "component_ordering_exploratory": environment["mode"] == "component",
            "component_effects_are_not_additive": True,
            "qwen3_5_results_not_pooled": True,
        },
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    if args.nomination_output:
        if environment["mode"] != "residual" or environment["stage"] != "full":
            raise ValueError("layer nomination is allowed only after the full residual scan")
        if not audit["success"]:
            raise ValueError("cannot nominate layers from a failed residual audit")
        nominated, ranking = _nominate_layers(rows, contract)
        nomination = {
            "schema_version": 1,
            "locked": True,
            "selection_partition": contract["data_split"]["partition"],
            "mechanism_contract_sha256": sha256_file(args.mechanism_contract),
            "mechanism_contract_erratum_sha256": mechanism_erratum_sha256,
            "source_residual_rows_sha256": sha256_file(args.input),
            "source_residual_analysis_sha256": sha256_file(args.output),
            "nominated_layers": nominated,
            "ranking": ranking,
            "selection_rule": contract["residual_scan"]["component_nomination_rule"],
            "final_test_open": False,
            "production_rollout_approved": False,
        }
        atomic_write_text(
            args.nomination_output,
            json.dumps(nomination, indent=2, sort_keys=True) + "\n",
            args.allow_overwrite,
        )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not audit["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
