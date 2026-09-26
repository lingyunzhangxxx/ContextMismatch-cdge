#!/usr/bin/env python3
"""Lock Qwen3-8B operator sites from the frozen component-discovery scan."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .common import atomic_write_text, load_jsonl, sha256_file


def _mean(rows: list[dict], field: str) -> float:
    return statistics.fmean(float(row[field]) for row in rows)


def _site_score(rows: list[dict], layer: int, component: str) -> tuple[float, list[dict]]:
    selected = [
        row
        for row in rows
        if int(row["layer"]) == layer
        and row["component"] == component
        and row["history_style"] == "lexical_matched"
        and float(row["patch_coefficient"]) == 1.0
    ]
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in selected:
        grouped[(row["benchmark"], row["source_regime"])].append(row)
    cells = []
    for (benchmark, source_regime), values in sorted(grouped.items()):
        effect = _mean(values, "patch_effect")
        gap = _mean(values, "target_gap")
        ratio = abs(effect / gap) if gap else 0.0
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
    if len(cells) != 12:
        raise ValueError(
            f"site {(layer, component)} has {len(cells)} primary cells instead of 12"
        )
    return statistics.fmean(cell["ranking_ratio_clipped_0_1"] for cell in cells), cells


def _sign_diagnostic(rows: list[dict], layer: int, component: str) -> list[dict]:
    selected = [
        row
        for row in rows
        if int(row["layer"]) == layer
        and row["component"] == component
        and row["history_style"] == "lexical_matched"
        and float(row["patch_coefficient"]) == 1.0
    ]
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in selected:
        grouped[(row["source_regime"], int(row["label_swap"]))].append(row)
    result = []
    for (source_regime, label_swap), values in sorted(grouped.items()):
        effect = _mean(values, "patch_effect")
        expected = effect > 0 if source_regime == "obedience" else effect < 0
        result.append(
            {
                "source_regime": source_regime,
                "label_swap": label_swap,
                "mean_patch_effect": effect,
                "expected_direction": expected,
            }
        )
    if len(result) != 4:
        raise ValueError(
            f"site {(layer, component)} has {len(result)} sign cells instead of four"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-rows", type=Path, required=True)
    parser.add_argument("--component-analysis", type=Path, required=True)
    parser.add_argument("--component-manifest", type=Path, required=True)
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mechanism = json.loads(args.mechanism_contract.read_text())
    execution = json.loads(args.execution_contract.read_text())
    analysis = json.loads(args.component_analysis.read_text())
    component_manifest = json.loads(args.component_manifest.read_text())
    mechanism_sha = sha256_file(args.mechanism_contract)
    if execution["mechanism_contract_sha256"] != mechanism_sha:
        raise ValueError("execution contract does not bind the mechanism contract")
    if execution["operator_contract_sha256"] != sha256_file(args.operator_contract):
        raise ValueError("execution contract does not bind the operator contract")
    if component_manifest["mechanism_contract_sha256"] != mechanism_sha:
        raise ValueError("component manifest mechanism binding mismatch")
    if analysis.get("mode") != "component" or analysis.get("stage") != "full":
        raise ValueError("operator sites require the full component scan")
    if not analysis.get("audit", {}).get("success"):
        raise ValueError("component analysis did not pass its audit")
    if analysis["input_sha256"] != sha256_file(args.component_rows):
        raise ValueError("component row lineage mismatch")

    rows = load_jsonl(args.component_rows)
    if len(rows) != mechanism["component_scan"]["expected_full_rows"]:
        raise ValueError("component row count mismatch")
    layers = [int(value) for value in component_manifest["nominated_layers"]]
    components = list(execution["site_nomination"]["eligible_components"])
    rankings = []
    selected_sites = []
    for layer in layers:
        candidates = []
        for component in components:
            score, cells = _site_score(rows, layer, component)
            candidates.append(
                {
                    "layer": layer,
                    "component": component,
                    "ranking_score": score,
                    "cells": cells,
                    "sign_diagnostic": _sign_diagnostic(rows, layer, component),
                }
            )
        tie_order = {"self_attn": 0, "mlp": 1}
        candidates.sort(
            key=lambda row: (-row["ranking_score"], tie_order.get(row["component"], 99), row["component"])
        )
        rankings.extend(candidates)
        selected_sites.append(
            {
                "layer": layer,
                "component": candidates[0]["component"],
                "ranking_score": candidates[0]["ranking_score"],
                "all_four_direction_label_cells_expected": all(
                    cell["expected_direction"] for cell in candidates[0]["sign_diagnostic"]
                ),
            }
        )
    selected_sites.sort(key=lambda row: (-row["ranking_score"], row["layer"], row["component"]))
    output = {
        "schema_version": 1,
        "locked": True,
        "selection_partition": "component_discovery",
        "mechanism_contract_sha256": mechanism_sha,
        "operator_site_execution_contract_sha256": sha256_file(args.execution_contract),
        "component_layer_manifest_sha256": sha256_file(args.component_manifest),
        "component_rows_sha256": sha256_file(args.component_rows),
        "component_analysis_sha256": sha256_file(args.component_analysis),
        "selected_sites_ordered": selected_sites,
        "nominated_layers": layers,
        "rankings": rankings,
        "selection_rule": execution["site_nomination"],
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(output, indent=2, sort_keys=True) + "\n")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
