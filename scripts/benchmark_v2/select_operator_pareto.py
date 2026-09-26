#!/usr/bin/env python3
"""Apply frozen admissibility gates, report the Pareto front, and lock one operator."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def _candidate_metrics(entry: dict, parent: Path, extension: dict) -> dict:
    manifest_path = _resolve(parent, entry["candidate_manifest"])
    analysis_path = _resolve(parent, entry["selection_analysis"])
    controls_path = _resolve(parent, entry["controls_analysis"])
    manifest = json.loads(manifest_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    controls = json.loads(controls_path.read_text())
    candidate_id = manifest["candidate_id"]
    if analysis.get("candidate_id") != candidate_id or controls.get("candidate_id") != candidate_id:
        raise ValueError(f"candidate identity mismatch: {candidate_id}")
    if manifest.get("final_test_open") is not False:
        raise ValueError("candidate manifest unexpectedly opens final test")
    if manifest.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    manifest_remote = entry.get("candidate_manifest_remote")
    tensor_remote = entry.get("candidate_tensor_remote")
    for name, value in (
        ("candidate_manifest_remote", manifest_remote),
        ("candidate_tensor_remote", tensor_remote),
    ):
        if not isinstance(value, str) or not value.startswith(
            "/workspace/context-mismatch-qwen3-8b/"
        ):
            raise ValueError(f"invalid owned remote artifact path {name}: {candidate_id}")
    selection = extension["selection"]["admissibility"]
    reduction = analysis["matched_minus_mismatched_reduction"]
    reduction_bootstrap = analysis["matched_minus_mismatched_reduction_bootstrap"]
    normalized_bootstrap = analysis["normalized_gap_reduction_bootstrap"]
    directional = analysis["directional_recovery"]
    label_swaps = analysis["gap_reduction_by_label_swap"]
    benchmarks = analysis["gap_reduction_by_benchmark"]
    matched = analysis["matched_cell_collateral"]
    reasons = []
    if not analysis.get("audit", {}).get("success"):
        reasons.append("selection behavior audit failed")
    if not controls.get("audit", {}).get("success"):
        reasons.append("protected-control audit failed")
    if float(analysis["audit"].get("zero_gate_max_error", math.inf)) != 0.0:
        reasons.append("zero gate is not exact identity")
    if float(reduction_bootstrap["ci95"][0]) < float(selection["gap_reduction_lower_95_minimum"]):
        reasons.append("gap-reduction lower confidence bound is too small")
    if float(normalized_bootstrap["ci95"][0]) < float(
        selection["normalized_gap_reduction_lower_95_minimum"]
    ):
        reasons.append("normalized-reduction lower confidence bound is too small")
    for name, report in directional.items():
        if float(report["equal_weight_benchmark_mean"]) <= 0.0:
            reasons.append(f"non-positive directional recovery: {name}")
    for name, report in label_swaps.items():
        if float(report["equal_weight_benchmark_mean"]) <= 0.0:
            reasons.append(f"non-positive label-swap recovery: {name}")
    for name, report in benchmarks.items():
        if float(report["equal_weight_benchmark_mean"]) < 0.0:
            reasons.append(f"negative benchmark recovery: {name}")
    matched_fractions = []
    for name, report in matched.items():
        loss_upper = float(report["margin_loss_upper"]["upper_one_sided_95"])
        kl_upper = float(report["binary_kl_upper"]["upper_one_sided_95"])
        loss_limit = float(selection["matched_cell_margin_loss_upper_95_maximum"])
        kl_limit = float(selection["matched_cell_binary_kl_upper_95_maximum"])
        matched_fractions.extend((loss_upper / loss_limit, kl_upper / kl_limit))
        if loss_upper > loss_limit:
            reasons.append(f"matched-cell margin loss violation: {name}")
        if kl_upper > kl_limit:
            reasons.append(f"matched-cell KL violation: {name}")
    if controls.get("controls_admissible") is not True:
        reasons.append("protected controls are not admissible")
    control_fractions = []
    for family in controls["family_reports"].values():
        for mode in family.values():
            control_fractions.extend(
                (
                    float(mode["margin_loss"]["upper_one_sided_95"])
                    / float(mode["margin_loss_tolerance"]),
                    float(mode["binary_kl"]["upper_one_sided_95"])
                    / float(mode["binary_kl_tolerance"]),
                )
            )
    config = manifest["config"]
    site_order = list(manifest["site_order"])
    if "per_site_configs" in manifest:
        total_rank = sum(
            int(value["history_rank"])
            + int(value["context_rank"])
            + int(value["correction_rank"])
            for value in manifest["per_site_configs"].values()
        )
    else:
        total_rank = len(site_order) * (
            int(config["history_rank"])
            + int(config["context_rank"])
            + int(config["correction_rank"])
        )
    return {
        "candidate_id": candidate_id,
        "candidate_manifest": str(manifest_path),
        "candidate_manifest_remote": manifest_remote,
        "candidate_manifest_sha256": sha256_file(manifest_path),
        "candidate_tensor": str(manifest_path.parent / manifest["tensor_file"]),
        "candidate_tensor_remote": tensor_remote,
        "candidate_tensor_sha256": manifest["tensor_sha256"],
        "selection_analysis": str(analysis_path),
        "selection_analysis_sha256": sha256_file(analysis_path),
        "controls_analysis": str(controls_path),
        "controls_analysis_sha256": sha256_file(controls_path),
        "config": config,
        "site_order": site_order,
        "gap_reduction": float(reduction["equal_weight_benchmark_mean"]),
        "gap_reduction_lower_95": float(reduction_bootstrap["ci95"][0]),
        "normalized_gap_reduction_lower_95": float(normalized_bootstrap["ci95"][0]),
        "worst_protection_fraction": max([*matched_fractions, *control_fractions]),
        "total_rank": total_rank,
        "edited_sites": len(site_order),
        "mean_relative_intervention_norm": float(
            analysis["intervention_norm"]["mean_sum_site_relative_norm"]
        ),
        "admissible": not reasons,
        "rejection_reasons": reasons,
    }


def _dominates(left: dict, right: dict) -> bool:
    left_values = (
        -left["gap_reduction"],
        left["worst_protection_fraction"],
        left["total_rank"],
        left["edited_sites"],
        left["mean_relative_intervention_norm"],
    )
    right_values = (
        -right["gap_reduction"],
        right["worst_protection_fraction"],
        right["total_rank"],
        right["edited_sites"],
        right["mean_relative_intervention_norm"],
    )
    return all(a <= b for a, b in zip(left_values, right_values)) and any(
        a < b for a, b in zip(left_values, right_values)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists() or args.output_lock.exists():
        raise FileExistsError("refusing existing Pareto/lock output")
    ledger = json.loads(args.ledger.read_text())
    extension = json.loads(args.extension_contract.read_text())
    candidates = [
        _candidate_metrics(entry, args.ledger.parent, extension)
        for entry in ledger["candidates"]
    ]
    if len({row["candidate_id"] for row in candidates}) != len(candidates):
        raise ValueError("candidate ledger contains duplicate ids")
    admissible = [row for row in candidates if row["admissible"]]
    if not admissible:
        raise SystemExit("no candidate satisfies every frozen admissibility gate")
    front = [
        row
        for row in admissible
        if not any(_dominates(other, row) for other in admissible if other is not row)
    ]
    chosen = min(
        front,
        key=lambda row: (
            -row["gap_reduction"],
            row["worst_protection_fraction"],
            row["total_rank"],
            row["edited_sites"],
            row["mean_relative_intervention_norm"],
            row["candidate_id"],
        ),
    )
    report = {
        "schema_version": 1,
        "ledger_sha256": sha256_file(args.ledger),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "candidate_count": len(candidates),
        "admissible_count": len(admissible),
        "pareto_front_ids": sorted(row["candidate_id"] for row in front),
        "chosen_candidate_id": chosen["candidate_id"],
        "candidates": candidates,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    lock = {
        "schema_version": 1,
        "manifest_id": "context-mismatch-qwen3-8b-operator-lock-v1",
        "locked": True,
        "selection_partition": "operator_dev only",
        "chosen_candidate_id": chosen["candidate_id"],
        "candidate_tensor": chosen["candidate_tensor"],
        "candidate_tensor_remote": chosen["candidate_tensor_remote"],
        "candidate_tensor_sha256": chosen["candidate_tensor_sha256"],
        "candidate_manifest": chosen["candidate_manifest"],
        "candidate_manifest_remote": chosen["candidate_manifest_remote"],
        "candidate_manifest_sha256": chosen["candidate_manifest_sha256"],
        "selection_analysis_sha256": chosen["selection_analysis_sha256"],
        "controls_analysis_sha256": chosen["controls_analysis_sha256"],
        "config": chosen["config"],
        "site_order": chosen["site_order"],
        "locked_metrics": {
            key: chosen[key]
            for key in (
                "gap_reduction",
                "gap_reduction_lower_95",
                "normalized_gap_reduction_lower_95",
                "worst_protection_fraction",
                "total_rank",
                "edited_sites",
                "mean_relative_intervention_norm",
            )
        },
        "pareto_report_sha256": sha256_file(args.output_report),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output_lock, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
