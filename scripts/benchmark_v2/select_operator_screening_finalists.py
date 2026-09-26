#!/usr/bin/env python3
"""Select up to three behaviorally eligible screening candidates per family."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def _resolve(parent: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else parent / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing output: {args.output}")
    ledger = json.loads(args.ledger.read_text())
    extension = json.loads(args.extension_contract.read_text())
    per_family = defaultdict(list)
    audited = []
    for entry in ledger["candidates"]:
        manifest_path = _resolve(args.ledger.parent, entry["candidate_manifest"])
        analysis_path = _resolve(args.ledger.parent, entry["screening_analysis"])
        manifest = json.loads(manifest_path.read_text())
        analysis = json.loads(analysis_path.read_text())
        if manifest["candidate_id"] != analysis["candidate_id"]:
            raise ValueError("candidate identity mismatch")
        directional = analysis["directional_recovery"]
        label_swaps = analysis["gap_reduction_by_label_swap"]
        eligible = (
            analysis.get("audit", {}).get("success") is True
            and float(analysis["audit"].get("zero_gate_max_error", 1.0)) == 0.0
            and float(
                analysis["matched_minus_mismatched_reduction"][
                    "equal_weight_benchmark_mean"
                ]
            )
            > 0.0
            and all(
                float(value["equal_weight_benchmark_mean"]) > 0.0
                for value in directional.values()
            )
            and all(
                float(value["equal_weight_benchmark_mean"]) > 0.0
                for value in label_swaps.values()
            )
        )
        record = {
            "candidate_id": manifest["candidate_id"],
            "family": manifest["config"]["family"],
            "site_order": manifest["site_order"],
            "candidate_manifest": str(manifest_path),
            "candidate_manifest_sha256": sha256_file(manifest_path),
            "candidate_tensor": str(manifest_path.parent / manifest["tensor_file"]),
            "candidate_tensor_sha256": manifest["tensor_sha256"],
            "screening_analysis": str(analysis_path),
            "screening_analysis_sha256": sha256_file(analysis_path),
            "gap_reduction": float(
                analysis["matched_minus_mismatched_reduction"][
                    "equal_weight_benchmark_mean"
                ]
            ),
            "mean_relative_intervention_norm": float(
                analysis["intervention_norm"]["mean_sum_site_relative_norm"]
            ),
            "screening_eligible": eligible,
        }
        audited.append(record)
        if eligible:
            per_family[record["family"]].append(record)
    finalists = []
    target_counts = extension["compute_schedule"]["screening_shortlist"]
    for family in sorted(target_counts):
        ordered = sorted(
            per_family[family],
            key=lambda row: (
                -row["gap_reduction"],
                row["mean_relative_intervention_norm"],
                row["candidate_id"],
            ),
        )
        finalists.extend(ordered[:3])
    report = {
        "schema_version": 1,
        "ledger_sha256": sha256_file(args.ledger),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "audited_candidates": len(audited),
        "eligible_candidates": sum(row["screening_eligible"] for row in audited),
        "finalist_count": len(finalists),
        "audited": audited,
        "finalists": finalists,
        "selection_partition": "operator_dev screening half only",
        "protected_controls_still_required": True,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "audited_candidates": report["audited_candidates"],
                "eligible_candidates": report["eligible_candidates"],
                "finalist_count": report["finalist_count"],
                "final_test_open": False,
                "production_rollout_approved": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
