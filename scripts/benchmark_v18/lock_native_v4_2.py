#!/usr/bin/env python3
"""Select and lock C-DGE V4.2 after every frozen tied finalist has controls."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v18.materialize_native_controls_authorization import frozen_tie

EXPECTED_KEY = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"
RANK_ORDER = {"compact": 0, "balanced": 1, "wide": 2}


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _receipt(path: Path, run_id: str) -> dict:
    value = json.loads(path.read_text())
    for field, expected in {"run_id": run_id, "cluster_shared_copy_verified": True, "host_data_copy_verified": True, "local_copy_verified": True}.items():
        if value.get(field) != expected:
            raise ValueError(f"controls receipt mismatch: {field}")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError("controls receipt lacks terminal provenance")
    return value


def build(merge: dict, controls: list[dict], receipts: list[dict], sources: list[dict]) -> tuple[dict, dict | None]:
    tied = frozen_tie(merge)
    tied_by_id = {row["candidate_id"]: row for row in tied}
    controls_by_id = {row.get("candidate_id"): row for row in controls}
    checks = {
        "behavior_complete": merge.get("behavior_complete") is True and merge.get("candidate_count") == 27,
        "frozen_tie_complete": len(tied_by_id) >= 2 and set(controls_by_id) == set(tied_by_id),
        "receipt_count": len(receipts) == len(tied_by_id),
        "safety_flags": merge.get("final_test_open") is False and merge.get("final_test_open_count") == 0 and merge.get("production_rollout_approved") is False,
        "finite": _finite(merge) and _finite(controls),
    }
    candidates = []
    for candidate_id, behavior in sorted(tied_by_id.items()):
        control = controls_by_id.get(candidate_id, {})
        audit = control.get("audit", {})
        correction = control.get("protected_forced_on_relative_correction", {})
        candidate_checks = {
            "identity": control.get("method") == "C-DGE-V4.2" and control.get("stage") == "qwen35_cdge_v4_2_native_protected_controls" and control.get("candidate_id") == candidate_id,
            "audit": audit.get("success") is True and audit.get("row_count") == audit.get("unique_case_keys") == 2856,
            "key": audit.get("observed_key_sha256") == audit.get("expected_key_sha256") == EXPECTED_KEY,
            "admissible": control.get("controls_admissible") is True,
            "correction": correction.get("aggregation") == "maximum of the two per-direction means over protected-control rows" and isinstance(correction.get("maximum"), (int, float)) and math.isfinite(float(correction["maximum"])),
            "safety": control.get("final_test_open") is False and control.get("final_test_open_count") == 0 and control.get("production_rollout_approved") is False,
        }
        checks[f"candidate_{candidate_id}"] = all(candidate_checks.values())
        candidates.append({
            "candidate_id": candidate_id, "candidate_config": behavior["candidate_config"],
            "normalized_gap_reduction_lower_95": behavior["normalized_gap_reduction_lower_95"],
            "gap_reduction_lower_95": behavior["gap_reduction_lower_95"],
            "protected_forced_on_relative_correction": correction.get("maximum"),
            "gate_checks": candidate_checks,
        })
    failures = [name for name, passed in checks.items() if not passed]
    selected = None
    if not failures:
        selected = min(candidates, key=lambda row: (
            float(row["protected_forced_on_relative_correction"]),
            float(row["candidate_config"]["max_relative_correction"]),
            RANK_ORDER[row["candidate_config"]["rank_profile"]],
            int(row["candidate_config"]["layer"]),
        ))
    report = {
        "schema_version": 1, "stage": "qwen35_cdge_v4_2_native_pareto_report",
        "method": "C-DGE-V4.2", "model": "Qwen3.5-9B", "admissible": not failures,
        "gate_checks": checks, "rejection_reasons": failures, "frozen_tied_candidates": candidates,
        "selected_candidate": selected, "sources": sources,
        "candidate_may_be_locked": not failures, "final_test_open": False,
        "final_test_open_count": 0, "production_rollout_approved": False,
    }
    if failures:
        return report, None
    lock = {
        **report, "manifest_id": "context-mismatch-qwen3-5-9b-cdge-v4-2-native-lock-v2",
        "stage": "qwen35_cdge_v4_2_native_pareto_lock", "locked": True,
        "candidate_id": selected["candidate_id"], "candidate_config": selected["candidate_config"],
        "selection_partition": "prospectively frozen fold 7 behavior plus all tied protected controls",
    }
    return report, lock


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-merge", type=Path, required=True)
    parser.add_argument("--controls-analysis", type=Path, action="append", required=True)
    parser.add_argument("--controls-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists() or args.output_lock.exists():
        raise FileExistsError("refusing existing native lock output")
    if len(args.controls_analysis) != len(args.controls_receipt):
        raise ValueError("control analysis/receipt count mismatch")
    merge = json.loads(args.behavior_merge.read_text())
    controls, receipts, sources = [], [], []
    for analysis_path, receipt_path in zip(args.controls_analysis, args.controls_receipt, strict=True):
        control = json.loads(analysis_path.read_text())
        receipt_value = json.loads(receipt_path.read_text())
        receipt = _receipt(receipt_path, str(receipt_value.get("run_id", "")))
        controls.append(control); receipts.append(receipt)
        sources.append({
            "candidate_id": control.get("candidate_id"),
            "controls_analysis": {"path": str(analysis_path), "sha256": sha256_file(analysis_path)},
            "controls_receipt": {"path": str(receipt_path), "sha256": sha256_file(receipt_path), "archive_sha256": receipt["archive_sha256"]},
        })
    report, lock = build(merge, controls, receipts, sources)
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if lock is None:
        raise SystemExit("C-DGE V4.2 tied controls are not admissible; final test remains closed")
    lock["behavior_merge_sha256"] = sha256_file(args.behavior_merge)
    lock["pareto_report_sha256"] = sha256_file(args.output_report)
    atomic_write_text(args.output_lock, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_id": lock["candidate_id"], "locked": True}, sort_keys=True))


if __name__ == "__main__":
    main()
