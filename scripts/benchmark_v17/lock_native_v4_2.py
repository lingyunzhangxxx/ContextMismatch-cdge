#!/usr/bin/env python3
"""Create the C-DGE V4.2 Pareto report and lock from terminal evidence only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


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
    checks = {
        "run_id": run_id,
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }
    for field, expected in checks.items():
        if value.get(field) != expected:
            raise ValueError(f"controls receipt mismatch: {field}")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError("controls receipt lacks terminal provenance")
    return value


def build(*, merge: dict, controls: dict, receipt: dict, sources: dict) -> tuple[dict, dict | None]:
    finalist = merge.get("control_finalist") or {}
    candidate_id = finalist.get("candidate_id")
    checks = {
        "behavior_complete": merge.get("behavior_complete") is True,
        "candidate_grid_complete": merge.get("candidate_count") == 27,
        "behavior_finalist_selected": merge.get("control_finalist_selected") is True,
        "behavior_finalist_eligible": finalist.get("behavior_eligible") is True,
        "controls_identity": controls.get("method") == "C-DGE-V4.2"
        and controls.get("stage") == "qwen35_cdge_v4_2_native_protected_controls"
        and controls.get("candidate_id") == candidate_id,
        "controls_audit": controls.get("audit", {}).get("success") is True,
        "controls_rows": controls.get("audit", {}).get("row_count") == 2856
        and controls.get("audit", {}).get("unique_case_keys") == 2856,
        "controls_key": controls.get("audit", {}).get("observed_key_sha256")
        == controls.get("audit", {}).get("expected_key_sha256")
        == "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77",
        "controls_admissible": controls.get("controls_admissible") is True,
        "receipt_archive": len(str(receipt.get("archive_sha256", ""))) == 64,
        "safety_flags": all(value is False for value in (
            merge.get("final_test_open"), merge.get("production_rollout_approved"),
            controls.get("final_test_open"), controls.get("production_rollout_approved"),
        )) and merge.get("final_test_open_count") == controls.get("final_test_open_count") == 0,
        "finite": _finite(merge) and _finite(controls),
    }
    failures = [name for name, passed in checks.items() if not passed]
    report = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_pareto_report",
        "method": "C-DGE-V4.2",
        "model": "Qwen3.5-9B",
        "candidate_id": candidate_id,
        "candidate_config": finalist.get("candidate_config"),
        "admissible": not failures,
        "gate_checks": checks,
        "rejection_reasons": failures,
        "locked_metrics": {
            "normalized_gap_reduction_lower_95": finalist.get("normalized_gap_reduction_lower_95"),
            "gap_reduction_lower_95": finalist.get("gap_reduction_lower_95"),
            "maximum_relative_correction": finalist.get("candidate_config", {}).get("max_relative_correction"),
            "rank_profile": finalist.get("candidate_config", {}).get("rank_profile"),
            "controls_admissible": controls.get("controls_admissible"),
        },
        "sources": sources,
        "candidate_may_be_locked": not failures,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if failures:
        return report, None
    lock = {
        **report,
        "manifest_id": "context-mismatch-qwen3-5-9b-cdge-v4-2-native-lock-v1",
        "stage": "qwen35_cdge_v4_2_native_pareto_lock",
        "locked": True,
        "selection_partition": "prospectively frozen fold 7 behavior plus protected controls",
    }
    return report, lock


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("behavior_merge", "controls_analysis", "controls_receipt", "output_report", "output_lock"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists() or args.output_lock.exists():
        raise FileExistsError("refusing existing native lock output")
    merge = json.loads(args.behavior_merge.read_text())
    controls = json.loads(args.controls_analysis.read_text())
    receipt_value = json.loads(args.controls_receipt.read_text())
    receipt = _receipt(args.controls_receipt, str(receipt_value.get("run_id", "")))
    sources = {
        "behavior_merge": {"path": str(args.behavior_merge), "sha256": sha256_file(args.behavior_merge)},
        "controls_analysis": {"path": str(args.controls_analysis), "sha256": sha256_file(args.controls_analysis)},
        "controls_receipt": {"path": str(args.controls_receipt), "sha256": sha256_file(args.controls_receipt), "archive_sha256": receipt["archive_sha256"]},
    }
    report, lock = build(merge=merge, controls=controls, receipt=receipt, sources=sources)
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if lock is None:
        raise SystemExit("C-DGE V4.2 is not admissible; final test remains closed")
    lock["pareto_report_sha256"] = sha256_file(args.output_report)
    atomic_write_text(args.output_lock, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"candidate_id": lock["candidate_id"], "locked": True}, sort_keys=True))


if __name__ == "__main__":
    main()
