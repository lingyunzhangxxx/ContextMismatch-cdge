#!/usr/bin/env python3
"""Merge three terminal native behavior shards and freeze the control finalist."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def _receipt(path: Path, run_id: str) -> dict:
    value = json.loads(path.read_text())
    required = {
        "run_id": run_id,
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"behavior receipt mismatch: {field}")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError("behavior receipt lacks terminal Slurm provenance")
    if len(str(value.get("archive_sha256", ""))) != 64:
        raise ValueError("behavior receipt lacks archive SHA")
    return value


def _run_dir(bundle: Path, run_id: str) -> Path:
    direct = bundle / "runs" / run_id
    if direct.is_dir():
        return direct
    if bundle.name == run_id and (bundle / "behavior").is_dir():
        return bundle
    raise ValueError(f"behavior run directory is missing: {run_id}")


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def merge(
    *, protocol_path: Path, grid_path: Path, bundles: list[Path],
    receipts: list[Path], output: Path,
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if len(bundles) != 3 or len(receipts) != 3:
        raise ValueError("exactly three behavior bundles and receipts are required")
    protocol = json.loads(protocol_path.read_text())
    grid = json.loads(grid_path.read_text())
    if protocol.get("method_short_name") != "C-DGE-V4.2":
        raise ValueError("unexpected native protocol")
    grid_rows = {row["candidate_id"]: row for row in grid.get("candidates", [])}
    if len(grid_rows) != 27:
        raise ValueError("candidate grid is incomplete")
    candidates: list[dict] = []
    sources = []
    sites = set()
    for bundle, receipt_path in zip(bundles, receipts, strict=True):
        receipt_value = json.loads(receipt_path.read_text())
        run_id = str(receipt_value.get("run_id", ""))
        receipt = _receipt(receipt_path, run_id)
        run = _run_dir(bundle, run_id)
        report_path = run / "behavior" / "native_behavior_shard_report.json"
        report = json.loads(report_path.read_text())
        required = {
            "stage": "qwen35_cdge_v4_2_native_behavior_shard",
            "method": "C-DGE-V4.2",
            "behavior_complete": True,
            "candidate_count": 9,
            "rows_per_candidate": 896,
            "total_rows": 8064,
            "expected_key_sha256_per_candidate": "738cda7c0964ab75ef864405412028a95d207a1915dd4b58569428b525b874b1",
            "protected_controls_complete": False,
            "candidate_may_be_locked": False,
            "operator_dev_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        for field, expected in required.items():
            if report.get(field) != expected:
                raise ValueError(f"behavior shard mismatch: {field}")
        sites.add(report["site"])
        shard_candidates = report.get("candidates", [])
        if len(shard_candidates) != 9:
            raise ValueError("behavior shard candidate list is incomplete")
        for row in shard_candidates:
            candidate_id = row.get("candidate_id")
            frozen = grid_rows.get(candidate_id)
            if frozen is None or row.get("candidate_config") != frozen.get("config"):
                raise ValueError("behavior candidate differs from frozen grid")
            analysis = run / "behavior" / row["analysis"]
            raw = run / "behavior" / row["rows"]
            if sha256_file(analysis) != row.get("analysis_sha256"):
                raise ValueError("behavior analysis SHA mismatch")
            if sha256_file(raw) != row.get("rows_sha256"):
                raise ValueError("behavior rows SHA mismatch")
            analysis_value = json.loads(analysis.read_text())
            if analysis_value.get("candidate_id") != candidate_id or not _finite(analysis_value):
                raise ValueError("behavior analysis identity or finite audit failed")
            candidates.append({
                **row,
                "site": report["site"],
                "analysis_path": str(analysis),
                "analysis_sha256": sha256_file(analysis),
                "source_run_id": run_id,
                "source_archive_sha256": receipt["archive_sha256"],
            })
        sources.append({
            "run_id": run_id,
            "report": str(report_path),
            "report_sha256": sha256_file(report_path),
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "archive_sha256": receipt["archive_sha256"],
        })
    if len(sites) != 3 or len(candidates) != 27 or len({r["candidate_id"] for r in candidates}) != 27:
        raise ValueError("merged behavior evidence does not cover three sites and 27 candidates")
    if set(grid_rows) != {row["candidate_id"] for row in candidates}:
        raise ValueError("merged behavior candidates do not equal the frozen grid")
    eligible = [row for row in candidates if row.get("behavior_eligible") is True]
    finalist = None
    rejection = None
    if eligible:
        best_normalized = max(float(row["normalized_gap_reduction_lower_95"]) for row in eligible)
        tied = [row for row in eligible if float(row["normalized_gap_reduction_lower_95"]) == best_normalized]
        best_raw = max(float(row["gap_reduction_lower_95"]) for row in tied)
        tied = [row for row in tied if float(row["gap_reduction_lower_95"]) == best_raw]
        if len(tied) == 1:
            finalist = tied[0]
        else:
            rejection = "Top behavior candidates tie before the protected forced-on correction tie-break; controls are required for every tied candidate."
    else:
        rejection = "No candidate passed the prospectively frozen behavior gates; thresholds were not weakened."
    result = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_behavior_merge",
        "method": "C-DGE-V4.2",
        "behavior_complete": True,
        "site_count": 3,
        "candidate_count": 27,
        "eligible_candidate_count": len(eligible),
        "control_finalist": finalist,
        "control_finalist_selected": finalist is not None,
        "selection_pending_reason": rejection,
        "selection_rule": protocol["native_discovery"]["behavior_selection_rule"],
        "candidates": sorted(candidates, key=lambda row: row["candidate_id"]),
        "sources": sorted(sources, key=lambda row: row["run_id"]),
        "protocol_sha256": sha256_file(protocol_path),
        "candidate_grid_file_sha256": sha256_file(grid_path),
        "candidate_grid_sha256": grid["candidate_grid_sha256"],
        "protected_controls_complete": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--behavior-bundle", type=Path, action="append", required=True)
    parser.add_argument("--behavior-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = merge(
        protocol_path=args.protocol, grid_path=args.candidate_grid,
        bundles=args.behavior_bundle, receipts=args.behavior_receipt, output=args.output,
    )
    print(json.dumps({
        "candidate_count": result["candidate_count"],
        "eligible_candidate_count": result["eligible_candidate_count"],
        "control_finalist_selected": result["control_finalist_selected"],
        "control_finalist_id": (
            result["control_finalist"]["candidate_id"] if result["control_finalist"] else None
        ),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
