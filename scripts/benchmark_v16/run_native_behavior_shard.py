#!/usr/bin/env python3
"""Run fold-7 behavior characterization for all nine candidates at one native site."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v16.materialize_native_behavior_authorization import (
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
)


def _write(path: Path, value: object, *, allow_overwrite: bool = False) -> None:
    atomic_write_text(
        path,
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        allow_overwrite=allow_overwrite,
    )


def _bound_path(authorization: dict, name: str) -> Path:
    record = authorization.get("bound_artifacts", {}).get(name, {})
    path = Path(str(record.get("path", "")))
    if not path.is_file() or sha256_file(path) != record.get("sha256"):
        raise ValueError(f"native behavior bound artifact mismatch: {name}")
    return path


def _validate_authorization(args: argparse.Namespace) -> tuple[dict, dict[str, Path]]:
    value = json.loads(args.execution_authorization.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_behavior_shard",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "candidate_count": 9,
        "rows_per_candidate": EXPECTED_ROWS,
        "expected_key_sha256_per_candidate": EXPECTED_KEY_SHA256,
        "total_rows": 9 * EXPECTED_ROWS,
        "fit_gate_failure_does_not_suppress_behavior_characterization": True,
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"native behavior authorization mismatch: {field}")
    resource = value.get("resource_contract", {})
    for field, expected in {
        "partition": "a01", "nodes": 1, "ntasks": 1,
        "node": os.uname().nodename, "cpus_per_task": 8,
        "mem_mib": 131072, "npu_type": "910B3", "npus": 1,
        "time_limit": "12:00:00",
    }.items():
        if resource.get(field) != expected:
            raise ValueError(f"native behavior resource mismatch: {field}")
    candidates = value.get("candidates", [])
    if len(candidates) != 9 or len({row.get("candidate_id") for row in candidates}) != 9:
        raise ValueError("native behavior candidate authorization is incomplete")
    for candidate in candidates:
        for name in ("checkpoint", "editor_contract", "source_fit_report", "cap_manifest"):
            record = candidate.get(name, {})
            path = Path(str(record.get("path", "")))
            if not path.is_file() or sha256_file(path) != record.get("sha256"):
                raise ValueError(f"native behavior candidate artifact mismatch: {name}")
    paths = {
        name: _bound_path(value, name)
        for name in (
            "protocol", "candidate_grid", "fit_profiles", "fit_shard_report",
            "fit_receipt", "crossover_contract", "manifest", "manifest_report",
            "design_audit", "model_manifest",
        )
    }
    return value, paths


def _compatibility_fit_report(
    *, records: list[dict], contract: Path, output: Path,
) -> None:
    site = f"{records[0]['candidate_config']['layer']}:{records[0]['candidate_config']['component']}"
    value = {
        "schema_version": 1,
        "stage": "governance_directional_fit",
        "method": "DSGE-V3",
        "fit_complete": True,
        "candidate_count": 3,
        "all_three_candidate_checkpoints_materialized": True,
        "editor_contract_sha256": sha256_file(contract),
        "candidate_reports": [
            {
                "candidate_id": row["candidate_id"],
                "site": site,
                "checkpoint_sha256": row["checkpoint"]["sha256"],
                "fit_eligible": bool(row["fit_eligible"]),
                "direct_behavior_characterization_required_even_if_ineligible": True,
            }
            for row in sorted(records, key=lambda item: item["candidate_id"])
        ],
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    _write(output, value)


def _behavior_checks(report: dict, fit_eligible: bool) -> dict:
    directions = report.get("directional_recovery", {})
    swaps = report.get("gap_reduction_by_label_swap", {})
    benchmarks = report.get("gap_reduction_by_benchmark", {})
    return {
        "fit_and_structural_gates_pass": fit_eligible,
        "both_mismatch_direction_point_estimates_positive": len(directions) == 2 and all(
            value.get("equal_weight_benchmark_mean") is not None
            and float(value["equal_weight_benchmark_mean"]) > 0.0
            for value in directions.values()
        ),
        "both_label_swap_point_estimates_positive": len(swaps) == 2 and all(
            value.get("equal_weight_benchmark_mean") is not None
            and float(value["equal_weight_benchmark_mean"]) > 0.0
            for value in swaps.values()
        ),
        "all_six_benchmark_point_estimates_nonnegative": len(benchmarks) == 6 and all(
            value.get("equal_weight_benchmark_mean") is not None
            and float(value["equal_weight_benchmark_mean"]) >= 0.0
            for value in benchmarks.values()
        ),
        "matched_selected_logit_exact_identity": float(
            report.get("matched_selected_logit_max_error", math.inf)
        ) == 0.0,
        "external_zero_gate_exact_identity": float(
            report.get("audit", {}).get("zero_gate_max_error", math.inf)
        ) == 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("code_root", "execution_authorization", "model_path", "output_dir"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    authorization, paths = _validate_authorization(args)
    args.output_dir.mkdir(parents=True)
    protocol = json.loads(paths["protocol"].read_text())
    grid = json.loads(paths["candidate_grid"].read_text())
    grid_map = {row["candidate_id"]: row for row in grid["candidates"]}
    model_contract = json.loads(paths["model_manifest"].read_text())
    revision = model_contract.get("revision") or model_contract.get("lineage", {}).get(
        "weight_and_config_file_revision"
    )
    model_envelope = args.output_dir / "runtime_model_manifest.json"
    _write(model_envelope, {
        "schema_version": 1, "verified": True, "revision": revision,
        "source_model_contract_sha256": sha256_file(paths["model_manifest"]),
        "production_rollout_approved": False,
    })
    grouped: dict[str, list[dict]] = defaultdict(list)
    for record in authorization["candidates"]:
        grouped[record["fit_profile_id"]].append(record)
    if len(grouped) != 3 or any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("native behavior profiles must form three groups of three caps")

    summaries = []
    for profile_id in sorted(grouped):
        records = grouped[profile_id]
        contracts = {row["editor_contract"]["sha256"] for row in records}
        if len(contracts) != 1:
            raise ValueError("cap candidates do not share one fitted editor contract")
        contract = Path(records[0]["editor_contract"]["path"])
        profile_dir = args.output_dir / profile_id
        profile_dir.mkdir()
        compat_fit = profile_dir / "compatibility_fit_report.json"
        _compatibility_fit_report(records=records, contract=contract, output=compat_fit)
        for record in sorted(records, key=lambda row: row["candidate_id"]):
            candidate_id = record["candidate_id"]
            checkpoint = Path(record["checkpoint"]["path"])
            candidate_dir = profile_dir / candidate_id
            candidate_dir.mkdir()
            runtime_auth = candidate_dir / "runtime_authorization.json"
            _write(runtime_auth, {
                "schema_version": 1,
                "authorization_id": authorization["authorization_id"] + "-" + candidate_id,
                "stage": "governance_directional_fit_audit",
                "code_root": str(args.code_root),
                "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
                "execution_allowed": True,
                "checkpoint_sha256": sha256_file(checkpoint),
                "fit_report_sha256": sha256_file(compat_fit),
                "editor_contract_sha256": sha256_file(contract),
                "crossover_contract_sha256": sha256_file(paths["crossover_contract"]),
                "benchmark_manifest_sha256": sha256_file(paths["manifest"]),
                "design_audit_sha256": sha256_file(paths["design_audit"]),
                "model_manifest_sha256": sha256_file(model_envelope),
                "expected_rows": EXPECTED_ROWS,
                "expected_key_sha256": EXPECTED_KEY_SHA256,
                "final_test_open": False,
                "final_test_open_count": 0,
                "production_rollout_approved": False,
            })
            rows_path = candidate_dir / "behavior.jsonl"
            environment = candidate_dir / "behavior.environment.json"
            identity = candidate_dir / "behavior.identity.json"
            analysis = candidate_dir / "behavior.analysis.json"
            subprocess.run([
                sys.executable, "-m", "scripts.benchmark_v4.run_directional_governance",
                "--checkpoint", str(checkpoint), "--fit-report", str(compat_fit),
                "--editor-contract", str(contract), "--execution-authorization", str(runtime_auth),
                "--crossover-contract", str(paths["crossover_contract"]),
                "--manifest", str(paths["manifest"]), "--manifest-report", str(paths["manifest_report"]),
                "--design-audit", str(paths["design_audit"]), "--model-manifest", str(model_envelope),
                "--model-path", str(args.model_path), "--evaluation-split", "fit_audit",
                "--output", str(rows_path), "--environment-output", str(environment),
                "--identity-output", str(identity), "--device", args.device,
                "--attn-implementation", "eager",
            ], check=True, cwd=args.code_root)
            subprocess.run([
                sys.executable, "-m", "scripts.benchmark_v4.analyze_directional_governance",
                "--input", str(rows_path), "--environment", str(environment),
                "--identity-report", str(identity), "--editor-contract", str(contract),
                "--output", str(analysis),
            ], check=True, cwd=args.code_root)
            report = json.loads(analysis.read_text())
            checks = _behavior_checks(report, bool(record["fit_eligible"]))
            eligible = all(checks.values())
            config = grid_map[candidate_id]["config"]
            report.update({
                "governance_method": "C-DGE-V4.2",
                "candidate_id": candidate_id,
                "candidate_config": config,
                "fit_profile_id": profile_id,
                "native_behavior_checks": checks,
                "native_behavior_eligible": eligible,
                "native_protocol_sha256": sha256_file(paths["protocol"]),
                "candidate_grid_file_sha256": sha256_file(paths["candidate_grid"]),
                "formal_authorization_sha256": sha256_file(args.execution_authorization),
                "final_test_open": False,
                "final_test_open_count": 0,
                "production_rollout_approved": False,
            })
            _write(analysis, report, allow_overwrite=True)
            raw_ci = float(report["matched_minus_mismatched_reduction_bootstrap"]["ci95"][0])
            normalized_ci = float(report["normalized_gap_reduction_bootstrap"]["ci95"][0])
            summaries.append({
                "candidate_id": candidate_id,
                "candidate_config": config,
                "fit_profile_id": profile_id,
                "fit_eligible": bool(record["fit_eligible"]),
                "behavior_eligible": eligible,
                "checks": checks,
                "normalized_gap_reduction_lower_95": normalized_ci,
                "gap_reduction_lower_95": raw_ci,
                "analysis": str(analysis.relative_to(args.output_dir)),
                "analysis_sha256": sha256_file(analysis),
                "rows": str(rows_path.relative_to(args.output_dir)),
                "rows_sha256": sha256_file(rows_path),
            })
    total = sum(len(load_jsonl(args.output_dir / row["rows"])) for row in summaries)
    if len(summaries) != 9 or total != 9 * EXPECTED_ROWS:
        raise RuntimeError("native behavior shard output is incomplete")
    eligible_count = sum(row["behavior_eligible"] for row in summaries)
    shard = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_behavior_shard",
        "method": "C-DGE-V4.2",
        "site": authorization["site"],
        "behavior_complete": True,
        "candidate_count": 9,
        "rows_per_candidate": EXPECTED_ROWS,
        "total_rows": total,
        "expected_key_sha256_per_candidate": EXPECTED_KEY_SHA256,
        "behavior_eligible_candidate_count": eligible_count,
        "candidates": sorted(summaries, key=lambda row: row["candidate_id"]),
        "scientific_falsifier": (
            None if eligible_count else
            "No candidate at this model-native site passed the frozen fit, structural, directional, label-swap, benchmark, and exact-identity gates."
        ),
        "protected_controls_complete": False,
        "candidate_may_be_locked": False,
        "authorization_sha256": sha256_file(args.execution_authorization),
        "protocol_sha256": sha256_file(paths["protocol"]),
        "candidate_grid_file_sha256": sha256_file(paths["candidate_grid"]),
        "fit_shard_report_sha256": sha256_file(paths["fit_shard_report"]),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    _write(args.output_dir / "native_behavior_shard_report.json", shard)
    print(json.dumps({"site": shard["site"], "candidate_count": 9, "total_rows": total,
                      "behavior_eligible_candidate_count": eligible_count}, sort_keys=True))


if __name__ == "__main__":
    main()
