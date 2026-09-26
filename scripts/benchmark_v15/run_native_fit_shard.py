#!/usr/bin/env python3
"""Fit three rank profiles for one locked V4.2 native site, once at cap 0.075."""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import validate_checkpoint_against_candidate
from scripts.benchmark_v15.materialize_cap_checkpoints import (
    materialize_cap_checkpoints,
)


PROFILE_ORDER = {"compact": 0, "balanced": 1, "wide": 2}


def _write(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _source_matches(record: dict, path: Path) -> bool:
    return record.get("path") == str(path) and record.get("sha256") == sha256_file(path)


def _require_authorization(args: argparse.Namespace) -> dict:
    value = json.loads(args.execution_authorization.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_fit_shard",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True,
        "site": args.site,
        "candidate_grid_sha256": json.loads(args.candidate_grid.read_text())[
            "candidate_grid_sha256"
        ],
        "fit_profile_count": 3,
        "training_candidate_count": 3,
        "materialized_candidate_count": 9,
        "fit_once_at_maximum_cap": 0.075,
        "replication_protocol_sha256": sha256_file(args.protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "v3_template_sha256": sha256_file(args.v3_template),
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"native fit authorization mismatch: {field}")
    for field, path in (
        ("site_manifest", args.site_manifest),
        ("candidate_grid", args.candidate_grid),
        ("fit_profiles", args.fit_profiles),
    ):
        if not _source_matches(value.get(field, {}), path):
            raise ValueError(f"native fit authorization source mismatch: {field}")
    for stage, path in (
        ("governance", args.capture_manifest),
        ("gradient", args.gradient_manifest),
        ("protected", args.protected_manifest),
    ):
        if not _source_matches(value.get("capture_manifests", {}).get(stage, {}), path):
            raise ValueError(f"native fit capture source mismatch: {stage}")
        receipt = value.get("capture_archive_receipts", {}).get(stage, {})
        if receipt.get("three_copy_verified") is not True:
            raise ValueError(f"native fit capture receipt is not closed: {stage}")
    profiles = value.get("fit_profile_ids", [])
    if len(profiles) != 3 or len(set(profiles)) != 3:
        raise ValueError("native fit authorization profile set is invalid")
    return value


def _site_key(row: dict) -> str:
    return f"{int(row['layer'])}:{row['component']}"


def _profiles_for_site(path: Path, site: str, authorized: list[str]) -> list[dict]:
    value = json.loads(path.read_text())
    profiles = [row for row in value.get("profiles", []) if _site_key(row["site"]) == site]
    profiles.sort(key=lambda row: PROFILE_ORDER[row["rank_profile"]])
    if len(profiles) != 3 or [row["fit_profile_id"] for row in profiles] != authorized:
        raise ValueError("fit-profile file differs from the authorized shard")
    return profiles


def _candidate_map(path: Path) -> tuple[dict, dict[str, dict]]:
    grid = json.loads(path.read_text())
    candidates = {row["candidate_id"]: row for row in grid.get("candidates", [])}
    if len(candidates) != 27:
        raise ValueError("native candidate grid is incomplete")
    return grid, candidates


def derive_directional_contract(
    *,
    template: dict,
    protocol: dict,
    profile: dict,
    training_candidate: dict,
    args: argparse.Namespace,
    created_utc: str,
) -> dict:
    config = training_candidate["config"]
    if _site_key(config) != args.site:
        raise ValueError("training candidate belongs to another site")
    if float(config["max_relative_correction"]) != 0.075:
        raise ValueError("native fit must train the maximum cap exactly once")
    contract = copy.deepcopy(template)
    contract.update({
        "contract_id": (
            "context-mismatch-qwen3-5-9b-cdge-v4-2-native-directional-"
            f"{profile['fit_profile_id'].lower()}"
        ),
        "created_utc": created_utc,
        "frozen_utc": created_utc,
        "status": "native_discovery_locked_before_candidate_fit",
        "method_name": "Model-Native Directional Structural Governance Editor",
        "scientific_change_from_v2": (
            "model-native locked site and rank profile; fit once at trust-region cap "
            "0.075 and clone unchanged weights for smaller-cap behavior evaluation"
        ),
        "candidate_sites": [{
            "layer": int(config["layer"]),
            "component": str(config["component"]),
            "boundary_rank": int(config["boundary_rank"]),
            "context_rank": int(config["context_rank"]),
            "positive_output_rank": int(config["positive_output_rank"]),
            "negative_output_rank": int(config["negative_output_rank"]),
            "max_relative_correction": float(config["max_relative_correction"]),
        }],
        "fit_split": copy.deepcopy(protocol["native_discovery"]["fit_split"]),
        "native_discovery_lineage": {
            "site_manifest_sha256": sha256_file(args.site_manifest),
            "candidate_grid_file_sha256": sha256_file(args.candidate_grid),
            "candidate_grid_sha256": json.loads(args.candidate_grid.read_text())[
                "candidate_grid_sha256"
            ],
            "fit_profiles_file_sha256": sha256_file(args.fit_profiles),
            "fit_profile_id": profile["fit_profile_id"],
            "fit_profile_sha256": profile["fit_profile_sha256"],
            "training_candidate_id": training_candidate["candidate_id"],
            "rank_profile": profile["rank_profile"],
            "training_cap": 0.075,
        },
    })
    contract["bound_inputs"] = copy.deepcopy(template["bound_inputs"])
    contract["bound_inputs"].update({
        "governance_capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_manifest),
        "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
        "native_site_manifest_sha256": sha256_file(args.site_manifest),
        "native_candidate_grid_sha256": sha256_file(args.candidate_grid),
        "native_fit_profiles_sha256": sha256_file(args.fit_profiles),
    })
    contract["architecture"] = copy.deepcopy(template["architecture"])
    contract["architecture"].update({
        "maximum_relative_correction_per_candidate": 0.075,
        "single_selected_site_per_checkpoint": True,
    })
    contract["optimization"] = copy.deepcopy(template["optimization"])
    contract["optimization"]["three_single_site_candidates_fit_independently"] = False
    contract["optimization"]["one_site_rank_profile_fit_once_at_maximum_cap"] = True
    contract["fit_gates_per_candidate_without_cross_site_averaging"] = copy.deepcopy(
        template["fit_gates_per_candidate_without_cross_site_averaging"]
    )
    contract["fit_gates_per_candidate_without_cross_site_averaging"][
        "maximum_observed_relative_correction"
    ] = 0.075001
    return contract


def _directional_authorization(
    *, contract_path: Path, args: argparse.Namespace
) -> dict:
    return {
        "schema_version": 1,
        "stage": "governance_directional_fit",
        "execution_allowed": True,
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "editor_contract_sha256": sha256_file(contract_path),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_manifest),
        "gradient_capture_manifest_sha256": sha256_file(args.gradient_manifest),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def _bind_training_checkpoint(
    *, source: Path, output: Path, profile: dict, candidate: dict,
    candidate_grid: dict, args: argparse.Namespace,
) -> None:
    value = torch.load(source, map_location="cpu", weights_only=False)
    value.update({
        "governance_method": "C-DGE-V4.2",
        "candidate_id": candidate["candidate_id"],
        "candidate_grid_sha256": candidate_grid["candidate_grid_sha256"],
        "candidate_grid_file_sha256": sha256_file(args.candidate_grid),
        "fit_profiles_file_sha256": sha256_file(args.fit_profiles),
        "fit_profile_id": profile["fit_profile_id"],
        "fit_profile_sha256": profile["fit_profile_sha256"],
        "cap_materialized_without_refit": False,
    })
    validate_checkpoint_against_candidate(
        value, candidate,
        candidate_grid_sha256=candidate_grid["candidate_grid_sha256"],
    )
    incoming = output.parent / ("." + output.name + ".incoming")
    torch.save(value, incoming)
    os.replace(incoming, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "code_root", "protocol", "crossover_contract", "v3_template",
        "site_manifest", "candidate_grid", "fit_profiles", "capture_manifest",
        "gradient_manifest", "protected_manifest", "execution_authorization",
        "output_dir",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--site", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    authorization = _require_authorization(args)
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("method_short_name") != "C-DGE-V4.2":
        raise ValueError("unexpected native-fit protocol")
    profiles = _profiles_for_site(
        args.fit_profiles, args.site, authorization["fit_profile_ids"]
    )
    grid, candidates = _candidate_map(args.candidate_grid)
    template = json.loads(args.v3_template.read_text())
    args.output_dir.mkdir(parents=True)
    reports = []
    for profile in profiles:
        training = candidates[profile["training_candidate_id"]]
        profile_dir = args.output_dir / profile["fit_profile_id"]
        profile_dir.mkdir()
        contract_path = profile_dir / "directional_contract.json"
        contract = derive_directional_contract(
            template=template,
            protocol=protocol,
            profile=profile,
            training_candidate=training,
            args=args,
            created_utc=str(authorization["created_utc"]),
        )
        _write(contract_path, contract)
        directional_authorization_path = profile_dir / "directional_fit_authorization.json"
        _write(
            directional_authorization_path,
            _directional_authorization(contract_path=contract_path, args=args),
        )
        directional_output = profile_dir / "directional_fit"
        subprocess.run([
            sys.executable, "-m", "scripts.benchmark_v4.fit_directional_governance",
            "--editor-contract", str(contract_path),
            "--capture-manifest", str(args.capture_manifest),
            "--protected-capture-manifest", str(args.protected_manifest),
            "--gradient-capture-manifest", str(args.gradient_manifest),
            "--execution-authorization", str(directional_authorization_path),
            "--output-dir", str(directional_output),
        ], check=True, cwd=args.code_root)
        fit_report_path = directional_output / "directional_fit_report.json"
        fit_report = json.loads(fit_report_path.read_text())
        rows = fit_report.get("candidate_reports", [])
        if fit_report.get("candidate_count") != 1 or len(rows) != 1:
            raise ValueError("native profile fit did not produce exactly one checkpoint")
        raw_checkpoint = directional_output / rows[0]["checkpoint"]
        if sha256_file(raw_checkpoint) != rows[0].get("checkpoint_sha256"):
            raise ValueError("native profile raw checkpoint SHA mismatch")
        bound_checkpoint = profile_dir / "native_training_checkpoint.pt"
        _bind_training_checkpoint(
            source=raw_checkpoint,
            output=bound_checkpoint,
            profile=profile,
            candidate=training,
            candidate_grid=grid,
            args=args,
        )
        cap_dir = profile_dir / "cap_checkpoints"
        cap_manifest = materialize_cap_checkpoints(
            training_checkpoint=bound_checkpoint,
            candidate_grid_path=args.candidate_grid,
            fit_profiles_path=args.fit_profiles,
            fit_profile_id=profile["fit_profile_id"],
            output_dir=cap_dir,
        )
        reports.append({
            "fit_profile_id": profile["fit_profile_id"],
            "rank_profile": profile["rank_profile"],
            "training_candidate_id": training["candidate_id"],
            "training_cap": 0.075,
            "directional_contract": str(contract_path.relative_to(args.output_dir)),
            "directional_contract_sha256": sha256_file(contract_path),
            "directional_fit_report": str(fit_report_path.relative_to(args.output_dir)),
            "directional_fit_report_sha256": sha256_file(fit_report_path),
            "training_checkpoint": str(bound_checkpoint.relative_to(args.output_dir)),
            "training_checkpoint_sha256": sha256_file(bound_checkpoint),
            "cap_checkpoint_manifest": str(
                (cap_dir / "cap_checkpoint_manifest.json").relative_to(args.output_dir)
            ),
            "cap_checkpoint_manifest_sha256": sha256_file(
                cap_dir / "cap_checkpoint_manifest.json"
            ),
            "materialized_candidate_count": cap_manifest["candidate_count"],
            "fit_eligible": bool(rows[0].get("fit_eligible")),
            "direct_behavior_characterization_required_even_if_ineligible": True,
        })
    report = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_fit_shard",
        "method": "C-DGE-V4.2",
        "site": args.site,
        "fit_complete": True,
        "fit_profile_count": len(reports),
        "training_run_count": len(reports),
        "training_caps": sorted({row["training_cap"] for row in reports}),
        "materialized_candidate_count": sum(
            row["materialized_candidate_count"] for row in reports
        ),
        "fit_once_per_site_rank_profile": True,
        "cap_checkpoints_share_fitted_weights": True,
        "profiles": reports,
        "candidate_grid_sha256": grid["candidate_grid_sha256"],
        "candidate_grid_file_sha256": sha256_file(args.candidate_grid),
        "fit_profiles_file_sha256": sha256_file(args.fit_profiles),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if report["fit_profile_count"] != 3 or report["materialized_candidate_count"] != 9:
        raise RuntimeError("native fit shard did not materialize its complete grid slice")
    _write(args.output_dir / "native_fit_shard_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
