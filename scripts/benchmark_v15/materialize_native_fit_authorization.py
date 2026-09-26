#!/usr/bin/env python3
"""Materialize one SHA-bound C-DGE V4.2 native fit-shard authorization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import value_sha256


EXPECTED_KEY_SHA256 = (
    "488835b013ebdf7413d29fa3490bf937700e1b542bac9b0539aa31ba966dcd35"
)
EXPECTED_PROTECTED_FAMILIES = {
    "fresh": 768,
    "verification": 1536,
    "obedience_reset": 1536,
    "supported_user_authority": 144,
    "factual_boundary_memory": 24,
}


def _source(path: Path) -> dict:
    return {"path": str(path), "sha256": sha256_file(path)}


def _site_key(row: dict) -> str:
    return f"{int(row['layer'])}:{row['component']}"


def _validate_site_manifest(path: Path, requested_site: str) -> tuple[dict, list[str]]:
    value = json.loads(path.read_text())
    for field, expected in {
        "stage": "qwen35_cdge_v4_2_native_site_selection",
        "method": "C-DGE-V4.2",
        "selection_partition": "subspace_fit",
        "locked": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"site-manifest mismatch: {field}")
    sites = [_site_key(row) for row in value.get("selected_sites_ordered", [])]
    if len(sites) != 3 or len(set(sites)) != 3:
        raise ValueError("site manifest must lock exactly three unique sites")
    if requested_site not in sites:
        raise ValueError("requested fit site is not locked by native discovery")
    return value, sites


def _validate_grid(path: Path, site_manifest: dict) -> dict:
    value = json.loads(path.read_text())
    declared = value.get("candidate_grid_sha256")
    unhashed = dict(value)
    unhashed.pop("candidate_grid_sha256", None)
    if declared != value_sha256(unhashed):
        raise ValueError("candidate-grid semantic SHA mismatch")
    for field, expected in {
        "stage": "qwen35_cdge_v4_2_native_candidate_grid",
        "method": "C-DGE-V4.2",
        "candidate_count": 27,
        "site_manifest_sha256": value_sha256(site_manifest),
        "selection_performed": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"candidate-grid mismatch: {field}")
    candidates = value.get("candidates", [])
    if len(candidates) != 27 or len({row.get("candidate_id") for row in candidates}) != 27:
        raise ValueError("candidate grid is incomplete or duplicated")
    return value


def _validate_profiles(path: Path, grid_path: Path, grid: dict, site: str) -> list[dict]:
    value = json.loads(path.read_text())
    declared = value.get("fit_profiles_sha256")
    unhashed = dict(value)
    unhashed.pop("fit_profiles_sha256", None)
    # The file SHA is appended after the semantic manifest is created.
    unhashed.pop("candidate_grid_file_sha256", None)
    if declared != value_sha256(unhashed):
        raise ValueError("fit-profile semantic SHA mismatch")
    for field, expected in {
        "stage": "qwen35_cdge_v4_2_native_fit_profiles",
        "method": "C-DGE-V4.2",
        "candidate_grid_sha256": grid["candidate_grid_sha256"],
        "candidate_grid_file_sha256": sha256_file(grid_path),
        "fit_profile_count": 9,
        "candidate_count": 27,
        "fit_once_per_site_rank_profile": True,
        "cap_evaluation_requires_checkpoint_clone_without_refit": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"fit-profile mismatch: {field}")
    profiles = [row for row in value.get("profiles", []) if _site_key(row["site"]) == site]
    if len(profiles) != 3 or {row.get("rank_profile") for row in profiles} != {
        "compact", "balanced", "wide"
    }:
        raise ValueError("fit shard must contain compact, balanced, and wide profiles")
    if any(float(row.get("training_cap", -1)) != 0.075 for row in profiles):
        raise ValueError("every profile must train exactly once at cap 0.075")
    if sum(len(row.get("cap_candidates", [])) for row in profiles) != 9:
        raise ValueError("fit shard must materialize nine cap candidates")
    return sorted(profiles, key=lambda row: ("compact", "balanced", "wide").index(row["rank_profile"]))


def _validate_capture(stage: str, path: Path, sites: list[str]) -> None:
    value = json.loads(path.read_text())
    expected_rows = {"governance": 6144, "gradient": 6144, "protected": 4008}[stage]
    for field, expected in {
        "stage": f"qwen35_cdge_v4_2_{stage}_capture",
        "partition": "subspace_fit",
        "rows": expected_rows,
        "complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if value.get(field) != expected:
            raise ValueError(f"{stage} capture mismatch: {field}")
    manifest_sites = sorted(_site_key(row) for row in value.get("sites", []))
    if manifest_sites != sorted(sites):
        raise ValueError(f"{stage} capture site set mismatch")
    if stage in {"governance", "gradient"}:
        if value.get("unique_job_keys") != 6144:
            raise ValueError(f"{stage} unique-key count mismatch")
        if value.get("expected_key_sha256") != EXPECTED_KEY_SHA256:
            raise ValueError(f"{stage} expected key SHA mismatch")
        if value.get("observed_key_sha256") != EXPECTED_KEY_SHA256:
            raise ValueError(f"{stage} observed key SHA mismatch")
    if stage == "gradient" and value.get("base_model_parameter_gradients") is not False:
        raise ValueError("gradient capture permits base-model gradients")
    if stage == "protected" and value.get("family_rows") != EXPECTED_PROTECTED_FAMILIES:
        raise ValueError("protected family counts mismatch")


def _validate_receipt(stage: str, path: Path) -> dict:
    value = json.loads(path.read_text())
    run_id = str(value.get("run_id", ""))
    if not re.fullmatch(
        rf"qwen3-5-9b-cdge-v4-2-{stage}-capture-[0-9]{{8}}T[0-9]{{6}}Z",
        run_id,
    ):
        raise ValueError(f"{stage} receipt run mismatch")
    for field in (
        "cluster_shared_copy_verified",
        "host_data_copy_verified",
        "local_copy_verified",
    ):
        if value.get(field) is not True:
            raise ValueError(f"{stage} receipt is not three-copy verified: {field}")
    archive_sha = str(value.get("archive_sha256", ""))
    if not re.fullmatch(r"[0-9a-f]{64}", archive_sha):
        raise ValueError(f"{stage} archive SHA is invalid")
    terminal = str(value.get("slurm_terminal_record", ""))
    if "JobState=COMPLETED" not in terminal or "ExitCode=0:0" not in terminal:
        raise ValueError(f"{stage} receipt lacks terminal Slurm provenance")
    return {
        "run_id": run_id,
        "job_id": int(value["job_id"]),
        "archive_sha256": archive_sha,
        "receipt_sha256": sha256_file(path),
        "three_copy_verified": True,
    }


def materialize(args: argparse.Namespace) -> dict:
    allowed_code_roots = {
        "/workspace/context-mismatch-qwen3-5-9b/code-v91",
        "/workspace/context-mismatch-qwen3-5-9b/code-v95",
        "/workspace/context-mismatch-qwen3-5-9b/code-v96",
        "/workspace/context-mismatch-qwen3-5-9b/code-v97",
    }
    if str(args.code_root) not in allowed_code_roots:
        raise ValueError("approved immutable native-fit code root required")
    code_version = args.code_root.name
    if not re.fullmatch(r"[0-9]+:(self_attn|mlp)", args.site):
        raise ValueError("invalid native site key")
    snapshot = json.loads(args.selector_snapshot.read_text())
    if snapshot.get("node") != args.selected_node:
        raise ValueError("selector snapshot node mismatch")
    selection = snapshot.get("selection_contract", {})
    if selection.get("npus") != 0 or selection.get("cpus") != 32:
        raise ValueError("fit selector CPU/NPU contract mismatch")
    if selection.get("mem_mib") != 196608:
        raise ValueError("fit selector memory contract mismatch")
    protocol = json.loads(args.protocol.read_text())
    for field, expected in {
        "method_short_name": "C-DGE-V4.2",
        "status": "frozen_before_qwen3_5_native_discovery_forward",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if protocol.get(field) != expected:
            raise ValueError(f"protocol mismatch: {field}")
    site_manifest, sites = _validate_site_manifest(args.site_manifest, args.site)
    grid = _validate_grid(args.candidate_grid, site_manifest)
    profiles = _validate_profiles(args.fit_profiles, args.candidate_grid, grid, args.site)
    captures = {
        "governance": args.governance_manifest,
        "gradient": args.gradient_manifest,
        "protected": args.protected_manifest,
    }
    receipts = {
        "governance": args.governance_receipt,
        "gradient": args.gradient_receipt,
        "protected": args.protected_receipt,
    }
    for stage, path in captures.items():
        _validate_capture(stage, path, sites)
    receipt_values = {
        stage: _validate_receipt(stage, path) for stage, path in receipts.items()
    }
    created = args.created_utc or dt.datetime.now(dt.timezone.utc).replace(
        microsecond=0
    ).isoformat().replace("+00:00", "Z")
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-5-9b-cdge-v4-2-native-fit-{args.site.replace(':', '-')}-{code_version}",
        "created_utc": created,
        "stage": "qwen35_cdge_v4_2_native_fit_shard",
        "method": "C-DGE-V4.2",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_node": args.selected_node,
        "slurm_partition": "a01",
        "execution_allowed": True,
        "site": args.site,
        "site_manifest": _source(args.site_manifest),
        "candidate_grid": _source(args.candidate_grid),
        "candidate_grid_sha256": grid["candidate_grid_sha256"],
        "fit_profiles": _source(args.fit_profiles),
        "fit_profile_ids": [row["fit_profile_id"] for row in profiles],
        "fit_profile_count": 3,
        "training_candidate_count": 3,
        "materialized_candidate_count": 9,
        "fit_once_at_maximum_cap": 0.075,
        "replication_protocol_sha256": sha256_file(args.protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "v3_template_sha256": sha256_file(args.v3_template),
        "capture_manifests": {stage: _source(path) for stage, path in captures.items()},
        "capture_archive_receipts": receipt_values,
        "node_selection_snapshot": _source(args.selector_snapshot),
        "monitoring_contract": {
            "continuous_watch_required": True,
            "poll_seconds": 1,
            "archive_helper": _source(args.archive_helper),
            "archive_coordinator": _source(args.archive_coordinator),
            "pull_helper": _source(args.pull_helper),
        },
        "resource_contract": {
            "partition": "a01", "nodes": 1, "ntasks": 1,
            "cpus_per_task": 32, "mem_mib": 196608,
            "npu_type": "910B3", "npus": 0,
            "time_limit": "08:00:00", "node": args.selected_node,
        },
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "code_root", "selector_snapshot", "protocol", "crossover_contract",
        "v3_template", "site_manifest", "candidate_grid", "fit_profiles",
        "governance_manifest", "gradient_manifest", "protected_manifest",
        "governance_receipt", "gradient_receipt", "protected_receipt",
        "archive_helper", "archive_coordinator", "pull_helper", "output",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--selected-node", required=True)
    parser.add_argument("--site", required=True)
    parser.add_argument("--created-utc")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    value = materialize(args)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "site": value["site"],
        "fit_profile_count": value["fit_profile_count"],
        "materialized_candidate_count": value["materialized_candidate_count"],
        "authorization_sha256": sha256_file(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
