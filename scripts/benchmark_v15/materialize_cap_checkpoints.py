#!/usr/bin/env python3
"""Clone one native fit checkpoint across frozen correction caps without refit."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import validate_checkpoint_against_candidate


def _profile(value: dict, fit_profile_id: str) -> dict:
    matches = [
        row for row in value.get("profiles", [])
        if row.get("fit_profile_id") == fit_profile_id
    ]
    if len(matches) != 1:
        raise ValueError("fit profile ID is missing or duplicated")
    return matches[0]


def _candidate_map(value: dict) -> dict[str, dict]:
    candidates = value.get("candidates", [])
    result = {row.get("candidate_id"): row for row in candidates}
    if len(result) != len(candidates) or None in result:
        raise ValueError("candidate grid contains duplicate or missing IDs")
    return result


def materialize_cap_checkpoints(
    *,
    training_checkpoint: Path,
    candidate_grid_path: Path,
    fit_profiles_path: Path,
    fit_profile_id: str,
    output_dir: Path,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    candidate_grid = json.loads(candidate_grid_path.read_text())
    fit_profiles = json.loads(fit_profiles_path.read_text())
    if fit_profiles.get("candidate_grid_file_sha256") != sha256_file(candidate_grid_path):
        raise ValueError("fit profiles do not bind the candidate-grid file")
    if fit_profiles.get("fit_profile_count") != 9 or fit_profiles.get("candidate_count") != 27:
        raise ValueError("fit profile manifest is incomplete")
    profile = _profile(fit_profiles, fit_profile_id)
    candidates = _candidate_map(candidate_grid)
    training_candidate = candidates.get(profile["training_candidate_id"])
    if training_candidate is None:
        raise ValueError("training candidate is absent from the candidate grid")

    checkpoint = torch.load(training_checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("base_model_weights_included") is not False:
        raise ValueError("training checkpoint contains or does not exclude base weights")
    records = checkpoint.get("sites", [])
    if len(records) != 1:
        raise ValueError("training checkpoint must contain exactly one site")
    record = records[0]
    config = training_candidate["config"]
    for field in (
        "layer", "component", "boundary_rank", "context_rank",
        "positive_output_rank", "negative_output_rank",
    ):
        if record.get(field) != config[field]:
            raise ValueError(f"training checkpoint/profile mismatch: {field}")
    if float(record.get("maximum_relative_correction")) != float(profile["training_cap"]):
        raise ValueError("training checkpoint does not use the frozen training cap")
    if any(
        str(name).startswith(("model.", "transformer.", "base_model.", "lm_head."))
        for name in record.get("state_dict", {})
    ):
        raise ValueError("training checkpoint contains a base-weight key")

    output_dir.mkdir(parents=True)
    outputs = []
    for cap_record in profile["cap_candidates"]:
        candidate = candidates.get(cap_record["candidate_id"])
        if candidate is None:
            raise ValueError("cap candidate is absent from the candidate grid")
        cloned = copy.deepcopy(checkpoint)
        cloned["governance_method"] = "C-DGE-V4.2"
        cloned["candidate_id"] = candidate["candidate_id"]
        cloned["candidate_grid_sha256"] = candidate_grid["candidate_grid_sha256"]
        cloned["candidate_grid_file_sha256"] = sha256_file(candidate_grid_path)
        cloned["fit_profiles_file_sha256"] = sha256_file(fit_profiles_path)
        cloned["fit_profile_id"] = fit_profile_id
        cloned["training_checkpoint_sha256"] = sha256_file(training_checkpoint)
        cloned["cap_materialized_without_refit"] = True
        cloned["sites"][0]["maximum_relative_correction"] = float(
            cap_record["max_relative_correction"]
        )
        validate_checkpoint_against_candidate(
            cloned,
            candidate,
            candidate_grid_sha256=candidate_grid["candidate_grid_sha256"],
        )
        filename = candidate["candidate_id"] + ".pt"
        path = output_dir / filename
        incoming = output_dir / ("." + filename + ".incoming")
        torch.save(cloned, incoming)
        os.replace(incoming, path)
        outputs.append({
            "candidate_id": candidate["candidate_id"],
            "max_relative_correction": float(
                candidate["config"]["max_relative_correction"]
            ),
            "checkpoint": filename,
            "checkpoint_sha256": sha256_file(path),
        })
    outputs.sort(key=lambda row: row["candidate_id"])
    manifest = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_cap_checkpoint_materialization",
        "method": "C-DGE-V4.2",
        "fit_profile_id": fit_profile_id,
        "training_checkpoint_sha256": sha256_file(training_checkpoint),
        "candidate_grid_file_sha256": sha256_file(candidate_grid_path),
        "fit_profiles_file_sha256": sha256_file(fit_profiles_path),
        "candidate_count": len(outputs),
        "checkpoints": outputs,
        "cap_materialized_without_refit": True,
        "base_model_weights_included": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    manifest_path = output_dir / "cap_checkpoint_manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-checkpoint", type=Path, required=True)
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--fit-profiles", type=Path, required=True)
    parser.add_argument("--fit-profile-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    value = materialize_cap_checkpoints(
        training_checkpoint=args.training_checkpoint,
        candidate_grid_path=args.candidate_grid,
        fit_profiles_path=args.fit_profiles,
        fit_profile_id=args.fit_profile_id,
        output_dir=args.output_dir,
    )
    print(json.dumps({
        "fit_profile_id": value["fit_profile_id"],
        "candidate_count": value["candidate_count"],
        "manifest_sha256": sha256_file(args.output_dir / "cap_checkpoint_manifest.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
