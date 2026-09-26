#!/usr/bin/env python3
"""Freeze one-fit/many-cap profiles from a V4.2 native candidate grid."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v14.native_discovery import value_sha256


def _value_sha256(value: object) -> str:
    return value_sha256(value)


def materialize_fit_profiles(candidate_grid: dict) -> dict:
    required = {
        "stage": "qwen35_cdge_v4_2_native_candidate_grid",
        "method": "C-DGE-V4.2",
        "candidate_count": 27,
        "selection_performed": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if candidate_grid.get(field) != expected:
            raise ValueError(f"candidate-grid mismatch: {field}")
    declared_grid_sha = candidate_grid.get("candidate_grid_sha256")
    unhashed_grid = dict(candidate_grid)
    unhashed_grid.pop("candidate_grid_sha256", None)
    if declared_grid_sha != _value_sha256(unhashed_grid):
        raise ValueError("candidate-grid semantic SHA mismatch")
    candidates = candidate_grid.get("candidates", [])
    if len(candidates) != 27 or len({row.get("candidate_id") for row in candidates}) != 27:
        raise ValueError("candidate grid must contain 27 unique candidates")

    groups: dict[tuple, list[dict]] = defaultdict(list)
    identity_fields = (
        "layer", "component", "rank_profile", "boundary_rank", "context_rank",
        "positive_output_rank", "negative_output_rank", "protected_rank",
    )
    for candidate in candidates:
        config = candidate.get("config", {})
        if not isinstance(candidate.get("candidate_id"), str):
            raise ValueError("candidate ID is missing")
        try:
            key = tuple(config[field] for field in identity_fields)
            cap = float(config["max_relative_correction"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("candidate config is incomplete") from error
        if cap <= 0.0:
            raise ValueError("candidate correction cap must be positive")
        groups[key].append(candidate)

    profiles = []
    for key, rows in groups.items():
        ordered = sorted(rows, key=lambda row: float(row["config"]["max_relative_correction"]))
        caps = [float(row["config"]["max_relative_correction"]) for row in ordered]
        if caps != [0.025, 0.05, 0.075]:
            raise ValueError(f"fit profile has an unexpected cap set: {caps}")
        common = dict(zip(identity_fields, key))
        training = ordered[-1]
        record = {
            "fit_profile_id": "CDGE42FIT-" + _value_sha256(common)[:16],
            "site": {"layer": int(common["layer"]), "component": common["component"]},
            "rank_profile": common["rank_profile"],
            "rank_config": {
                field: int(common[field])
                for field in identity_fields
                if field.endswith("_rank")
            },
            "training_cap": caps[-1],
            "training_candidate_id": training["candidate_id"],
            "cap_candidates": [
                {
                    "candidate_id": row["candidate_id"],
                    "max_relative_correction": float(
                        row["config"]["max_relative_correction"]
                    ),
                }
                for row in ordered
            ],
        }
        record["fit_profile_sha256"] = _value_sha256(record)
        profiles.append(record)
    profiles.sort(key=lambda row: row["fit_profile_id"])
    if len(profiles) != 9 or len({row["fit_profile_id"] for row in profiles}) != 9:
        raise ValueError("candidate grid must produce nine unique fit profiles")
    if {row["training_candidate_id"] for row in profiles} != {
        row["candidate_id"]
        for row in candidates
        if float(row["config"]["max_relative_correction"]) == 0.075
    }:
        raise ValueError("training candidates do not match the maximum-cap grid slice")

    value = {
        "schema_version": 1,
        "stage": "qwen35_cdge_v4_2_native_fit_profiles",
        "method": "C-DGE-V4.2",
        "candidate_grid_sha256": candidate_grid["candidate_grid_sha256"],
        "fit_profile_count": len(profiles),
        "candidate_count": len(candidates),
        "fit_once_per_site_rank_profile": True,
        "cap_evaluation_requires_checkpoint_clone_without_refit": True,
        "profiles": profiles,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    value["fit_profiles_sha256"] = _value_sha256(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-grid", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    candidate_grid = json.loads(args.candidate_grid.read_text())
    value = materialize_fit_profiles(candidate_grid)
    value["candidate_grid_file_sha256"] = sha256_file(args.candidate_grid)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "fit_profile_count": value["fit_profile_count"],
        "candidate_count": value["candidate_count"],
        "output_sha256": sha256_file(args.output),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
