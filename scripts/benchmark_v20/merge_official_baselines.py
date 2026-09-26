#!/usr/bin/env python3
"""Merge input controls with terminal official-code-derived baseline shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


OFFICIAL_METHODS = {"caa", "cast", "loreft", "reps"}
INPUT_METHODS = {"governance_reset_prompt", "session_isolation"}
OFFICIAL_PROVENANCE = {
    "official_code_derived_task_adaptation": True,
    "adapter_only": False,
    "unmodified_official_implementation": False,
}


def _load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-control-shard", type=Path, required=True)
    parser.add_argument("--activation-shard", type=Path, required=True)
    parser.add_argument("--representation-shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing existing official-derived comparison report")

    paths = {
        "input": args.input_control_shard,
        "activation": args.activation_shard,
        "representation": args.representation_shard,
    }
    shards = {name: _load(path) for name, path in paths.items()}
    expected = {
        "input": INPUT_METHODS,
        "activation": {"caa", "cast"},
        "representation": {"loreft", "reps"},
    }
    for group, shard in shards.items():
        if shard.get("group") != group or set(shard.get("methods", {})) != expected[group]:
            raise ValueError(f"incomplete {group} shard")
        if shard.get("rows_per_method") != 3072 or shard.get("scientific_audit_complete") is not True:
            raise ValueError(f"unaudited {group} shard")
        for key, value in {
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }.items():
            if shard.get(key) != value:
                raise ValueError(f"{group} shard safety mismatch: {key}")

    for group in ("activation", "representation"):
        if shards[group].get("stage") != "qwen3_8b_external_baseline_official_evaluation":
            raise ValueError(f"{group} shard is not official-derived")
        for method in expected[group]:
            analysis = _load(paths[group].parent / f"{method}.analysis.json")
            for key, value in OFFICIAL_PROVENANCE.items():
                if analysis.get(key) != value:
                    raise ValueError(f"{method} provenance mismatch: {key}")

    baseline_digests = {shard["baseline_logits_sha256"] for shard in shards.values()}
    if len(baseline_digests) != 1:
        raise RuntimeError("baseline logits are not bit-identical across all methods")
    methods = {}
    for shard in shards.values():
        overlap = set(methods) & set(shard["methods"])
        if overlap:
            raise ValueError(f"duplicate methods: {sorted(overlap)}")
        methods.update(shard["methods"])
    if set(methods) != INPUT_METHODS | OFFICIAL_METHODS:
        raise ValueError("formal comparison method set is incomplete")

    result = {
        "schema_version": 2,
        "stage": "qwen3_8b_external_baseline_official_derived_comparison",
        "terminology": "official-code-derived task adaptations",
        "methods": methods,
        "method_count": len(methods),
        "rows_per_method": 3072,
        "total_method_rows": 3072 * len(methods),
        "baseline_logits_sha256": next(iter(baseline_digests)),
        "baseline_logits_bit_identical_across_all_methods": True,
        "official_code_derived_methods": sorted(OFFICIAL_METHODS),
        "input_control_methods": sorted(INPUT_METHODS),
        "adapter_only_results_included": False,
        "input_shards": [
            {"group": group, "path": str(paths[group]), "sha256": sha256_file(paths[group])}
            for group in ("input", "activation", "representation")
        ],
        "scientific_audit_complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
