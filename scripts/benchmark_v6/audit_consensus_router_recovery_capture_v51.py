#!/usr/bin/env python3
"""Strict finite/identity audit for V5.1 recovery capture shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


EXPECTED_KEY = "4e6301fce193060c68bb284f7b3349aec61766eb1daa60ff85d200ef9d981900"
EXPECTED_FAMILIES = {
    "fresh": 768,
    "verification": 1536,
    "obedience_reset": 1536,
    "supported_user_authority": 144,
    "factual_boundary_memory": 24,
}


def _finite_tensors(value: object, label: str) -> int:
    count = 0
    if isinstance(value, torch.Tensor):
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError(f"non-finite tensor: {label}")
        return int(value.numel())
    if isinstance(value, dict):
        for key, child in value.items():
            count += _finite_tensors(child, f"{label}.{key}")
    return count


def _load_and_verify_shards(manifest_path: Path) -> tuple[list[dict], int]:
    manifest = json.loads(manifest_path.read_text())
    metadata: list[dict] = []
    tensor_values = 0
    for record in manifest.get("shards", []):
        shard = manifest_path.parent / record["file"]
        if not shard.is_file() or sha256_file(shard) != record.get("sha256"):
            raise ValueError(f"shard SHA mismatch: {shard}")
        value = torch.load(shard, map_location="cpu", weights_only=False)
        rows = value.get("metadata")
        if not isinstance(rows, list) or len(rows) != int(record.get("rows", -1)):
            raise ValueError(f"shard metadata row mismatch: {shard}")
        metadata.extend(rows)
        for key in ("boundary_states", "component_inputs", "component_outputs"):
            tensor_values += _finite_tensors(value.get(key), f"{shard.name}.{key}")
        del value
    return metadata, tensor_values


def _audit_governance(manifest_path: Path, expected_authorization_sha256: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    required = {
        "complete": True,
        "stage": "governance_consensus_router_audit_capture",
        "partition": "component_discovery",
        "rows": 6144,
        "unique_job_keys": 6144,
        "counterfactual_pairs": 3072,
        "shard_count": 24,
        "expected_key_sha256": EXPECTED_KEY,
        "observed_key_sha256": EXPECTED_KEY,
        "authorization_sha256": expected_authorization_sha256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"governance manifest mismatch: {field}")
    metadata, tensor_values = _load_and_verify_shards(manifest_path)
    keys = [str(row.get("job_key", "")) for row in metadata]
    if len(keys) != 6144 or len(set(keys)) != 6144 or "" in keys:
        raise ValueError("governance job-key identity mismatch")
    observed = hashlib.sha256(("\n".join(sorted(keys)) + "\n").encode()).hexdigest()
    if observed != EXPECTED_KEY:
        raise ValueError("governance observed key SHA mismatch")
    pairs: dict[tuple, set[str]] = defaultdict(set)
    for row in metadata:
        for field in ("logit_a", "logit_b", "task_aligned_margin", "factual_margin"):
            if not math.isfinite(float(row[field])):
                raise FloatingPointError(f"non-finite governance metadata: {field}")
        pairs[(
            row["item_id"], row["declared_role"], row["history_style"],
            int(row["history_realization"]), row["task_requirement"],
            int(row["label_swap"]),
        )].add(row["history_condition"])
    if len(pairs) != 3072 or any(v != {"verification", "obedience"} for v in pairs.values()):
        raise ValueError("governance counterfactual pair mismatch")
    return {"rows": 6144, "unique_job_keys": 6144, "counterfactual_pairs": 3072,
            "tensor_values_checked": tensor_values, "all_finite": True}


def _audit_protected(manifest_path: Path, expected_authorization_sha256: str) -> dict:
    manifest = json.loads(manifest_path.read_text())
    required = {
        "stage": "governance_consensus_router_audit_protected_capture",
        "partition": "component_discovery",
        "rows": 4008,
        "family_rows": EXPECTED_FAMILIES,
        "authorization_sha256": expected_authorization_sha256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"protected manifest mismatch: {field}")
    metadata, tensor_values = _load_and_verify_shards(manifest_path)
    if len(metadata) != 4008:
        raise ValueError("protected metadata row mismatch")
    families: dict[str, int] = defaultdict(int)
    for row in metadata:
        for field in ("logit_a", "logit_b", "correct_logit_margin"):
            if not math.isfinite(float(row[field])):
                raise FloatingPointError(f"non-finite protected metadata: {field}")
        families[str(row["control_family"])] += 1
    if dict(families) != EXPECTED_FAMILIES:
        raise ValueError("protected family row mismatch")
    return {"rows": 4008, "family_rows": EXPECTED_FAMILIES,
            "tensor_values_checked": tensor_values, "all_finite": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["governance", "protected"], required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-authorization-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing audit output: {args.output}")
    if not re_full_sha(args.expected_authorization_sha256):
        raise ValueError("invalid expected authorization SHA")
    audit = (
        _audit_governance(args.manifest, args.expected_authorization_sha256)
        if args.kind == "governance"
        else _audit_protected(args.manifest, args.expected_authorization_sha256)
    )
    value = {
        "schema_version": 1,
        "kind": args.kind,
        "manifest_sha256": sha256_file(args.manifest),
        "expected_authorization_sha256": args.expected_authorization_sha256,
        "audit": audit,
        "success": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


def re_full_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


if __name__ == "__main__":
    main()
