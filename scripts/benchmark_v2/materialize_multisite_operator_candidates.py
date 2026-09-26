#!/usr/bin/env python3
"""Combine admissible single-site tensors under one reallocated global trust budget."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v2.operator_grid import candidate_id


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-plan", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output: {args.output_dir}")
    plan = json.loads(args.source_plan.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    extension = json.loads(args.extension_contract.read_text())
    ranking_scores = {
        f"{int(row['layer'])}:{row['component']}": max(float(row["ranking_score"]), 0.0)
        for row in site_manifest["selected_sites_ordered"]
    }
    args.output_dir.mkdir(parents=True)
    records = []
    for current_plan in plan["plans"]:
        loaded = []
        for source in current_plan["sources"]:
            manifest_path = Path(source["candidate_manifest"])
            manifest = json.loads(manifest_path.read_text())
            tensor_path = Path(source["candidate_tensor"])
            if sha256_file(manifest_path) != source["candidate_manifest_sha256"]:
                raise ValueError("source candidate manifest changed")
            if sha256_file(tensor_path) != source["candidate_tensor_sha256"]:
                raise ValueError("source candidate tensor changed")
            tensor = _load_torch(tensor_path)
            if manifest["candidate_id"] != tensor["candidate_id"]:
                raise ValueError("source candidate identity mismatch")
            if manifest["config"]["family"] != current_plan["family"]:
                raise ValueError("multisite sources do not share a family")
            if len(manifest["site_order"]) != 1:
                raise ValueError("multisite source is not single-site")
            loaded.append((source, manifest, tensor))
        site_keys = [value[1]["site_order"][0] for value in loaded]
        if len(site_keys) != len(set(site_keys)):
            raise ValueError("multisite plan repeats a site")
        denominator = sum(ranking_scores[key] for key in site_keys)
        shares = {
            key: (ranking_scores[key] / denominator if denominator else 1.0 / len(site_keys))
            for key in site_keys
        }
        for radius in extension["global_trust_budget"]["radii"]:
            config = {
                "family": current_plan["family"],
                "multisite": True,
                "site_keys": site_keys,
                "source_candidate_ids": [value[1]["candidate_id"] for value in loaded],
                "global_trust_radius": float(radius),
            }
            current_id = candidate_id(config)
            sites = {}
            per_site_configs = {}
            for _source, manifest, tensor in loaded:
                key = manifest["site_order"][0]
                site = copy.deepcopy(tensor["sites"][key])
                site["trust_share"] = shares[key]
                site["site_trust_radius"] = float(radius) * shares[key]
                site["max_relative_correction"] = float(radius) * shares[key]
                sites[key] = site
                per_site_configs[key] = manifest["config"]
            candidate = {
                "schema_version": 1,
                "candidate_id": current_id,
                "config": config,
                "sites": sites,
                "site_order": site_keys,
                "source_plan_sha256": sha256_file(args.source_plan),
                "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
                "extension_contract_sha256": sha256_file(args.extension_contract),
                "final_test_open": False,
                "production_rollout_approved": False,
            }
            tensor_path = args.output_dir / f"operator_candidate_{current_id}.pt"
            incoming = args.output_dir / f".operator_candidate_{current_id}.pt.incoming"
            torch.save(candidate, incoming)
            os.replace(incoming, tensor_path)
            manifest = {
                "schema_version": 1,
                "candidate_id": current_id,
                "config": config,
                "per_site_configs": per_site_configs,
                "tensor_file": tensor_path.name,
                "tensor_sha256": sha256_file(tensor_path),
                "site_order": site_keys,
                "site_trust_shares": shares,
                "sum_site_trust_radii": sum(float(radius) * value for value in shares.values()),
                "global_trust_radius": float(radius),
                "source_plan_sha256": candidate["source_plan_sha256"],
                "operator_site_manifest_sha256": candidate["operator_site_manifest_sha256"],
                "extension_contract_sha256": candidate["extension_contract_sha256"],
                "selection_performed": False,
                "final_test_open": False,
                "production_rollout_approved": False,
            }
            if manifest["sum_site_trust_radii"] > float(radius) + 1e-8:
                raise RuntimeError("multisite radii exceed the global trust budget")
            manifest_path = args.output_dir / f"operator_candidate_{current_id}.manifest.json"
            atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            records.append(
                {
                    "candidate_id": current_id,
                    "family": current_plan["family"],
                    "site_count": len(site_keys),
                    "global_trust_radius": float(radius),
                    "tensor_file": tensor_path.name,
                    "tensor_sha256": manifest["tensor_sha256"],
                    "manifest_file": manifest_path.name,
                    "manifest_sha256": sha256_file(manifest_path),
                }
            )
    output_manifest = {
        "schema_version": 1,
        "source_plan_sha256": sha256_file(args.source_plan),
        "candidate_count": len(records),
        "records": records,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "MULTISITE_CANDIDATE_MANIFEST.json",
        json.dumps(output_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(output_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
