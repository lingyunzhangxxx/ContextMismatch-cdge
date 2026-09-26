#!/usr/bin/env python3
"""Materialize one SHA-bound v3/v4/v5 candidate from subspace-fit tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v1.operators import bilinear_transport_features, fit_ridge_transport


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _candidate_id(config: dict) -> str:
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def _validate_config(config: dict, bundle: dict, operator_contract: dict, extension: dict) -> None:
    if config["family"] not in {
        "fixed_translation",
        "symmetric_rank1",
        "v2_one_sided",
        "v3_full",
        "v4_bilinear",
        "v5_signed",
    }:
        raise ValueError("unsupported materialized family")
    fit = operator_contract["subspace_fit"]
    allowed = {
        "history_rank": {int(value) for value in fit["history_ranks"]},
        "context_rank": {int(value) for value in fit["task_context_ranks"]},
        "correction_rank": {int(value) for value in fit["correction_ranks"]},
        "protected_rank": {int(value) for value in fit["protected_ranks"]},
        "ridge_penalty": {float(value) for value in fit["ridge_penalties"]},
    }
    for field, choices in allowed.items():
        if config[field] not in choices:
            raise ValueError(f"candidate {field} is outside the frozen grid")
    if config["family"] in {
        "fixed_translation",
        "symmetric_rank1",
        "v2_one_sided",
        "v3_full",
    } and int(config["context_rank"]) != 0:
        raise ValueError("fixed/v2/v3 families must use context rank zero")
    if config["family"] in {"v4_bilinear", "v5_signed"} and int(config["context_rank"]) <= 0:
        raise ValueError("v4/v5 require positive context rank")
    radii = {float(value) for value in extension["global_trust_budget"]["radii"]}
    if float(config["global_trust_radius"]) not in radii:
        raise ValueError("trust radius is outside the frozen grid")
    deadzones = {float(value) for value in extension["alignment_gate"]["deadzones"]}
    if float(config["deadzone"]) not in deadzones:
        raise ValueError("deadzone is outside the frozen grid")
    if config["family"] != "v5_signed" and float(config["deadzone"]) != 0.0:
        raise ValueError("deadzone applies only to v5")
    site_keys = config.get("site_keys")
    if site_keys is not None:
        if not site_keys or len(site_keys) != len(set(site_keys)):
            raise ValueError("site_keys must be a non-empty unique list")
        if any(key not in bundle["site_order"] for key in site_keys):
            raise ValueError("candidate contains an unknown site key")
    else:
        site_count = int(config["site_count"])
        if not 1 <= site_count <= len(bundle["site_order"]):
            raise ValueError("invalid site count")
    if config["token_scope"] not in {"last_token", "all_suffix"}:
        raise ValueError("invalid token scope")
    if float(config.get("projection_strength", 1.0)) not in {0.25, 0.5, 1.0}:
        raise ValueError("projection/translation strength is outside the frozen grid")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-bundle", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    bundle = _load_torch(args.fit_bundle)
    fit_report = json.loads(args.fit_report.read_text())
    operator_contract = json.loads(args.operator_contract.read_text())
    extension = json.loads(args.extension_contract.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    config = json.loads(args.config.read_text())
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing candidate directory: {args.output_dir}")
    if fit_report["bundle_sha256"] != sha256_file(args.fit_bundle):
        raise ValueError("fit report bundle binding mismatch")
    if bundle["operator_contract_sha256"] != sha256_file(args.operator_contract):
        raise ValueError("fit bundle operator contract mismatch")
    if bundle["extension_contract_sha256"] != sha256_file(args.extension_contract):
        raise ValueError("fit bundle extension contract mismatch")
    if bundle["operator_site_manifest_sha256"] != sha256_file(args.operator_site_manifest):
        raise ValueError("fit bundle site manifest mismatch")
    if not site_manifest.get("locked"):
        raise ValueError("site manifest is not locked")
    _validate_config(config, bundle, operator_contract, extension)

    selected_keys = list(
        config.get("site_keys")
        or bundle["site_order"][: int(config["site_count"])]
    )
    selected_scores = {
        f"{int(row['layer'])}:{row['component']}": max(float(row["ranking_score"]), 0.0)
        for row in site_manifest["selected_sites_ordered"]
        if f"{int(row['layer'])}:{row['component']}" in selected_keys
    }
    denominator = sum(selected_scores.values())
    shares = {
        key: (selected_scores[key] / denominator if denominator > 0 else 1.0 / len(selected_keys))
        for key in selected_keys
    }
    if abs(sum(shares.values()) - 1.0) > 1e-6:
        raise RuntimeError("site trust shares do not sum to one")
    materialized_sites = {}
    for key in selected_keys:
        source = bundle["sites"][key]
        history_rank = int(config["history_rank"])
        context_rank = int(config["context_rank"])
        correction_rank = int(config["correction_rank"])
        protected_rank = int(config["protected_rank"])
        history_scores = source["source_history_scores"][:, :history_rank]
        threshold = source["history_threshold"][:history_rank]
        history_reference = source["history_reference"][:history_rank]
        if context_rank:
            context_code = source["source_context_code"][:, :context_rank]
        else:
            context_code = None
        output = source["output_by_protected_rank"][str(protected_rank)]
        layer_text, component = key.split(":", 1)
        site_value = {
            "layer": int(layer_text),
            "component": component,
            "operator_kind": "context_transport",
            "history_basis": source["history_basis"][:, :history_rank],
            "history_center": source["history_center"],
            "history_threshold": threshold,
            "history_reference": history_reference
            if config["family"] == "v5_signed"
            else None,
            "context_basis": source["context_basis"][:, :context_rank],
            "context_center": source["context_center"],
            "context_scale": source["context_scale"][:context_rank],
            "output_basis": output["basis"][:, :correction_rank],
            "gate_weight": source["gate_weight"],
            "gate_bias": float(source["gate_bias"]),
            "deadzone": float(config["deadzone"]),
            "trust_share": shares[key],
            "site_trust_radius": float(config["global_trust_radius"]) * shares[key],
            "max_relative_correction": float(config["global_trust_radius"]) * shares[key],
            "edit_last_token_only": config["token_scope"] == "last_token",
        }
        if config["family"] in {
            "fixed_translation",
            "symmetric_rank1",
            "v2_one_sided",
        }:
            site_value["operator_kind"] = (
                "fixed_translation"
                if config["family"] == "fixed_translation"
                else "projection"
            )
            site_value["output_basis"] = output["basis"][:, :correction_rank]
            site_value["projection_threshold"] = output["projection_threshold"][
                :correction_rank
            ]
            site_value["component_center"] = source["component_center"]
            site_value["projection_strength"] = float(
                config.get("projection_strength", 1.0)
            )
            site_value["one_sided"] = config["family"] == "v2_one_sided"
        else:
            fit_history_scores = (
                history_reference.unsqueeze(0).expand(history_scores.shape[0], -1)
                if config["family"] == "v5_signed"
                else history_scores
            )
            fit_threshold = (
                torch.zeros_like(threshold)
                if config["family"] == "v5_signed"
                else threshold
            )
            features = bilinear_transport_features(
                fit_history_scores,
                fit_threshold,
                context_code,
                one_sided=True,
            )
            targets = output["target_coordinates"][:, :correction_rank]
            site_value["transport"] = fit_ridge_transport(
                features,
                targets,
                penalty=float(config["ridge_penalty"]),
            )
        materialized_sites[key] = site_value
    candidate_id = _candidate_id(config)
    args.output_dir.mkdir(parents=True)
    tensor_path = args.output_dir / f"operator_candidate_{candidate_id}.pt"
    incoming = args.output_dir / f".operator_candidate_{candidate_id}.pt.incoming"
    candidate = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "config": config,
        "sites": materialized_sites,
        "site_order": selected_keys,
        "fit_bundle_sha256": sha256_file(args.fit_bundle),
        "fit_report_sha256": sha256_file(args.fit_report),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    torch.save(candidate, incoming)
    os.replace(incoming, tensor_path)
    manifest = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "config": config,
        "tensor_file": tensor_path.name,
        "tensor_sha256": sha256_file(tensor_path),
        "site_order": selected_keys,
        "site_trust_shares": shares,
        "sum_site_trust_radii": sum(
            float(value["site_trust_radius"]) for value in materialized_sites.values()
        ),
        "global_trust_radius": float(config["global_trust_radius"]),
        "fit_bundle_sha256": candidate["fit_bundle_sha256"],
        "fit_report_sha256": candidate["fit_report_sha256"],
        "operator_contract_sha256": candidate["operator_contract_sha256"],
        "extension_contract_sha256": candidate["extension_contract_sha256"],
        "operator_site_manifest_sha256": candidate["operator_site_manifest_sha256"],
        "selection_performed": False,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    if manifest["sum_site_trust_radii"] > manifest["global_trust_radius"] + 1e-8:
        raise RuntimeError("materialized site radii exceed the global trust budget")
    manifest_path = args.output_dir / f"operator_candidate_{candidate_id}.manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
