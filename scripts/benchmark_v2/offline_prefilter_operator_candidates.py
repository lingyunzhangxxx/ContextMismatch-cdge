#!/usr/bin/env python3
"""Four-fold pair-hash reconstruction prefilter for the frozen operator grid."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v1.operators import bilinear_transport_features, fit_ridge_transport
from scripts.benchmark_v2.operator_grid import candidate_id, canonical_json, pair_fold


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _features(source: dict, config: dict) -> torch.Tensor:
    history_rank = int(config["history_rank"])
    history_scores = source["source_history_scores"][:, :history_rank]
    threshold = source["history_threshold"][:history_rank]
    family = config["family"]
    if family == "v5_signed":
        history_scores = source["history_reference"][:history_rank].unsqueeze(0).expand(
            history_scores.shape[0], -1
        )
        threshold = torch.zeros_like(threshold)
    context_rank = int(config["context_rank"])
    context = source["source_context_code"][:, :context_rank] if context_rank else None
    return bilinear_transport_features(
        history_scores,
        threshold,
        context,
        one_sided=True,
    )


def _clip_coordinates(
    prediction: torch.Tensor, source_output_norms: torch.Tensor, radius: float
) -> torch.Tensor:
    correction_norm = torch.linalg.vector_norm(prediction.float(), dim=-1, keepdim=True)
    budget = float(radius) * source_output_norms.float().unsqueeze(-1)
    scale = torch.clamp(budget / torch.clamp(correction_norm, min=1e-12), max=1.0)
    return prediction.float() * scale


def _score_config(source: dict, config: dict) -> dict:
    family = config["family"]
    rank = int(config["correction_rank"])
    output = source["output_by_protected_rank"][str(int(config["protected_rank"]))]
    target = output["target_coordinates"][:, :rank].float()
    folds = [int(value) for value in source["pair_fold_ids"]]
    keys = list(source["pair_keys"])
    if len(keys) != target.shape[0] or len(folds) != target.shape[0]:
        raise ValueError("pair-key/fold/target row mismatch")
    if any(pair_fold(key) != fold for key, fold in zip(keys, folds)):
        raise ValueError("stored pair fold does not match frozen pair-hash rule")
    if set(folds) != {0, 1, 2, 3}:
        raise ValueError("all four pair-hash folds must be non-empty")
    learned = family in {"v3_full", "v4_bilinear", "v5_signed"}
    if learned:
        features = _features(source, config)
        full_prediction = None
    elif family == "fixed_translation":
        full_prediction = torch.zeros_like(target)
        full_prediction[:, 0] = float(config["projection_strength"])
    else:
        coordinates = output["source_output_coordinates"][:, :rank].float()
        if family == "v2_one_sided":
            coordinates = torch.relu(
                coordinates - output["projection_threshold"][:rank].float()
            )
        elif family != "symmetric_rank1":
            raise ValueError(f"unknown prefilter family: {family}")
        full_prediction = coordinates * float(config["projection_strength"])
    fold_reports = []
    for fold in range(4):
        test = torch.tensor([value == fold for value in folds], dtype=torch.bool)
        train = ~test
        if learned:
            transport = fit_ridge_transport(
                features[train],
                target[train],
                penalty=float(config["ridge_penalty"]),
            )
            prediction = features[test] @ transport
        else:
            prediction = full_prediction[test]
        prediction = _clip_coordinates(
            prediction,
            source["source_output_norms"][test],
            float(config["global_trust_radius"]),
        )
        residual = prediction - target[test]
        mse = float(residual.square().mean())
        target_energy = float(target[test].square().mean())
        normalized_mse = mse / max(target_energy, 1e-12)
        fold_reports.append(
            {
                "fold": fold,
                "test_rows": int(test.sum()),
                "mse": mse,
                "target_energy": target_energy,
                "normalized_mse": normalized_mse,
            }
        )
    mean_normalized_mse = sum(row["normalized_mse"] for row in fold_reports) / 4.0
    return {
        "mean_normalized_mse": mean_normalized_mse,
        "mean_normalized_rmse": math.sqrt(max(mean_normalized_mse, 0.0)),
        "folds": fold_reports,
    }


def _rank_key(row: dict) -> tuple:
    config = row["config"]
    return (
        float(row["prefilter"]["mean_normalized_mse"]),
        int(config["history_rank"]) + int(config["context_rank"]) + int(config["correction_rank"]),
        int(config["protected_rank"]),
        float(config["global_trust_radius"]),
        row["candidate_id"],
    )


def _coverage_tags(config: dict) -> list[str]:
    if config["family"] == "v4_bilinear":
        return [f"token_scope={config['token_scope']}"]
    if config["family"] == "v5_signed":
        return [
            f"deadzone={float(config['deadzone']):.1f}",
            f"token_scope={config['token_scope']}",
        ]
    return []


def _select_with_coverage(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=_rank_key)
    required_tags = sorted({tag for row in ordered for tag in _coverage_tags(row["config"])})
    selected = []
    selected_ids = set()
    for tag in required_tags:
        choice = next(
            row
            for row in ordered
            if tag in _coverage_tags(row["config"]) and row["candidate_id"] not in selected_ids
        )
        selected.append(choice)
        selected_ids.add(choice["candidate_id"])
    for row in ordered:
        if len(selected) >= count:
            break
        if row["candidate_id"] not in selected_ids:
            selected.append(row)
            selected_ids.add(row["candidate_id"])
    if len(selected) != count:
        raise ValueError(f"could not fill deterministic shortlist of {count}")
    return sorted(selected, key=_rank_key)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-bundle", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--grid", type=Path, required=True)
    parser.add_argument("--grid-report", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--shortlist-dir", type=Path, required=True)
    args = parser.parse_args()
    bundle = _load_torch(args.fit_bundle)
    fit_report = json.loads(args.fit_report.read_text())
    grid_report = json.loads(args.grid_report.read_text())
    extension = json.loads(args.extension_contract.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    if args.output_report.exists() or args.shortlist_dir.exists():
        raise FileExistsError("refusing existing prefilter output")
    if fit_report["bundle_sha256"] != sha256_file(args.fit_bundle):
        raise ValueError("fit report bundle binding mismatch")
    if grid_report["grid_sha256"] != sha256_file(args.grid):
        raise ValueError("grid report binding mismatch")
    if bundle["operator_contract_sha256"] != sha256_file(args.operator_contract):
        raise ValueError("fit bundle operator contract mismatch")
    if bundle["extension_contract_sha256"] != sha256_file(args.extension_contract):
        raise ValueError("fit bundle extension contract mismatch")
    if bundle["operator_site_manifest_sha256"] != sha256_file(args.operator_site_manifest):
        raise ValueError("fit bundle site manifest mismatch")
    records = load_jsonl(args.grid)
    scored = []
    for record in records:
        config = record["config"]
        if record["candidate_id"] != candidate_id(config):
            raise ValueError("grid candidate id mismatch")
        if len(config["site_keys"]) != 1:
            raise ValueError("offline prefilter accepts single-site candidates only")
        site_key = config["site_keys"][0]
        scored.append(
            {
                "candidate_id": record["candidate_id"],
                "site_key": site_key,
                "config": config,
                "prefilter": _score_config(bundle["sites"][site_key], config),
            }
        )
    shortlist_counts = {
        key: int(value)
        for key, value in extension["compute_schedule"]["screening_shortlist"].items()
    }
    sites = [
        f"{int(row['layer'])}:{row['component']}"
        for row in site_manifest["selected_sites_ordered"]
    ]
    shortlisted = []
    for family, total in shortlist_counts.items():
        if total % len(sites):
            raise ValueError("family shortlist must divide evenly over locked sites")
        per_site = total // len(sites)
        for site in sites:
            eligible = [
                row
                for row in scored
                if row["config"]["family"] == family and row["site_key"] == site
            ]
            shortlisted.extend(_select_with_coverage(eligible, per_site))
    args.shortlist_dir.mkdir(parents=True)
    shortlist_records = []
    for rank, row in enumerate(shortlisted, start=1):
        filename = f"{rank:03d}_{row['config']['family']}_{row['candidate_id']}.json"
        atomic_write_text(
            args.shortlist_dir / filename,
            json.dumps(row["config"], indent=2, sort_keys=True) + "\n",
        )
        shortlist_records.append(
            {
                "rank": rank,
                "candidate_id": row["candidate_id"],
                "family": row["config"]["family"],
                "site_key": row["site_key"],
                "config_file": filename,
                "config_sha256": sha256_file(args.shortlist_dir / filename),
                "mean_normalized_mse": row["prefilter"]["mean_normalized_mse"],
            }
        )
    shortlist_manifest = {
        "schema_version": 1,
        "shortlist_rows": len(shortlist_records),
        "family_counts": dict(sorted(Counter(row["family"] for row in shortlist_records).items())),
        "site_counts": dict(sorted(Counter(row["site_key"] for row in shortlist_records).items())),
        "records": shortlist_records,
        "offline_prefilter_only": True,
        "behavioral_evaluation_still_required": True,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.shortlist_dir / "SHORTLIST_MANIFEST.json",
        json.dumps(shortlist_manifest, indent=2, sort_keys=True) + "\n",
    )
    report = {
        "schema_version": 1,
        "fit_bundle_sha256": sha256_file(args.fit_bundle),
        "fit_report_sha256": sha256_file(args.fit_report),
        "grid_sha256": sha256_file(args.grid),
        "grid_report_sha256": sha256_file(args.grid_report),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "fold_rule": "uint64_be(first_8_bytes(SHA256(pair_key_utf8))) mod 4",
        "grid_rows": len(scored),
        "shortlist_rows": len(shortlist_records),
        "all_scores_finite": all(
            math.isfinite(float(row["prefilter"]["mean_normalized_mse"])) for row in scored
        ),
        "scored_candidates": sorted(scored, key=lambda row: (row["config"]["family"], row["site_key"], *_rank_key(row))),
        "shortlist_manifest_sha256": sha256_file(args.shortlist_dir / "SHORTLIST_MANIFEST.json"),
        "selection_performed": False,
        "offline_prefilter_may_replace_behavioral_evaluation": False,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {key: report[key] for key in (
                "grid_rows", "shortlist_rows", "all_scores_finite",
                "shortlist_manifest_sha256", "final_test_open",
                "production_rollout_approved",
            )},
            indent=2,
            sort_keys=True,
        )
    )
    if not report["all_scores_finite"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
