#!/usr/bin/env python3
"""Fit split-isolated governance/operator bases without selecting an operator."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v1.operators import orthonormalize, remove_protected_subspace
from scripts.benchmark_v2.operator_grid import pair_fold


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _check_capture(directory: Path) -> tuple[dict, list[tuple[dict, dict]]]:
    manifest_path = directory / "capture_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shards = []
    for record in manifest["shards"]:
        path = directory / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"capture shard hash mismatch: {path}")
        shard = _load_torch(path)
        if len(shard["metadata"]) != int(record["rows"]):
            raise ValueError(f"capture shard row mismatch: {path}")
        shards.append((record, shard))
    return manifest, shards


def _oriented_svd_basis(
    matrix: torch.Tensor,
    rank: int,
    reference: torch.Tensor | None,
    seed: int = 7703,
) -> torch.Tensor:
    value = matrix.float()
    if value.ndim != 2 or min(value.shape) < rank:
        raise ValueError(f"cannot fit rank {rank} basis from shape {tuple(value.shape)}")
    if min(value.shape) > rank + 16:
        torch.manual_seed(seed)
        q = min(rank + 8, min(value.shape) - 1)
        _u, _s, v = torch.svd_lowrank(value, q=q, niter=4)
        basis = v[:, :rank].contiguous()
    else:
        _u, _s, vh = torch.linalg.svd(value, full_matrices=False)
        basis = vh[:rank].transpose(0, 1).contiguous()
    if reference is not None:
        for index in range(basis.shape[1]):
            if float(reference.float() @ basis[:, index]) < 0:
                basis[:, index].neg_()
    else:
        for index in range(basis.shape[1]):
            column = basis[:, index]
            pivot = int(torch.argmax(torch.abs(column)))
            if float(column[pivot]) < 0:
                basis[:, index].neg_()
    return orthonormalize(basis)


def _low_rank_variation_basis(matrix: torch.Tensor, rank: int, seed: int) -> torch.Tensor:
    centered = matrix.float() - matrix.float().mean(dim=0, keepdim=True)
    if min(centered.shape) <= rank + 4:
        return _oriented_svd_basis(centered, rank, reference=None)
    torch.manual_seed(seed)
    q = min(rank + 8, min(centered.shape) - 1)
    _u, _s, v = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    basis = v[:, :rank].contiguous()
    for index in range(basis.shape[1]):
        column = basis[:, index]
        pivot = int(torch.argmax(torch.abs(column)))
        if float(column[pivot]) < 0:
            basis[:, index].neg_()
    return orthonormalize(basis)


def _auc(labels: list[int], scores: list[float]) -> float:
    positive = [score for label, score in zip(labels, scores) if label == 1]
    negative = [score for label, score in zip(labels, scores) if label == 0]
    if not positive or not negative:
        return math.nan
    wins = 0.0
    for left in positive:
        for right in negative:
            wins += float(left > right) + 0.5 * float(left == right)
    return wins / (len(positive) * len(negative))


def _linear_gate(states: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    positive = states[labels == 1].float()
    negative = states[labels == 0].float()
    direction = positive.mean(dim=0) - negative.mean(dim=0)
    direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=1e-12)
    raw_scores = states.float() @ direction
    positive_mean = raw_scores[labels == 1].mean()
    negative_mean = raw_scores[labels == 0].mean()
    threshold = 0.5 * (positive_mean + negative_mean)
    pooled = torch.cat(
        (raw_scores[labels == 1] - positive_mean, raw_scores[labels == 0] - negative_mean)
    )
    scale = float(torch.clamp(pooled.std(unbiased=False), min=1e-3))
    weight = direction / scale
    bias = -float(threshold) / scale
    return weight, bias, scale


def _gate_report(states: torch.Tensor, metadata: list[dict]) -> dict:
    labels = torch.tensor(
        [int(row["regime"] == "obedience") for row in metadata], dtype=torch.long
    )
    weight, bias, scale = _linear_gate(states, labels)
    scores = (states.float() @ weight + bias).tolist()
    report = {
        "all_auc": _auc(labels.tolist(), [float(value) for value in scores]),
        "scale": scale,
        "by_declared_role": {},
        "leave_one_history_realization_out": {},
        "cross_style": {},
    }
    for role in sorted({row["declared_role"] for row in metadata}):
        indices = [index for index, row in enumerate(metadata) if row["declared_role"] == role]
        report["by_declared_role"][role] = _auc(
            [int(labels[index]) for index in indices],
            [float(scores[index]) for index in indices],
        )
    for held in sorted({int(row["history_realization"]) for row in metadata}):
        train = [index for index, row in enumerate(metadata) if int(row["history_realization"]) != held]
        test = [index for index, row in enumerate(metadata) if int(row["history_realization"]) == held]
        fold_weight, fold_bias, _ = _linear_gate(states[train], labels[train])
        fold_scores = (states[test].float() @ fold_weight + fold_bias).tolist()
        report["leave_one_history_realization_out"][str(held)] = _auc(
            [int(labels[index]) for index in test], [float(value) for value in fold_scores]
        )
    styles = sorted({row["history_style"] for row in metadata})
    for train_style in styles:
        test_styles = [style for style in styles if style != train_style]
        if len(test_styles) != 1:
            continue
        test_style = test_styles[0]
        train = [index for index, row in enumerate(metadata) if row["history_style"] == train_style]
        test = [index for index, row in enumerate(metadata) if row["history_style"] == test_style]
        fold_weight, fold_bias, _ = _linear_gate(states[train], labels[train])
        fold_scores = (states[test].float() @ fold_weight + fold_bias).tolist()
        report["cross_style"][f"train_{train_style}__test_{test_style}"] = _auc(
            [int(labels[index]) for index in test], [float(value) for value in fold_scores]
        )
    report["finite"] = all(
        math.isfinite(float(value))
        for value in [report["all_auc"], *report["by_declared_role"].values(), *report["leave_one_history_realization_out"].values(), *report["cross_style"].values()]
    )
    return {"weight": weight, "bias": bias, "report": report}


def _pair_key(row: dict) -> tuple:
    return (
        row["benchmark"],
        row["item_id"],
        row["partition"],
        row["declared_role"],
        row["history_style"],
        row["history_realization"],
        row["label_swap"],
    )


def _collect_site(
    *,
    layer: int,
    component: str,
    subspace_shards: list[tuple[dict, dict]],
    protected_shards: list[tuple[dict, dict]],
    max_history_rank: int,
    max_context_rank: int,
    max_correction_rank: int,
    protected_ranks: list[int],
) -> tuple[dict, dict]:
    component_key = f"{layer}:{component}"
    boundary_states = []
    boundary_metadata = []
    task_records = []
    final_boundaries = {}
    for _record, shard in subspace_shards:
        states = shard["boundary_states"][str(layer)].float()
        if states.shape[0] != len(shard["boundary_metadata"]):
            raise ValueError("boundary state/metadata mismatch")
        boundary_states.append(states)
        boundary_metadata.extend(shard["boundary_metadata"])
        for index, metadata in enumerate(shard["boundary_metadata"]):
            if int(metadata["turn"]) == 32:
                key = (
                    metadata["declared_role"],
                    metadata["history_style"],
                    int(metadata["history_realization"]),
                    metadata["regime"],
                )
                final_boundaries[key] = states[index]
        inputs = shard["component_inputs"][component_key].float()
        outputs = shard["component_outputs"][component_key].float()
        if not (len(shard["metadata"]) == inputs.shape[0] == outputs.shape[0]):
            raise ValueError("component state/metadata mismatch")
        for index, metadata in enumerate(shard["metadata"]):
            task_records.append((metadata, inputs[index], outputs[index]))
    all_boundary_states = torch.cat(boundary_states, dim=0)
    paired_boundary = defaultdict(dict)
    for index, metadata in enumerate(boundary_metadata):
        key = (
            metadata["declared_role"],
            metadata["history_style"],
            int(metadata["history_realization"]),
            int(metadata["turn"]),
        )
        paired_boundary[key][metadata["regime"]] = all_boundary_states[index]
    boundary_differences = []
    for key, cells in sorted(paired_boundary.items()):
        if set(cells) != {"verification", "obedience"}:
            raise ValueError(f"incomplete boundary pair: {key}")
        boundary_differences.append(cells["obedience"] - cells["verification"])
    boundary_difference_matrix = torch.stack(boundary_differences)
    history_basis = _oriented_svd_basis(
        boundary_difference_matrix,
        max_history_rank,
        boundary_difference_matrix.mean(dim=0),
        seed=7703 + layer,
    )
    history_center = all_boundary_states.mean(dim=0)
    boundary_scores = (all_boundary_states - history_center) @ history_basis
    label_tensor = torch.tensor(
        [int(row["regime"] == "obedience") for row in boundary_metadata], dtype=torch.bool
    )
    history_threshold = 0.5 * (
        boundary_scores[label_tensor].mean(dim=0)
        + boundary_scores[~label_tensor].mean(dim=0)
    )
    gate = _gate_report(all_boundary_states, boundary_metadata)

    task_lookup = defaultdict(dict)
    for metadata, inputs, outputs in task_records:
        task_lookup[_pair_key(metadata)][metadata["regime"]] = (metadata, inputs, outputs)
    source_boundaries = []
    source_inputs = []
    source_outputs = []
    target_outputs = []
    output_differences = []
    pair_keys = []
    all_inputs = []
    for key, cells in sorted(task_lookup.items()):
        if set(cells) != {"verification", "obedience"}:
            raise ValueError(f"incomplete task pair: {key}")
        verification, obedience = cells["verification"], cells["obedience"]
        all_inputs.extend((verification[1], obedience[1]))
        boundary_key = (key[3], key[4], int(key[5]), "obedience")
        if boundary_key not in final_boundaries:
            raise ValueError(f"missing final boundary for task pair: {key}")
        source_boundaries.append(final_boundaries[boundary_key])
        source_inputs.append(obedience[1])
        source_outputs.append(obedience[2])
        target_outputs.append(verification[2])
        output_differences.append(obedience[2] - verification[2])
        pair_keys.append("\x1f".join(str(value) for value in key))
    source_boundary_matrix = torch.stack(source_boundaries)
    source_input_matrix = torch.stack(source_inputs)
    source_output_matrix = torch.stack(source_outputs)
    target_output_matrix = torch.stack(target_outputs)
    output_difference_matrix = torch.stack(output_differences)
    all_input_matrix = torch.stack(all_inputs)
    correction_basis = _oriented_svd_basis(
        output_difference_matrix,
        max_correction_rank,
        output_difference_matrix.mean(dim=0),
        seed=7901 + layer,
    )
    context_center = all_input_matrix.mean(dim=0)
    context_basis = (
        _low_rank_variation_basis(all_input_matrix, max_context_rank, 8801 + layer)
        if max_context_rank
        else torch.empty((all_input_matrix.shape[1], 0), dtype=torch.float32)
    )
    context_scores_all = (all_input_matrix - context_center) @ context_basis
    context_scale = torch.clamp(context_scores_all.std(dim=0, unbiased=False), min=1e-3)

    protected_values = []
    for _record, shard in protected_shards:
        if component_key not in shard["component_outputs"]:
            raise ValueError(f"protected shard lacks selected site {component_key}")
        values = shard["component_outputs"][component_key].float()
        if values.shape[0] != len(shard["metadata"]):
            raise ValueError("protected component state/metadata mismatch")
        protected_values.append(values)
    protected_matrix = torch.cat(protected_values, dim=0)
    max_protected_rank = max(protected_ranks)
    protected_basis = (
        _low_rank_variation_basis(protected_matrix, max_protected_rank, 9917 + layer)
        if max_protected_rank
        else torch.empty((protected_matrix.shape[1], 0), dtype=torch.float32)
    )
    output_by_protected_rank = {}
    component_center = torch.cat((source_output_matrix, target_output_matrix), dim=0).mean(dim=0)
    for protected_rank in protected_ranks:
        current_protected = protected_basis[:, :protected_rank] if protected_rank else None
        output_basis = remove_protected_subspace(correction_basis, current_protected)
        if output_basis.shape[1] < max_correction_rank:
            raise ValueError(
                f"protected rank {protected_rank} leaves only {output_basis.shape[1]} correction dimensions"
            )
        output_basis = output_basis[:, :max_correction_rank]
        output_by_protected_rank[str(protected_rank)] = {
            "basis": output_basis,
            "target_coordinates": output_difference_matrix @ output_basis,
            "source_output_coordinates":
                (source_output_matrix - component_center) @ output_basis,
            "projection_threshold": 0.5
            * (
                ((source_output_matrix - component_center) @ output_basis).mean(dim=0)
                + ((target_output_matrix - component_center) @ output_basis).mean(dim=0)
            ),
        }
    source_history_scores = (source_boundary_matrix - history_center) @ history_basis
    history_reference = torch.relu(source_history_scores - history_threshold).mean(dim=0)
    source_context_code = torch.tanh(
        ((source_input_matrix - context_center) @ context_basis) / context_scale
    ) if max_context_rank else torch.empty((source_input_matrix.shape[0], 0), dtype=torch.float32)
    pair_key_hash = hashlib.sha256(("\n".join(pair_keys) + "\n").encode("utf-8")).hexdigest()
    site = {
        "layer": layer,
        "component": component,
        "history_basis": history_basis,
        "history_center": history_center,
        "history_threshold": history_threshold,
        "context_basis": context_basis,
        "context_center": context_center,
        "context_scale": context_scale,
        "protected_basis": protected_basis,
        "component_center": component_center,
        "source_output_norms": torch.linalg.vector_norm(source_output_matrix, dim=-1),
        "output_by_protected_rank": output_by_protected_rank,
        "source_history_scores": source_history_scores,
        "history_reference": history_reference,
        "source_context_code": source_context_code,
        "gate_weight": gate["weight"],
        "gate_bias": float(gate["bias"]),
        "pair_key_sha256": pair_key_hash,
        "pair_keys": pair_keys,
        "pair_fold_ids": [pair_fold(value) for value in pair_keys],
        "pair_count": len(pair_keys),
    }
    report = {
        "layer": layer,
        "component": component,
        "history_captured_energy": (boundary_difference_matrix @ history_basis).square().sum(dim=0).tolist(),
        "correction_captured_energy": (output_difference_matrix @ correction_basis).square().sum(dim=0).tolist(),
        "history_rows": len(boundary_metadata),
        "paired_task_rows": len(pair_keys),
        "protected_rows": int(protected_matrix.shape[0]),
        "pair_key_sha256": pair_key_hash,
        "gate": gate["report"],
    }
    return site, report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--operator-erratum", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--subspace-capture-dir", type=Path, required=True)
    parser.add_argument("--protected-capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    operator_contract = json.loads(args.operator_contract.read_text())
    operator_erratum = json.loads(args.operator_erratum.read_text())
    extension = json.loads(args.extension_contract.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    subspace_manifest, subspace_shards = _check_capture(args.subspace_capture_dir)
    protected_manifest, protected_shards = _check_capture(args.protected_capture_dir)
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing operator-fit directory: {args.output_dir}")
    if subspace_manifest.get("partition") != "subspace_fit":
        raise ValueError("subspace capture is not from subspace_fit")
    if protected_manifest.get("partition") != "subspace_fit":
        raise ValueError("protected capture is not from subspace_fit")
    if not site_manifest.get("locked") or site_manifest.get("selection_partition") != "component_discovery":
        raise ValueError("operator sites are not locked from component_discovery")
    operator_sha = sha256_file(args.operator_contract)
    erratum_sha = sha256_file(args.operator_erratum)
    extension_sha = sha256_file(args.extension_contract)
    site_sha = sha256_file(args.operator_site_manifest)
    if operator_erratum.get("operator_contract_sha256") != operator_sha:
        raise ValueError("operator erratum binding mismatch")
    if operator_erratum.get("operator_site_execution_contract_sha256") != site_manifest.get(
        "operator_site_execution_contract_sha256"
    ):
        raise ValueError("operator erratum/site-execution binding mismatch")
    for name, capture in (
        ("subspace", subspace_manifest),
        ("protected", protected_manifest),
    ):
        required = {
            "operator_contract_sha256": operator_sha,
            "operator_site_manifest_sha256": site_sha,
            "final_test_open": False,
            "production_rollout_approved": False,
        }
        for field, expected in required.items():
            if capture.get(field) != expected:
                raise ValueError(f"{name} capture lineage mismatch: {field}")
    if subspace_manifest.get("operator_site_execution_contract_sha256") != site_manifest.get(
        "operator_site_execution_contract_sha256"
    ):
        raise ValueError("subspace capture execution binding mismatch")
    if protected_manifest.get("operator_site_execution_contract_sha256") != site_manifest.get(
        "operator_site_execution_contract_sha256"
    ):
        raise ValueError("protected capture execution binding mismatch")
    if extension.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    if extension.get("status") != "frozen_before_operator_forward":
        raise ValueError("operator extension is not frozen")

    fit = operator_contract["subspace_fit"]
    max_history_rank = max(int(value) for value in fit["history_ranks"])
    max_context_rank = max(int(value) for value in fit["task_context_ranks"])
    max_correction_rank = max(int(value) for value in fit["correction_ranks"])
    protected_ranks = [int(value) for value in fit["protected_ranks"]]
    sites = {}
    site_reports = []
    for selected in site_manifest["selected_sites_ordered"]:
        layer = int(selected["layer"])
        component = str(selected["component"])
        key = f"{layer}:{component}"
        site, report = _collect_site(
            layer=layer,
            component=component,
            subspace_shards=subspace_shards,
            protected_shards=protected_shards,
            max_history_rank=max_history_rank,
            max_context_rank=max_context_rank,
            max_correction_rank=max_correction_rank,
            protected_ranks=protected_ranks,
        )
        sites[key] = site
        site_reports.append(report)
    args.output_dir.mkdir(parents=True)
    bundle = {
        "schema_version": 1,
        "sites": sites,
        "site_order": [f"{int(row['layer'])}:{row['component']}" for row in site_manifest["selected_sites_ordered"]],
        "max_history_rank": max_history_rank,
        "max_context_rank": max_context_rank,
        "max_correction_rank": max_correction_rank,
        "protected_ranks": protected_ranks,
        "operator_contract_sha256": operator_sha,
        "operator_erratum_sha256": erratum_sha,
        "extension_contract_sha256": extension_sha,
        "operator_site_manifest_sha256": site_sha,
        "subspace_capture_manifest_sha256": sha256_file(args.subspace_capture_dir / "capture_manifest.json"),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_dir / "capture_manifest.json"),
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    bundle_path = args.output_dir / "operator_fit_bundle.pt"
    incoming = args.output_dir / ".operator_fit_bundle.pt.incoming"
    torch.save(bundle, incoming)
    os.replace(incoming, bundle_path)
    report = {
        "schema_version": 1,
        "bundle_sha256": sha256_file(bundle_path),
        "site_reports": site_reports,
        "lineage": {
            "operator_contract_sha256": bundle["operator_contract_sha256"],
            "operator_erratum_sha256": bundle["operator_erratum_sha256"],
            "extension_contract_sha256": bundle["extension_contract_sha256"],
            "operator_site_manifest_sha256": bundle["operator_site_manifest_sha256"],
            "subspace_capture_manifest_sha256": bundle["subspace_capture_manifest_sha256"],
            "protected_capture_manifest_sha256": bundle["protected_capture_manifest_sha256"],
        },
        "selection_performed": False,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "operator_fit_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
