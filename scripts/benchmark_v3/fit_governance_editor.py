#!/usr/bin/env python3
"""Fit AMSGE from split-isolated counterfactual and protected captures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v3.adaptive_governance import (
    AdaptiveGovernanceSiteEditor,
    AdaptiveMultiSiteGovernanceEditor,
    GovernanceSite,
    protected_residual_output_basis,
    safe_scale,
)


def _fold(item_id: str) -> int:
    return int.from_bytes(hashlib.sha256(item_id.encode("utf-8")).digest()[:8], "big") % 8


def _pair_key(row: dict) -> tuple:
    return (
        row["item_id"],
        row["declared_role"],
        row["history_style"],
        int(row["history_realization"]),
        row["task_requirement"],
        int(row["label_swap"]),
    )


def _matched_history(row: dict) -> str:
    return "obedience" if float(row["target_obedience"]) == 1.0 else "verification"


def _pca_basis(values: torch.Tensor, rank: int) -> torch.Tensor:
    centered = values.float() - values.float().mean(dim=0)
    q = min(rank + 8, min(centered.shape))
    if q < rank:
        raise ValueError(f"capture cannot support requested PCA rank {rank}")
    _, _, basis = torch.pca_lowrank(centered, q=q, center=False, niter=4)
    return basis[:, :rank]


def _load_capture(capture_dir: Path, manifest_path: Path, site_keys: list[str]) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("complete") is not True or manifest.get("rows") != 6144:
        raise ValueError("full governance capture is incomplete")
    rows: list[dict] = []
    boundary = {key: [] for key in site_keys}
    inputs = {key: [] for key in site_keys}
    outputs = {key: [] for key in site_keys}
    observed_keys = set()
    for record in manifest["shards"]:
        path = capture_dir / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"capture shard SHA mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        metadata = shard["metadata"]
        if len(metadata) != int(record["rows"]):
            raise ValueError(f"capture shard row mismatch: {path}")
        for row in metadata:
            key = row["job_key"]
            if key in observed_keys:
                raise ValueError(f"duplicate capture key: {key}")
            observed_keys.add(key)
        rows.extend(metadata)
        for site_key in site_keys:
            layer = site_key.split(":", 1)[0]
            state = shard["boundary_states"][layer].float()
            boundary[site_key].append(state.unsqueeze(0).expand(len(metadata), -1).clone())
            inputs[site_key].append(shard["component_inputs"][site_key].float())
            outputs[site_key].append(shard["component_outputs"][site_key].float())
    if len(rows) != 6144 or len(observed_keys) != 6144:
        raise ValueError("governance capture does not contain 6,144 unique rows")
    ordered = sorted(range(len(rows)), key=lambda index: rows[index]["job_key"])
    rows = [rows[index] for index in ordered]
    index_tensor = torch.tensor(ordered, dtype=torch.long)
    return {
        "manifest": manifest,
        "metadata": rows,
        "boundary": {
            key: torch.cat(values, dim=0).index_select(0, index_tensor)
            for key, values in boundary.items()
        },
        "inputs": {
            key: torch.cat(values, dim=0).index_select(0, index_tensor)
            for key, values in inputs.items()
        },
        "outputs": {
            key: torch.cat(values, dim=0).index_select(0, index_tensor)
            for key, values in outputs.items()
        },
    }


def _load_protected(capture_dir: Path, manifest_path: Path, site_keys: list[str]) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("rows") != 4008 or manifest.get("partition") != "subspace_fit":
        raise ValueError("protected capture is incomplete or from the wrong split")
    boundary = {key: [] for key in site_keys}
    inputs = {key: [] for key in site_keys}
    outputs = {key: [] for key in site_keys}
    families: list[str] = []
    for record in manifest["shards"]:
        path = capture_dir / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"protected shard SHA mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        metadata = shard["metadata"]
        families.extend(str(row["control_family"]) for row in metadata)
        for site_key in site_keys:
            layer = site_key.split(":", 1)[0]
            state = shard["boundary_states"][layer].float()
            boundary[site_key].append(state.unsqueeze(0).expand(len(metadata), -1).clone())
            inputs[site_key].append(shard["component_inputs"][site_key].float())
            outputs[site_key].append(shard["component_outputs"][site_key].float())
    if len(families) != 4008:
        raise ValueError("protected capture row count mismatch")
    return {
        "manifest": manifest,
        "families": families,
        "boundary": {key: torch.cat(values, dim=0) for key, values in boundary.items()},
        "inputs": {key: torch.cat(values, dim=0) for key, values in inputs.items()},
        "outputs": {key: torch.cat(values, dim=0) for key, values in outputs.items()},
    }


def _teacher_indices(metadata: list[dict]) -> torch.Tensor:
    lookup = {(_pair_key(row), row["history_condition"]): index for index, row in enumerate(metadata)}
    if len(lookup) != len(metadata):
        raise ValueError("capture metadata contains duplicate counterfactual cells")
    indices = []
    for row in metadata:
        key = (_pair_key(row), _matched_history(row))
        if key not in lookup:
            raise ValueError(f"missing matched counterfactual teacher for {row['job_key']}")
        indices.append(lookup[key])
    return torch.tensor(indices, dtype=torch.long)


def _balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    for label in (0.0, 1.0):
        mask = target == label
        if bool(mask.any()):
            values.append(float((prediction[mask] == target[mask]).float().mean()))
    return sum(values) / len(values)


def _fit_margin_probe(outputs: torch.Tensor, margins: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    features = outputs.float() @ basis.float()
    features = torch.cat((features, torch.ones(features.shape[0], 1)), dim=1)
    gram = features.transpose(0, 1) @ features
    gram = gram + 1e-3 * torch.eye(gram.shape[0])
    return torch.linalg.solve(gram, features.transpose(0, 1) @ margins.float())


def _relative_squared(error: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    numerator = torch.sum(error.float().square(), dim=-1)
    denominator = torch.clamp(torch.sum(reference.float().square(), dim=-1), min=1e-6)
    return numerator / denominator


def _scalar_rms_scale(values: torch.Tensor, center: torch.Tensor) -> float:
    scale = torch.sqrt(torch.mean((values.float() - center.float()).square()))
    if not torch.isfinite(scale):
        raise FloatingPointError("non-finite full-state head scale")
    return max(float(scale), 1e-4)


def _head_balanced_accuracies(
    editor: AdaptiveGovernanceSiteEditor,
    boundary: torch.Tensor,
    inputs: torch.Tensor,
    indices: torch.Tensor,
    history_target: torch.Tensor,
    task_target: torch.Tensor,
) -> tuple[float, float]:
    editor.eval()
    with torch.no_grad():
        history_logit, task_logit, _, _ = editor.governance_logits(
            boundary[indices], inputs[indices]
        )
    return (
        _balanced_accuracy(
            (torch.sigmoid(history_logit) >= 0.5).float(), history_target[indices]
        ),
        _balanced_accuracy(
            (torch.sigmoid(task_logit) >= 0.5).float(), task_target[indices]
        ),
    )


def _cpu_adamw(
    parameters,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Build AdamW without backend device discovery for the CPU-only fit stage."""
    parameter_list = list(parameters)
    if not parameter_list:
        raise ValueError("governance editor has no trainable parameters")
    non_cpu = sorted(
        {
            parameter.device.type
            for parameter in parameter_list
            if parameter.device.type != "cpu"
        }
    )
    if non_cpu:
        raise ValueError(
            f"governance fit optimizer requires CPU parameters, found {non_cpu}"
        )
    return torch.optim.AdamW(
        parameter_list,
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )


def _category_penalty(
    recovery: torch.Tensor,
    mismatch: torch.Tensor,
    category: torch.Tensor,
) -> torch.Tensor:
    means = []
    for value in (0, 1):
        mask = mismatch & (category == value)
        if bool(mask.any()):
            means.append(recovery[mask].mean())
    if len(means) != 2:
        return recovery.new_zeros(())
    return (means[0] - means[1]).square()


def _site_metrics(
    editor: AdaptiveGovernanceSiteEditor,
    *,
    boundary: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    targets: torch.Tensor,
    indices: torch.Tensor,
    mismatch: torch.Tensor,
    history_target: torch.Tensor,
    task_target: torch.Tensor,
    label_swap: torch.Tensor,
) -> dict:
    editor.eval()
    with torch.no_grad():
        correction, diagnostics = editor.correction(
            value=outputs[indices],
            boundary_state=boundary[indices],
            context_state=inputs[indices],
        )
    target = targets[indices]
    current_mismatch = mismatch[indices]
    result: dict[str, object] = {}
    if bool(current_mismatch.any()):
        denominator = torch.sum(target[current_mismatch].square()).clamp(min=1e-6)
        nmse = torch.sum((correction[current_mismatch] - target[current_mismatch]).square()) / denominator
        result["normalized_teacher_mse"] = float(nmse)
    else:
        result["normalized_teacher_mse"] = None
    per_row = 1.0 - _relative_squared(correction - target, target)
    directions = {}
    swaps = {}
    for value, name in ((0, "independent_verification"), (1, "delegated_choice")):
        mask = current_mismatch & (task_target[indices] == float(value))
        directions[name] = float(per_row[mask].mean()) if bool(mask.any()) else None
    for value in (0, 1):
        mask = current_mismatch & (label_swap[indices] == value)
        swaps[str(value)] = float(per_row[mask].mean()) if bool(mask.any()) else None
    result["teacher_mse_improvement_by_direction"] = directions
    result["teacher_mse_improvement_by_label_swap"] = swaps
    result["history_balanced_accuracy"] = _balanced_accuracy(
        (diagnostics["history_probability"] >= 0.5).float(), history_target[indices]
    )
    result["task_balanced_accuracy"] = _balanced_accuracy(
        (diagnostics["task_probability"] >= 0.5).float(), task_target[indices]
    )
    relative = torch.sqrt(_relative_squared(correction, outputs[indices]))
    result["mean_relative_correction"] = float(relative.mean())
    result["maximum_relative_correction"] = float(relative.max())
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing fit output: {args.output_dir}")
    contract = json.loads(args.editor_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    required_authorization = {
        "stage": "governance_fit",
        "execution_allowed": True,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_manifest),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_authorization.items():
        if authorization.get(field) != expected:
            raise ValueError(f"fit authorization mismatch: {field}")
    if contract.get("status") not in {
        "frozen_before_any_code_v22_forward",
        "frozen_before_any_v1_post_failure_operator_dev_forward",
    }:
        raise ValueError("editor contract is not frozen")
    seed = int(contract["optimization"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    locked_sites = sorted(contract["locked_sites"], key=lambda row: int(row["execution_order"]))
    site_keys = [f"{site['layer']}:{site['component']}" for site in locked_sites]
    capture = _load_capture(args.capture_manifest.parent, args.capture_manifest, site_keys)
    protected = _load_protected(
        args.protected_capture_manifest.parent,
        args.protected_capture_manifest,
        site_keys,
    )
    metadata = capture["metadata"]
    teacher_indices = _teacher_indices(metadata)
    folds = torch.tensor([_fold(row["item_id"]) for row in metadata], dtype=torch.long)
    train_indices = torch.nonzero(folds <= 5, as_tuple=False).squeeze(-1)
    calibration_indices = torch.nonzero(folds == 6, as_tuple=False).squeeze(-1)
    audit_indices = torch.nonzero(folds == 7, as_tuple=False).squeeze(-1)
    if min(len(train_indices), len(calibration_indices), len(audit_indices)) == 0:
        raise RuntimeError("one or more frozen fit folds are empty")
    history_target = torch.tensor(
        [1.0 if row["history_condition"] == "obedience" else 0.0 for row in metadata]
    )
    task_target = torch.tensor([float(row["target_obedience"]) for row in metadata])
    label_swap = torch.tensor([int(row["label_swap"]) for row in metadata])
    mismatch = torch.tensor(
        [row["history_condition"] != _matched_history(row) for row in metadata],
        dtype=torch.bool,
    )
    margins = torch.tensor([float(row["task_aligned_margin"]) for row in metadata])
    matched_margins = margins.index_select(0, teacher_indices)

    optimization = contract["optimization"]
    head_input_mode = str(
        contract.get("architecture", {}).get("head_input_mode", "pca_coordinates")
    )
    v2_fit = head_input_mode == "full_state"
    correction_optimization = (
        optimization["correction_fit"] if v2_fit else optimization
    )
    weights = correction_optimization["loss_weights"]
    site_editors: dict[GovernanceSite, AdaptiveGovernanceSiteEditor] = {}
    site_reports = []
    checkpoint_sites = []
    for site_number, site_config in enumerate(locked_sites):
        torch.manual_seed(seed + site_number)
        key = f"{site_config['layer']}:{site_config['component']}"
        boundary = capture["boundary"][key]
        inputs = capture["inputs"][key]
        outputs = capture["outputs"][key]
        teacher_outputs = outputs.index_select(0, teacher_indices)
        targets = teacher_outputs - outputs
        train_mismatch_indices = train_indices[mismatch[train_indices]]
        boundary_center = boundary[train_indices].mean(dim=0)
        context_center = inputs[train_indices].mean(dim=0)
        boundary_basis = _pca_basis(
            boundary[train_indices], int(site_config["boundary_rank"])
        )
        context_basis = _pca_basis(
            inputs[train_indices], int(site_config["context_rank"])
        )
        output_basis = protected_residual_output_basis(
            targets[train_mismatch_indices],
            protected["outputs"][key],
            int(site_config["output_rank"]),
            protected_rank=32,
        )
        boundary_coordinates = (boundary[train_indices] - boundary_center) @ boundary_basis
        context_coordinates = (inputs[train_indices] - context_center) @ context_basis
        boundary_head_scale = _scalar_rms_scale(
            boundary[train_indices], boundary_center
        )
        context_head_scale = _scalar_rms_scale(inputs[train_indices], context_center)
        editor = AdaptiveGovernanceSiteEditor(
            boundary_basis=boundary_basis,
            context_basis=context_basis,
            output_basis=output_basis,
            boundary_center=boundary_center,
            context_center=context_center,
            boundary_scale=safe_scale(boundary_coordinates),
            context_scale=safe_scale(context_coordinates),
            maximum_relative_correction=float(site_config["max_relative_correction"]),
            hidden_width=96,
            temperature=1.0,
            head_input_mode=head_input_mode,
            boundary_head_scale=boundary_head_scale,
            context_head_scale=context_head_scale,
        )
        batch_size = int(optimization["batch_size"])
        head_pretraining_report = {
            "performed": False,
            "reason": "V1 jointly trains PCA-coordinate heads and correction",
        }
        if v2_fit:
            head_config = optimization["head_pretraining"]
            for parameter in editor.conditioner.parameters():
                parameter.requires_grad_(False)
            for parameter in editor.output_head.parameters():
                parameter.requires_grad_(False)
            head_parameters = [
                *editor.history_head.parameters(),
                *editor.task_head.parameters(),
            ]
            head_optimizer = _cpu_adamw(
                head_parameters,
                learning_rate=float(head_config["learning_rate"]),
                weight_decay=float(head_config["weight_decay"]),
            )
            head_generator = torch.Generator().manual_seed(seed + 500 + site_number)
            best_head_score = (-math.inf, -math.inf)
            best_head_state = {
                "history_head": copy.deepcopy(editor.history_head.state_dict()),
                "task_head": copy.deepcopy(editor.task_head.state_dict()),
            }
            best_head_epoch = 0
            head_stale = 0
            head_trace = []
            for head_epoch in range(1, int(head_config["maximum_epochs"]) + 1):
                editor.train()
                permutation = train_indices[
                    torch.randperm(len(train_indices), generator=head_generator)
                ]
                head_losses = []
                for offset in range(0, len(permutation), batch_size):
                    indices = permutation[offset : offset + batch_size]
                    history_logit, task_logit, _, _ = editor.governance_logits(
                        boundary[indices], inputs[indices]
                    )
                    head_loss = F.binary_cross_entropy_with_logits(
                        history_logit, history_target[indices]
                    ) + F.binary_cross_entropy_with_logits(
                        task_logit, task_target[indices]
                    )
                    if not torch.isfinite(head_loss):
                        raise FloatingPointError(
                            f"non-finite head-pretraining loss at {key} epoch {head_epoch}"
                        )
                    head_optimizer.zero_grad(set_to_none=True)
                    head_loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        head_parameters, float(optimization["gradient_norm_clip"])
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError(
                            f"non-finite head-pretraining gradient at {key} epoch {head_epoch}"
                        )
                    head_optimizer.step()
                    head_losses.append(float(head_loss.detach()))
                history_ba, task_ba = _head_balanced_accuracies(
                    editor,
                    boundary,
                    inputs,
                    calibration_indices,
                    history_target,
                    task_target,
                )
                head_score = ((history_ba + task_ba) / 2.0, min(history_ba, task_ba))
                head_trace.append(
                    {
                        "epoch": head_epoch,
                        "mean_train_loss": sum(head_losses) / len(head_losses),
                        "calibration_history_balanced_accuracy": history_ba,
                        "calibration_task_balanced_accuracy": task_ba,
                        "selection_score": list(head_score),
                    }
                )
                if head_score > best_head_score:
                    best_head_score = head_score
                    best_head_epoch = head_epoch
                    best_head_state = {
                        "history_head": copy.deepcopy(editor.history_head.state_dict()),
                        "task_head": copy.deepcopy(editor.task_head.state_dict()),
                    }
                    head_stale = 0
                else:
                    head_stale += 1
                if head_stale >= int(head_config["early_stopping_patience"]):
                    break
            editor.history_head.load_state_dict(best_head_state["history_head"])
            editor.task_head.load_state_dict(best_head_state["task_head"])
            for parameter in editor.history_head.parameters():
                parameter.requires_grad_(False)
            for parameter in editor.task_head.parameters():
                parameter.requires_grad_(False)
            for parameter in editor.conditioner.parameters():
                parameter.requires_grad_(True)
            for parameter in editor.output_head.parameters():
                parameter.requires_grad_(True)
            head_pretraining_report = {
                "performed": True,
                "best_epoch": best_head_epoch,
                "best_calibration_mean_balanced_accuracy": best_head_score[0],
                "best_calibration_minimum_balanced_accuracy": best_head_score[1],
                "epochs_completed": len(head_trace),
                "trace": head_trace,
            }
        probe = _fit_margin_probe(
            outputs[train_indices], margins[train_indices], editor.output_basis
        )
        optimizer = _cpu_adamw(
            (parameter for parameter in editor.parameters() if parameter.requires_grad),
            learning_rate=float(correction_optimization["learning_rate"]),
            weight_decay=float(correction_optimization["weight_decay"]),
        )
        generator = torch.Generator().manual_seed(seed + 1000 + site_number)
        best_metric = math.inf
        best_state = copy.deepcopy(editor.state_dict())
        best_epoch = 0
        stale = 0
        epoch_reports = []
        protected_count = protected["outputs"][key].shape[0]
        for epoch in range(1, int(correction_optimization["maximum_epochs"]) + 1):
            editor.train()
            permutation = train_indices[torch.randperm(len(train_indices), generator=generator)]
            losses = []
            for offset in range(0, len(permutation), batch_size):
                indices = permutation[offset : offset + batch_size]
                correction, diagnostics = editor.correction(
                    value=outputs[indices],
                    boundary_state=boundary[indices],
                    context_state=inputs[indices],
                )
                desired = targets[indices]
                current_mismatch = mismatch[indices]
                if bool(current_mismatch.any()):
                    reconstruction = _relative_squared(
                        correction[current_mismatch] - desired[current_mismatch],
                        desired[current_mismatch],
                    ).mean()
                else:
                    reconstruction = correction.new_zeros(())
                matched_mask = ~current_mismatch
                preservation = (
                    _relative_squared(correction[matched_mask], outputs[indices][matched_mask]).mean()
                    if bool(matched_mask.any())
                    else correction.new_zeros(())
                )
                history_loss = F.binary_cross_entropy_with_logits(
                    diagnostics["history_logit"], history_target[indices]
                )
                task_loss = F.binary_cross_entropy_with_logits(
                    diagnostics["task_logit"], task_target[indices]
                )
                edited_coordinates = (outputs[indices] + correction) @ editor.output_basis
                margin_prediction = (
                    torch.cat(
                        (edited_coordinates, torch.ones(edited_coordinates.shape[0], 1)), dim=1
                    )
                    @ probe
                )
                margin_loss = ((margin_prediction - matched_margins[indices]) / 4.0).square().mean()
                recovery = 1.0 - _relative_squared(correction - desired, desired)
                swap_penalty = _category_penalty(
                    recovery, current_mismatch, label_swap[indices]
                )
                direction_penalty = _category_penalty(
                    recovery, current_mismatch, task_target[indices].long()
                )
                protected_indices = torch.randint(
                    0, protected_count, (len(indices),), generator=generator
                )
                protected_correction, _ = editor.correction(
                    value=protected["outputs"][key][protected_indices],
                    boundary_state=protected["boundary"][key][protected_indices],
                    context_state=protected["inputs"][key][protected_indices],
                )
                protected_loss = _relative_squared(
                    protected_correction,
                    protected["outputs"][key][protected_indices],
                ).mean()
                intervention = _relative_squared(correction, outputs[indices]).mean()
                loss = (
                    float(weights["mismatched_teacher_activation_reconstruction"]) * reconstruction
                    + float(weights["matched_state_preservation"]) * preservation
                    + float(weights["task_aligned_margin_probe_recovery"]) * margin_loss
                    + float(weights.get("history_classification", 0.0)) * history_loss
                    + float(weights.get("task_requirement_classification", 0.0)) * task_loss
                    + float(weights["label_swap_equivariance"]) * swap_penalty
                    + float(weights["direction_balance"]) * direction_penalty
                    + float(weights["protected_zero_correction"]) * protected_loss
                    + float(weights["intervention_norm"]) * intervention
                )
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite training loss at {key} epoch {epoch}")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in editor.parameters() if parameter.requires_grad],
                    float(optimization["gradient_norm_clip"]),
                )
                if not torch.isfinite(gradient_norm):
                    raise FloatingPointError(f"non-finite gradient at {key} epoch {epoch}")
                optimizer.step()
                losses.append(float(loss.detach()))
            calibration = _site_metrics(
                editor,
                boundary=boundary,
                inputs=inputs,
                outputs=outputs,
                targets=targets,
                indices=calibration_indices,
                mismatch=mismatch,
                history_target=history_target,
                task_target=task_target,
                label_swap=label_swap,
            )
            protected_eval_indices = torch.arange(min(512, protected_count))
            editor.eval()
            with torch.no_grad():
                protected_correction, _ = editor.correction(
                    value=protected["outputs"][key][protected_eval_indices],
                    boundary_state=protected["boundary"][key][protected_eval_indices],
                    context_state=protected["inputs"][key][protected_eval_indices],
                )
            protected_penalty = float(
                _relative_squared(
                    protected_correction,
                    protected["outputs"][key][protected_eval_indices],
                ).mean()
            )
            protected_metric_weight = (
                float(weights["protected_zero_correction"]) if v2_fit else 1.0
            )
            metric = (
                float(calibration["normalized_teacher_mse"])
                + protected_metric_weight * protected_penalty
            )
            epoch_reports.append(
                {
                    "epoch": epoch,
                    "mean_train_loss": sum(losses) / len(losses),
                    "calibration_metric": metric,
                }
            )
            if metric < best_metric - 1e-6:
                best_metric = metric
                best_epoch = epoch
                best_state = copy.deepcopy(editor.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= int(correction_optimization["early_stopping_patience"]):
                break
        editor.load_state_dict(best_state)
        # Learned heads are frozen only during phase two. Restore their state so
        # the checkpoint parameter count includes every learned editor weight.
        for parameter in editor.parameters():
            parameter.requires_grad_(True)
        audit = _site_metrics(
            editor,
            boundary=boundary,
            inputs=inputs,
            outputs=outputs,
            targets=targets,
            indices=audit_indices,
            mismatch=mismatch,
            history_target=history_target,
            task_target=task_target,
            label_swap=label_swap,
        )
        editor.eval()
        with torch.no_grad():
            protected_correction, _ = editor.correction(
                value=protected["outputs"][key],
                boundary_state=protected["boundary"][key],
                context_state=protected["inputs"][key],
            )
        protected_relative = torch.sqrt(
            _relative_squared(protected_correction, protected["outputs"][key])
        )
        audit["protected_forced_on_mean_relative_correction"] = float(
            protected_relative.mean()
        )
        audit["protected_forced_on_max_relative_correction"] = float(
            protected_relative.max()
        )
        gates = contract["fit_gates"]
        gate_checks = {
            "normalized_teacher_mse": float(audit["normalized_teacher_mse"])
            <= float(gates["audit_fold_normalized_teacher_mse_maximum"]),
            "both_directions_positive": all(
                value is not None and value > 0
                for value in audit["teacher_mse_improvement_by_direction"].values()
            ),
            "both_label_swaps_positive": all(
                value is not None and value > 0
                for value in audit["teacher_mse_improvement_by_label_swap"].values()
            ),
            "history_accuracy": float(audit["history_balanced_accuracy"])
            >= float(gates["history_head_balanced_accuracy_minimum"]),
            "task_accuracy": float(audit["task_balanced_accuracy"])
            >= float(gates["task_head_balanced_accuracy_minimum"]),
            "protected_forced_on": float(audit["protected_forced_on_mean_relative_correction"])
            <= float(gates["protected_forced_on_mean_relative_correction_maximum"]),
            "trust_region": float(audit["maximum_relative_correction"])
            <= float(gates["maximum_observed_relative_correction_per_site"]),
        }
        site_report = {
            "site": key,
            "trainable_parameters": editor.trainable_parameter_count,
            "best_epoch": best_epoch,
            "best_calibration_metric": best_metric,
            "epochs_completed": len(epoch_reports),
            "head_input_mode": head_input_mode,
            "head_pretraining": head_pretraining_report,
            "audit": audit,
            "gate_checks": gate_checks,
            "eligible": all(gate_checks.values()),
            "epoch_trace": epoch_reports,
        }
        site_reports.append(site_report)
        spec = GovernanceSite(
            int(site_config["layer"]),
            str(site_config["component"]),
            int(site_config["output_rank"]),
        )
        site_editors[spec] = editor
        checkpoint_sites.append(
            {
                "layer": spec.layer,
                "component": spec.component,
                "output_rank": spec.rank,
                "maximum_relative_correction": float(site_config["max_relative_correction"]),
                "hidden_width": 96,
                "temperature": 1.0,
                "head_input_mode": head_input_mode,
                "constructor_tensors": {
                    "boundary_basis": editor.boundary_basis.detach().cpu(),
                    "context_basis": editor.context_basis.detach().cpu(),
                    "output_basis": editor.output_basis.detach().cpu(),
                    "boundary_center": editor.boundary_center.detach().cpu(),
                    "context_center": editor.context_center.detach().cpu(),
                    "boundary_scale": editor.boundary_scale.detach().cpu(),
                    "context_scale": editor.context_scale.detach().cpu(),
                    "boundary_head_scale": editor.boundary_head_scale,
                    "context_head_scale": editor.context_head_scale,
                },
                "state_dict": {
                    name: value.detach().cpu() for name, value in editor.state_dict().items()
                },
            }
        )

    multisite = AdaptiveMultiSiteGovernanceEditor(site_editors)
    parameter_budget = int(
        contract["architecture"].get(
            "total_learned_editor_parameter_upper_bound",
            contract["architecture"].get("total_trainable_parameter_upper_bound", -1),
        )
    )
    if parameter_budget <= 0:
        raise ValueError("editor contract has no valid parameter budget")
    if multisite.trainable_parameter_count > parameter_budget:
        raise RuntimeError("governance editor exceeds the frozen parameter budget")
    checkpoint = {
        "schema_version": 1,
        "kind": "adaptive_governance_editor",
        "method": contract["method_short_name"],
        "head_input_mode": head_input_mode,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_manifest),
        "trainable_parameter_count": multisite.trainable_parameter_count,
        "base_model_weights_included": False,
        "sites": checkpoint_sites,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    args.output_dir.mkdir(parents=True)
    checkpoint_path = args.output_dir / "adaptive_governance_editor.pt"
    incoming = args.output_dir / ".adaptive_governance_editor.pt.incoming"
    torch.save(checkpoint, incoming)
    os.replace(incoming, checkpoint_path)
    gates = contract["fit_gates"]
    aggregate_gate_checks = {
        "mean_normalized_teacher_mse": sum(
            float(report["audit"]["normalized_teacher_mse"]) for report in site_reports
        )
        / len(site_reports)
        <= float(gates["audit_fold_normalized_teacher_mse_maximum"]),
        "both_directions_positive": all(
            sum(
                float(report["audit"]["teacher_mse_improvement_by_direction"][direction])
                for report in site_reports
            )
            / len(site_reports)
            > 0
            for direction in ("independent_verification", "delegated_choice")
        ),
        "both_label_swaps_positive": all(
            sum(
                float(report["audit"]["teacher_mse_improvement_by_label_swap"][swap])
                for report in site_reports
            )
            / len(site_reports)
            > 0
            for swap in ("0", "1")
        ),
        "history_accuracy": min(
            float(report["audit"]["history_balanced_accuracy"]) for report in site_reports
        )
        >= float(gates["history_head_balanced_accuracy_minimum"]),
        "task_accuracy": min(
            float(report["audit"]["task_balanced_accuracy"]) for report in site_reports
        )
        >= float(gates["task_head_balanced_accuracy_minimum"]),
        "protected_forced_on": sum(
            float(report["audit"]["protected_forced_on_mean_relative_correction"])
            for report in site_reports
        )
        / len(site_reports)
        <= float(gates["protected_forced_on_mean_relative_correction_maximum"]),
        "trust_region": all(
            float(report["audit"]["maximum_relative_correction"])
            <= float(gates["maximum_observed_relative_correction_per_site"])
            for report in site_reports
        ),
    }
    eligible = all(aggregate_gate_checks.values())
    report = {
        "schema_version": 1,
        "method": contract["method_short_name"],
        "head_input_mode": head_input_mode,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_manifest),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "trainable_parameter_count": multisite.trainable_parameter_count,
        "parameter_budget": parameter_budget,
        "training_phases": (
            ["supervised_full_state_head_pretraining", "frozen_head_correction_fit"]
            if v2_fit
            else ["joint_pca_head_and_correction_fit"]
        ),
        "fold_rows": {
            "train": len(train_indices),
            "calibration": len(calibration_indices),
            "audit": len(audit_indices),
        },
        "site_reports": site_reports,
        "aggregate_gate_checks": aggregate_gate_checks,
        "all_fit_gates_pass": eligible,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "fit_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not eligible:
        raise SystemExit("adaptive governance editor failed one or more frozen fit gates")


if __name__ == "__main__":
    main()
