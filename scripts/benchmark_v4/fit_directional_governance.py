#!/usr/bin/env python3
"""Fit three independent DSGE-V3 single-site candidates.

The fit is deliberately behavior-proxy skeptical: task-margin gradients drive
the output bases and primary loss, while matched-teacher activation recovery is
only an auxiliary loss and reported diagnostic.  Every candidate checkpoint is
materialized even when its fit gates fail so the subsequent frozen fold-7
behavior characterization can complete rather than producing a proxy-only
half experiment.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v1.operators import remove_protected_subspace
from scripts.benchmark_v3.adaptive_governance import safe_scale
from scripts.benchmark_v3.fit_governance_editor import (
    _balanced_accuracy,
    _cpu_adamw,
    _fold,
    _load_capture,
    _load_protected,
    _matched_history,
    _pca_basis,
    _scalar_rms_scale,
    _teacher_indices,
)
from scripts.benchmark_v4.directional_governance import (
    DirectionalGovernanceSiteEditor,
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_gradient_capture(
    manifest_path: Path,
    site_keys: list[str],
    editor_contract_path: Path,
) -> dict:
    manifest = json.loads(manifest_path.read_text())
    required = {
        "stage": "governance_gradient_capture_full",
        "partition": "subspace_fit",
        "rows": 6144,
        "unique_job_keys": 6144,
        "complete": True,
        "base_model_parameter_gradients": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "editor_contract_sha256": sha256_file(editor_contract_path),
    }
    for field, expected in required.items():
        if manifest.get(field) != expected:
            raise ValueError(f"gradient manifest mismatch: {field}")
    if manifest.get("observed_key_sha256") != manifest.get("expected_key_sha256"):
        raise ValueError("gradient manifest key SHA is not closed")
    manifest_sites = sorted(
        f"{int(site['layer'])}:{site['component']}" for site in manifest.get("sites", [])
    )
    if sorted(site_keys) != manifest_sites:
        raise ValueError("gradient manifest site set mismatch")

    metadata: list[dict] = []
    gradients = {key: [] for key in site_keys}
    observed: set[str] = set()
    for record in manifest.get("shards", []):
        path = manifest_path.parent / record["file"]
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"gradient shard SHA mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        rows = shard["metadata"]
        if len(rows) != int(record["rows"]):
            raise ValueError(f"gradient shard row mismatch: {path}")
        for row in rows:
            key = str(row["job_key"])
            if key in observed:
                raise ValueError(f"duplicate gradient key: {key}")
            observed.add(key)
        metadata.extend(rows)
        for key in site_keys:
            value = shard["task_margin_gradients"][key].float()
            if value.ndim != 2 or value.shape[0] != len(rows):
                raise ValueError(f"gradient tensor shape mismatch: {path} {key}")
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite gradient tensor: {path} {key}")
            gradients[key].append(value)
    if len(metadata) != 6144 or len(observed) != 6144:
        raise ValueError("gradient capture does not contain 6,144 unique rows")
    ordered = sorted(range(len(metadata)), key=lambda index: metadata[index]["job_key"])
    order = torch.tensor(ordered, dtype=torch.long)
    return {
        "manifest": manifest,
        "metadata": [metadata[index] for index in ordered],
        "gradients": {
            key: torch.cat(values, dim=0).index_select(0, order)
            for key, values in gradients.items()
        },
    }


def _validate_row_alignment(capture: dict, gradient: dict) -> None:
    capture_rows = capture["metadata"]
    gradient_rows = gradient["metadata"]
    if len(capture_rows) != len(gradient_rows):
        raise ValueError("capture/gradient row counts differ")
    fields = (
        "job_key",
        "item_id",
        "partition",
        "history_condition",
        "task_requirement",
        "target_obedience",
        "label_swap",
        "task_correct_label",
        "task_foil_label",
    )
    for capture_row, gradient_row in zip(capture_rows, gradient_rows, strict=True):
        for field in fields:
            if capture_row.get(field) != gradient_row.get(field):
                raise ValueError(
                    f"capture/gradient metadata mismatch at {capture_row['job_key']}: {field}"
                )


def _validate_authorization(
    path: Path,
    *,
    editor_contract: Path,
    capture_manifest: Path,
    protected_capture_manifest: Path,
    gradient_manifest: Path,
) -> dict:
    authorization = json.loads(path.read_text())
    required = {
        "stage": "governance_directional_fit",
        "execution_allowed": True,
        "editor_contract_sha256": sha256_file(editor_contract),
        "capture_manifest_sha256": sha256_file(capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(protected_capture_manifest),
        "gradient_capture_manifest_sha256": sha256_file(gradient_manifest),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"directional-fit authorization mismatch: {field}")
    code_root = str(authorization.get("code_root", ""))
    if not code_root.startswith("/workspace/context-mismatch-qwen3-8b/code-v"):
        raise ValueError("directional-fit code root is invalid")
    version = int(code_root.rsplit("code-v", 1)[1])
    if version < 29:
        raise ValueError("directional fit requires code-v29 or newer")
    bundle = Path(code_root) / "bundle.sha256"
    if not bundle.is_file():
        raise ValueError("directional-fit immutable bundle manifest is missing")
    if authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(bundle):
        raise ValueError("directional-fit bundle SHA mismatch")
    return authorization


def _gradient_output_basis(
    gradients: torch.Tensor,
    protected_states: torch.Tensor,
    rank: int,
    *,
    protected_rank: int = 32,
) -> torch.Tensor:
    """Build a direction-specific basis from row-normalized margin gradients."""
    if gradients.ndim != 2 or protected_states.ndim != 2:
        raise ValueError("gradient/protected matrices must be rank two")
    if gradients.shape[1] != protected_states.shape[1]:
        raise ValueError("gradient/protected hidden dimensions differ")
    if rank <= 0 or protected_rank < 0:
        raise ValueError("invalid output/protected rank")
    norms = torch.linalg.vector_norm(gradients.float(), dim=-1, keepdim=True)
    if bool(torch.any(norms <= 0)):
        raise ValueError("zero task-margin gradient cannot define an output basis")
    normalized = gradients.float() / norms
    q = min(rank + 8, min(normalized.shape))
    if q < rank:
        raise ValueError("insufficient directional gradients for output basis")
    _, _, gradient_v = torch.pca_lowrank(
        normalized, q=q, center=False, niter=4
    )
    harmful = gradient_v[:, :rank]
    protected = None
    if protected_rank:
        centered = protected_states.float() - protected_states.float().mean(dim=0)
        protected_q = min(protected_rank + 8, min(centered.shape))
        if protected_q < protected_rank:
            raise ValueError("insufficient protected rows for protected subspace")
        _, _, protected_v = torch.pca_lowrank(
            centered, q=protected_q, center=False, niter=4
        )
        protected = protected_v[:, :protected_rank]
    basis = remove_protected_subspace(harmful, protected)
    if basis.shape != (gradients.shape[1], rank):
        raise RuntimeError("protected residualization did not retain the requested rank")
    return basis


def _fit_platt(
    raw_logit: torch.Tensor,
    target: torch.Tensor,
    *,
    learning_rate: float,
    steps: int,
    minimum_scale: float,
) -> tuple[float, float, list[dict]]:
    if raw_logit.ndim != 1 or target.shape != raw_logit.shape:
        raise ValueError("Platt inputs must be aligned vectors")
    location = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    initial = max(1.0 - minimum_scale, 1e-4)
    log_scale = torch.nn.Parameter(torch.log(torch.expm1(torch.tensor(initial))))
    optimizer = torch.optim.Adam(
        [location, log_scale], lr=learning_rate, foreach=False, fused=False
    )
    trace = []
    for step in range(1, steps + 1):
        scale = F.softplus(log_scale) + minimum_scale
        calibrated = (raw_logit.float() - location) / scale
        loss = F.binary_cross_entropy_with_logits(calibrated, target.float())
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite Platt loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if location.grad is None or log_scale.grad is None:
            raise RuntimeError("missing Platt gradient")
        if not bool(torch.isfinite(location.grad)) or not bool(torch.isfinite(log_scale.grad)):
            raise FloatingPointError("non-finite Platt gradient")
        optimizer.step()
        if step in {1, steps} or step % 25 == 0:
            trace.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "location": float(location.detach()),
                    "scale": float((F.softplus(log_scale) + minimum_scale).detach()),
                }
            )
    scale = float((F.softplus(log_scale) + minimum_scale).detach())
    center = float(location.detach())
    if not math.isfinite(center) or not math.isfinite(scale) or scale <= 0:
        raise FloatingPointError("non-finite fitted Platt parameters")
    return center, scale, trace


def _fit_heads(
    editor: DirectionalGovernanceSiteEditor,
    *,
    boundary: torch.Tensor,
    inputs: torch.Tensor,
    train_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    history_target: torch.Tensor,
    task_target: torch.Tensor,
    config: dict,
    calibration_config: dict,
    batch_size: int,
    gradient_norm_clip: float,
    seed: int,
) -> dict:
    for parameter in editor.positive_expert.parameters():
        parameter.requires_grad_(False)
    for parameter in editor.negative_expert.parameters():
        parameter.requires_grad_(False)
    head_parameters = [*editor.history_head.parameters(), *editor.task_head.parameters()]
    optimizer = _cpu_adamw(
        head_parameters,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state = {
        "history": copy.deepcopy(editor.history_head.state_dict()),
        "task": copy.deepcopy(editor.task_head.state_dict()),
    }
    stale = 0
    trace = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator)
        ]
        train_losses = []
        editor.train()
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            history_logit, task_logit, _, _ = editor.governance_logits(
                boundary[indices], inputs[indices]
            )
            loss = F.binary_cross_entropy_with_logits(
                history_logit, history_target[indices]
            ) + F.binary_cross_entropy_with_logits(task_logit, task_target[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite governance-head loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(head_parameters, gradient_norm_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite governance-head gradient")
            optimizer.step()
            train_losses.append(float(loss.detach()))
        editor.eval()
        with torch.no_grad():
            history_logit, task_logit, _, _ = editor.governance_logits(
                boundary[calibration_indices], inputs[calibration_indices]
            )
            calibration_loss = F.binary_cross_entropy_with_logits(
                history_logit, history_target[calibration_indices]
            ) + F.binary_cross_entropy_with_logits(
                task_logit, task_target[calibration_indices]
            )
        value = float(calibration_loss)
        trace.append(
            {
                "epoch": epoch,
                "mean_train_loss": sum(train_losses) / len(train_losses),
                "calibration_loss": value,
            }
        )
        if value < best_loss - 1e-7:
            best_loss = value
            best_epoch = epoch
            best_state = {
                "history": copy.deepcopy(editor.history_head.state_dict()),
                "task": copy.deepcopy(editor.task_head.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    editor.history_head.load_state_dict(best_state["history"])
    editor.task_head.load_state_dict(best_state["task"])
    editor.eval()
    with torch.no_grad():
        history_raw, task_raw, _, _ = editor.governance_logits(
            boundary[calibration_indices], inputs[calibration_indices]
        )
    history_location, history_scale, history_trace = _fit_platt(
        history_raw.detach(),
        history_target[calibration_indices],
        learning_rate=float(calibration_config["learning_rate"]),
        steps=int(calibration_config["steps"]),
        minimum_scale=float(calibration_config["minimum_scale"]),
    )
    task_location, task_scale, task_trace = _fit_platt(
        task_raw.detach(),
        task_target[calibration_indices],
        learning_rate=float(calibration_config["learning_rate"]),
        steps=int(calibration_config["steps"]),
        minimum_scale=float(calibration_config["minimum_scale"]),
    )
    editor.history_logit_location.copy_(torch.tensor(history_location))
    editor.history_logit_scale.copy_(torch.tensor(history_scale))
    editor.task_logit_location.copy_(torch.tensor(task_location))
    editor.task_logit_scale.copy_(torch.tensor(task_scale))
    for parameter in head_parameters:
        parameter.requires_grad_(False)
    for parameter in editor.positive_expert.parameters():
        parameter.requires_grad_(True)
    for parameter in editor.negative_expert.parameters():
        parameter.requires_grad_(True)
    return {
        "best_epoch": best_epoch,
        "best_precalibration_loss": best_loss,
        "epochs_completed": len(trace),
        "epoch_trace": trace,
        "history_calibration": {
            "location": history_location,
            "scale": history_scale,
            "trace": history_trace,
        },
        "task_calibration": {
            "location": task_location,
            "scale": task_scale,
            "trace": task_trace,
        },
    }


def _label_swap_pairs(metadata: list[dict], indices: torch.Tensor) -> torch.Tensor:
    groups: dict[tuple, dict[int, int]] = {}
    for index in indices.tolist():
        row = metadata[index]
        key = (
            row["benchmark"],
            row["item_id"],
            row["declared_role"],
            row["history_condition"],
            row["history_style"],
            int(row["history_realization"]),
            row["task_requirement"],
        )
        groups.setdefault(key, {})[int(row["label_swap"])] = index
    if not groups or any(set(value) != {0, 1} for value in groups.values()):
        raise ValueError("label-swap fit pairs are incomplete")
    return torch.tensor(
        [[value[0], value[1]] for _, value in sorted(groups.items())],
        dtype=torch.long,
    )


def _gain_fraction(gain: torch.Tensor, target_gap: torch.Tensor) -> torch.Tensor:
    denominator = torch.clamp(target_gap, min=0.25)
    return gain / denominator


@dataclass(frozen=True)
class _LowRankSiteData:
    gradient_positive: torch.Tensor
    gradient_negative: torch.Tensor
    target_positive: torch.Tensor
    target_negative: torch.Tensor
    target_norm_squared: torch.Tensor
    output_norm_squared: torch.Tensor
    output_norm: torch.Tensor
    protected_output_norm_squared: torch.Tensor
    protected_output_norm: torch.Tensor


def _precompute_low_rank_site_data(
    editor: DirectionalGovernanceSiteEditor,
    *,
    gradients: torch.Tensor,
    targets: torch.Tensor,
    outputs: torch.Tensor,
    protected_outputs: torch.Tensor,
) -> _LowRankSiteData:
    """Project fixed fit tensors once into the two directional output bases."""
    positive_basis = editor.positive_output_basis.float()
    negative_basis = editor.negative_output_basis.float()
    target_norm_squared = torch.sum(targets.float().square(), dim=-1)
    output_norm_squared = torch.sum(outputs.float().square(), dim=-1)
    protected_output_norm_squared = torch.sum(
        protected_outputs.float().square(), dim=-1
    )
    values = (
        gradients,
        targets,
        outputs,
        protected_outputs,
        positive_basis,
        negative_basis,
    )
    if not all(bool(torch.isfinite(value).all()) for value in values):
        raise FloatingPointError("non-finite low-rank site input")
    return _LowRankSiteData(
        gradient_positive=gradients.float() @ positive_basis,
        gradient_negative=gradients.float() @ negative_basis,
        target_positive=targets.float() @ positive_basis,
        target_negative=targets.float() @ negative_basis,
        target_norm_squared=target_norm_squared,
        output_norm_squared=output_norm_squared,
        output_norm=torch.sqrt(output_norm_squared),
        protected_output_norm_squared=protected_output_norm_squared,
        protected_output_norm=torch.sqrt(protected_output_norm_squared),
    )


def _coordinate_squared_norm(
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> torch.Tensor:
    if positive.shape[0] != negative.shape[0]:
        raise ValueError("directional coordinate batches differ")
    return torch.sum(positive.float().square(), dim=-1) + torch.sum(
        negative.float().square(), dim=-1
    )


def _coordinate_dot(
    positive: torch.Tensor,
    negative: torch.Tensor,
    projected_positive: torch.Tensor,
    projected_negative: torch.Tensor,
) -> torch.Tensor:
    if positive.shape != projected_positive.shape:
        raise ValueError("positive projected tensor shape mismatch")
    if negative.shape != projected_negative.shape:
        raise ValueError("negative projected tensor shape mismatch")
    return torch.sum(positive.float() * projected_positive.float(), dim=-1) + torch.sum(
        negative.float() * projected_negative.float(), dim=-1
    )


def _coordinate_relative_squared(
    positive: torch.Tensor,
    negative: torch.Tensor,
    reference_norm_squared: torch.Tensor,
) -> torch.Tensor:
    return _coordinate_squared_norm(positive, negative) / torch.clamp(
        reference_norm_squared.float(), min=1e-6
    )


def _coordinate_reconstruction_relative_squared(
    positive: torch.Tensor,
    negative: torch.Tensor,
    target_positive: torch.Tensor,
    target_negative: torch.Tensor,
    target_norm_squared: torch.Tensor,
) -> torch.Tensor:
    correction_norm_squared = _coordinate_squared_norm(positive, negative)
    target_dot_correction = _coordinate_dot(
        positive,
        negative,
        target_positive,
        target_negative,
    )
    error_norm_squared = torch.clamp(
        correction_norm_squared
        + target_norm_squared.float()
        - 2.0 * target_dot_correction,
        min=0.0,
    )
    return error_norm_squared / torch.clamp(
        target_norm_squared.float(), min=1e-6
    )


def _expert_validation_metric(
    editor: DirectionalGovernanceSiteEditor,
    *,
    low_rank: _LowRankSiteData,
    boundary: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    matched_margins: torch.Tensor,
    margins: torch.Tensor,
    indices: torch.Tensor,
    pairs: torch.Tensor,
    protected: dict,
    site_key: str,
    target_fraction: float,
    target_clip: float,
    protected_weight: float,
) -> float:
    editor.eval()
    with torch.no_grad():
        positive, negative, _ = editor.correction_coordinates(
            value=outputs[indices],
            boundary_state=boundary[indices],
            context_state=inputs[indices],
            reference_norm=low_rank.output_norm[indices],
        )
        gain = _coordinate_dot(
            positive,
            negative,
            low_rank.gradient_positive[indices],
            low_rank.gradient_negative[indices],
        )
        gap = torch.clamp(
            matched_margins[indices] - margins[indices], min=0.0, max=target_clip
        )
        primary = torch.relu(target_fraction * gap - gain).square().mean()
        pair_flat = pairs.reshape(-1)
        pair_positive, pair_negative, _ = editor.correction_coordinates(
            value=outputs[pair_flat],
            boundary_state=boundary[pair_flat],
            context_state=inputs[pair_flat],
            reference_norm=low_rank.output_norm[pair_flat],
        )
        pair_gain = _coordinate_dot(
            pair_positive,
            pair_negative,
            low_rank.gradient_positive[pair_flat],
            low_rank.gradient_negative[pair_flat],
        )
        pair_gap = torch.clamp(
            matched_margins[pair_flat] - margins[pair_flat], min=0.0, max=target_clip
        )
        fractions = _gain_fraction(pair_gain, pair_gap).reshape(-1, 2)
        orbit = (fractions[:, 0] - fractions[:, 1]).square().mean()
        protected_directional_losses = []
        for direction in ("positive", "negative"):
            protected_positive, protected_negative, _ = (
                editor.forced_direction_coordinates(
                    direction=direction,
                    value=protected["outputs"][site_key],
                    boundary_state=protected["boundary"][site_key],
                    context_state=protected["inputs"][site_key],
                    reference_norm=low_rank.protected_output_norm,
                )
            )
            protected_directional_losses.append(
                _coordinate_relative_squared(
                    protected_positive,
                    protected_negative,
                    low_rank.protected_output_norm_squared,
                ).mean()
            )
        protected_loss = torch.stack(protected_directional_losses).max()
        teacher = _coordinate_reconstruction_relative_squared(
            positive,
            negative,
            low_rank.target_positive[indices],
            low_rank.target_negative[indices],
            low_rank.target_norm_squared[indices],
        ).mean()
        metric = primary + 0.25 * orbit + protected_weight * protected_loss + 0.05 * teacher
    if not bool(torch.isfinite(metric)):
        raise FloatingPointError("non-finite expert validation metric")
    return float(metric)


def _fit_experts(
    editor: DirectionalGovernanceSiteEditor,
    *,
    low_rank: _LowRankSiteData,
    metadata: list[dict],
    boundary: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    matched_margins: torch.Tensor,
    margins: torch.Tensor,
    mismatch: torch.Tensor,
    train_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    protected: dict,
    site_key: str,
    optimization: dict,
    batch_size: int,
    gradient_norm_clip: float,
    seed: int,
) -> dict:
    config = optimization["expert_fit"]
    weights = config["loss_weights"]
    target_fraction = float(optimization["primary_target_fraction_of_positive_gap"])
    target_clip = float(optimization["primary_target_gap_clip"])
    train_mismatch = train_indices[mismatch[train_indices]]
    calibration_mismatch = calibration_indices[mismatch[calibration_indices]]
    train_pairs = _label_swap_pairs(metadata, train_mismatch)
    calibration_pairs = _label_swap_pairs(metadata, calibration_mismatch)
    parameters = [
        *editor.positive_expert.parameters(),
        *editor.negative_expert.parameters(),
    ]
    optimizer = _cpu_adamw(
        parameters,
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    generator = torch.Generator().manual_seed(seed)
    protected_count = protected["outputs"][site_key].shape[0]
    best_metric = math.inf
    best_epoch = 0
    best_state = {
        "positive": copy.deepcopy(editor.positive_expert.state_dict()),
        "negative": copy.deepcopy(editor.negative_expert.state_dict()),
    }
    stale = 0
    trace = []
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        editor.train()
        permutation = train_mismatch[
            torch.randperm(len(train_mismatch), generator=generator)
        ]
        pair_order = train_pairs[
            torch.randperm(len(train_pairs), generator=generator)
        ]
        losses = []
        gradient_norms = []
        for batch_number, offset in enumerate(range(0, len(permutation), batch_size)):
            indices = permutation[offset : offset + batch_size]
            positive, negative, _ = editor.correction_coordinates(
                value=outputs[indices],
                boundary_state=boundary[indices],
                context_state=inputs[indices],
                reference_norm=low_rank.output_norm[indices],
            )
            gain = _coordinate_dot(
                positive,
                negative,
                low_rank.gradient_positive[indices],
                low_rank.gradient_negative[indices],
            )
            gap = torch.clamp(
                matched_margins[indices] - margins[indices], min=0.0, max=target_clip
            )
            primary = torch.relu(target_fraction * gap - gain).square().mean()
            teacher = _coordinate_reconstruction_relative_squared(
                positive,
                negative,
                low_rank.target_positive[indices],
                low_rank.target_negative[indices],
                low_rank.target_norm_squared[indices],
            ).mean()

            pair_count = max(1, min(len(indices) // 2, len(pair_order)))
            start = (batch_number * pair_count) % len(pair_order)
            selected_pairs = torch.cat((pair_order, pair_order), dim=0)[
                start : start + pair_count
            ]
            pair_flat = selected_pairs.reshape(-1)
            pair_positive, pair_negative, _ = editor.correction_coordinates(
                value=outputs[pair_flat],
                boundary_state=boundary[pair_flat],
                context_state=inputs[pair_flat],
                reference_norm=low_rank.output_norm[pair_flat],
            )
            pair_gain = _coordinate_dot(
                pair_positive,
                pair_negative,
                low_rank.gradient_positive[pair_flat],
                low_rank.gradient_negative[pair_flat],
            )
            pair_gap = torch.clamp(
                matched_margins[pair_flat] - margins[pair_flat],
                min=0.0,
                max=target_clip,
            )
            fraction = _gain_fraction(pair_gain, pair_gap).reshape(-1, 2)
            orbit = (fraction[:, 0] - fraction[:, 1]).square().mean()

            protected_indices = torch.randint(
                0, protected_count, (len(indices),), generator=generator
            )
            protected_directional_losses = []
            for direction in ("positive", "negative"):
                protected_positive, protected_negative, _ = (
                    editor.forced_direction_coordinates(
                        direction=direction,
                        value=protected["outputs"][site_key][protected_indices],
                        boundary_state=protected["boundary"][site_key][protected_indices],
                        context_state=protected["inputs"][site_key][protected_indices],
                        reference_norm=low_rank.protected_output_norm[protected_indices],
                    )
                )
                protected_directional_losses.append(
                    _coordinate_relative_squared(
                        protected_positive,
                        protected_negative,
                        low_rank.protected_output_norm_squared[protected_indices],
                    ).mean()
                )
            protected_loss = torch.stack(protected_directional_losses).max()
            intervention = _coordinate_relative_squared(
                positive,
                negative,
                low_rank.output_norm_squared[indices],
            ).mean()
            loss = (
                float(weights["true_downstream_first_order_gain"]) * primary
                + float(weights["directional_teacher_activation_reconstruction"]) * teacher
                + float(weights["paired_label_swap_orbit"]) * orbit
                + float(weights["protected_zero_correction"]) * protected_loss
                + float(weights["intervention_norm"]) * intervention
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite expert loss at {site_key} epoch {epoch}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_norm_clip)
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError(
                    f"non-finite expert gradient at {site_key} epoch {epoch}"
                )
            optimizer.step()
            losses.append(float(loss.detach()))
            gradient_norms.append(float(norm.detach()))
        metric = _expert_validation_metric(
            editor,
            low_rank=low_rank,
            boundary=boundary,
            inputs=inputs,
            outputs=outputs,
            matched_margins=matched_margins,
            margins=margins,
            indices=calibration_mismatch,
            pairs=calibration_pairs,
            protected=protected,
            site_key=site_key,
            target_fraction=target_fraction,
            target_clip=target_clip,
            protected_weight=float(weights["protected_zero_correction"]),
        )
        trace.append(
            {
                "epoch": epoch,
                "mean_train_loss": sum(losses) / len(losses),
                "mean_gradient_norm": sum(gradient_norms) / len(gradient_norms),
                "calibration_metric": metric,
            }
        )
        if metric < best_metric - 1e-7:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                "positive": copy.deepcopy(editor.positive_expert.state_dict()),
                "negative": copy.deepcopy(editor.negative_expert.state_dict()),
            }
            stale = 0
        else:
            stale += 1
        if stale >= int(config["early_stopping_patience"]):
            break
    editor.positive_expert.load_state_dict(best_state["positive"])
    editor.negative_expert.load_state_dict(best_state["negative"])
    return {
        "best_epoch": best_epoch,
        "best_calibration_metric": best_metric,
        "epochs_completed": len(trace),
        "trace": trace,
    }


def _subset_metrics(
    editor: DirectionalGovernanceSiteEditor,
    *,
    low_rank: _LowRankSiteData,
    boundary: torch.Tensor,
    inputs: torch.Tensor,
    outputs: torch.Tensor,
    indices: torch.Tensor,
    mismatch: torch.Tensor,
    history_target: torch.Tensor,
    task_target: torch.Tensor,
    label_swap: torch.Tensor,
) -> dict:
    editor.eval()
    with torch.no_grad():
        positive, negative, diagnostics = editor.correction_coordinates(
            value=outputs[indices],
            boundary_state=boundary[indices],
            context_state=inputs[indices],
            reference_norm=low_rank.output_norm[indices],
        )
    current_mismatch = mismatch[indices]
    current_matched = ~current_mismatch
    gain = _coordinate_dot(
        positive,
        negative,
        low_rank.gradient_positive[indices],
        low_rank.gradient_negative[indices],
    )
    teacher_improvement = 1.0 - _coordinate_reconstruction_relative_squared(
        positive,
        negative,
        low_rank.target_positive[indices],
        low_rank.target_negative[indices],
        low_rank.target_norm_squared[indices],
    )

    def category_means(values: torch.Tensor, category: torch.Tensor) -> dict[str, float | None]:
        result = {}
        for value in (0, 1):
            mask = current_mismatch & (category == value)
            result[str(value)] = float(values[mask].mean()) if bool(mask.any()) else None
        return result

    route_active = diagnostics["structural_gate_active"]
    relative = torch.sqrt(
        _coordinate_relative_squared(
            positive,
            negative,
            low_rank.output_norm_squared[indices],
        )
    )
    return {
        "rows": len(indices),
        "history_balanced_accuracy": _balanced_accuracy(
            (diagnostics["history_probability"] >= 0.5).float(),
            history_target[indices],
        ),
        "task_balanced_accuracy": _balanced_accuracy(
            (diagnostics["task_probability"] >= 0.5).float(),
            task_target[indices],
        ),
        "matched_route_active_fraction": (
            float(route_active[current_matched].float().mean())
            if bool(current_matched.any())
            else None
        ),
        "mismatch_route_active_fraction": (
            float(route_active[current_mismatch].float().mean())
            if bool(current_mismatch.any())
            else None
        ),
        "first_order_gain_by_direction": category_means(
            gain, task_target[indices].long()
        ),
        "first_order_gain_by_label_swap": category_means(
            gain, label_swap[indices]
        ),
        "teacher_reconstruction_improvement_by_direction": category_means(
            teacher_improvement, task_target[indices].long()
        ),
        "teacher_reconstruction_improvement_by_label_swap": category_means(
            teacher_improvement, label_swap[indices]
        ),
        "mean_relative_correction": float(relative.mean()),
        "maximum_relative_correction": float(relative.max()),
    }


def _protected_metrics(
    editor: DirectionalGovernanceSiteEditor,
    low_rank: _LowRankSiteData,
    protected: dict,
    site_key: str,
) -> dict:
    editor.eval()
    with torch.no_grad():
        _, _, diagnostics = editor.correction_coordinates(
            value=protected["outputs"][site_key],
            boundary_state=protected["boundary"][site_key],
            context_state=protected["inputs"][site_key],
            reference_norm=low_rank.protected_output_norm,
        )
        directional_relative = {}
        for direction in ("positive", "negative"):
            positive, negative, _ = editor.forced_direction_coordinates(
                direction=direction,
                value=protected["outputs"][site_key],
                boundary_state=protected["boundary"][site_key],
                context_state=protected["inputs"][site_key],
                reference_norm=low_rank.protected_output_norm,
            )
            directional_relative[direction] = torch.sqrt(
                _coordinate_relative_squared(
                    positive,
                    negative,
                    low_rank.protected_output_norm_squared,
                )
            )
    directional_means = {
        direction: float(value.mean())
        for direction, value in directional_relative.items()
    }
    directional_maxima = {
        direction: float(value.max())
        for direction, value in directional_relative.items()
    }
    return {
        "rows": len(low_rank.protected_output_norm),
        "route_active_fraction": float(
            diagnostics["structural_gate_active"].float().mean()
        ),
        "forced_on_mean_relative_correction_by_direction": directional_means,
        "forced_on_max_relative_correction_by_direction": directional_maxima,
        "forced_on_mean_relative_correction": max(directional_means.values()),
        "forced_on_max_relative_correction": max(directional_maxima.values()),
    }


def _checkpoint(
    editor: DirectionalGovernanceSiteEditor,
    site: dict,
    *,
    contract_sha256: str,
    capture_sha256: str,
    protected_sha256: str,
    gradient_sha256: str,
) -> dict:
    for parameter in editor.parameters():
        parameter.requires_grad_(True)
    record = {
        "layer": int(site["layer"]),
        "component": str(site["component"]),
        "positive_output_rank": int(site["positive_output_rank"]),
        "negative_output_rank": int(site["negative_output_rank"]),
        "maximum_relative_correction": float(site["max_relative_correction"]),
        "activation_threshold": float(editor.activation_threshold),
        "hidden_width": int(editor.hidden_width),
        "head_input_mode": str(editor.head_input_mode),
        "constructor_tensors": {
            "boundary_basis": editor.boundary_basis.detach().cpu(),
            "context_basis": editor.context_basis.detach().cpu(),
            "positive_output_basis": editor.positive_output_basis.detach().cpu(),
            "negative_output_basis": editor.negative_output_basis.detach().cpu(),
            "boundary_center": editor.boundary_center.detach().cpu(),
            "context_center": editor.context_center.detach().cpu(),
            "boundary_scale": editor.boundary_scale.detach().cpu(),
            "context_scale": editor.context_scale.detach().cpu(),
            "boundary_head_scale": editor.boundary_head_scale,
            "context_head_scale": editor.context_head_scale,
            "history_logit_location": float(editor.history_logit_location),
            "history_logit_scale": float(editor.history_logit_scale),
            "task_logit_location": float(editor.task_logit_location),
            "task_logit_scale": float(editor.task_logit_scale),
        },
        "state_dict": {
            name: value.detach().cpu() for name, value in editor.state_dict().items()
        },
    }
    return {
        "schema_version": 2,
        "kind": "directional_structural_governance_editor",
        "method": "DSGE-V3",
        "editor_contract_sha256": contract_sha256,
        "capture_manifest_sha256": capture_sha256,
        "protected_capture_manifest_sha256": protected_sha256,
        "gradient_capture_manifest_sha256": gradient_sha256,
        "trainable_parameter_count": editor.trainable_parameter_count,
        "base_model_weights_included": False,
        "sites": [record],
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--gradient-capture-manifest", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing directional fit output: {args.output_dir}")

    contract = json.loads(args.editor_contract.read_text())
    required_contract = {
        "status": "design_locked_after_v2_terminal_failure_audit_before_any_v3_forward",
        "method_short_name": "DSGE-V3",
        "code_version_minimum": 29,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"directional contract mismatch: {field}")
    bound_inputs = contract.get("bound_inputs", {})
    bound_manifests = {
        "governance_capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(
            args.protected_capture_manifest
        ),
    }
    for field, expected in bound_manifests.items():
        if bound_inputs.get(field) != expected:
            raise ValueError(f"directional contract input binding mismatch: {field}")
    capture_manifest_value = json.loads(args.capture_manifest.read_text())
    for field, expected in {
        "stage": "governance_capture_full",
        "partition": "subspace_fit",
        "rows": 6144,
        "unique_job_keys": 6144,
        "complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if capture_manifest_value.get(field) != expected:
            raise ValueError(f"governance capture manifest mismatch: {field}")
    if capture_manifest_value.get("observed_key_sha256") != capture_manifest_value.get(
        "expected_key_sha256"
    ):
        raise ValueError("governance capture key SHA is not closed")
    protected_manifest_value = json.loads(args.protected_capture_manifest.read_text())
    for field, expected in {
        "partition": "subspace_fit",
        "rows": 4008,
        "final_test_open": False,
        "production_rollout_approved": False,
    }.items():
        if protected_manifest_value.get(field) != expected:
            raise ValueError(f"protected capture manifest mismatch: {field}")
    _validate_authorization(
        args.execution_authorization,
        editor_contract=args.editor_contract,
        capture_manifest=args.capture_manifest,
        protected_capture_manifest=args.protected_capture_manifest,
        gradient_manifest=args.gradient_capture_manifest,
    )

    seed = int(contract["optimization"]["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    sites = sorted(contract["candidate_sites"], key=lambda row: int(row["layer"]))
    site_keys = [f"{int(site['layer'])}:{site['component']}" for site in sites]
    capture = _load_capture(args.capture_manifest.parent, args.capture_manifest, site_keys)
    protected = _load_protected(
        args.protected_capture_manifest.parent,
        args.protected_capture_manifest,
        site_keys,
    )
    gradient = _load_gradient_capture(
        args.gradient_capture_manifest, site_keys, args.editor_contract
    )
    _validate_row_alignment(capture, gradient)
    metadata = capture["metadata"]
    teacher_indices = _teacher_indices(metadata)
    folds = torch.tensor([_fold(row["item_id"]) for row in metadata], dtype=torch.long)
    train_indices = torch.nonzero(folds <= 5, as_tuple=False).squeeze(-1)
    calibration_indices = torch.nonzero(folds == 6, as_tuple=False).squeeze(-1)
    audit_indices = torch.nonzero(folds == 7, as_tuple=False).squeeze(-1)
    if min(len(train_indices), len(calibration_indices), len(audit_indices)) <= 0:
        raise RuntimeError("one or more directional fit folds are empty")
    if any(str(row["partition"]) != "subspace_fit" for row in metadata):
        raise ValueError("directional fit capture leaks a non-fit partition")

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
    batch_size = int(optimization["batch_size"])
    gradient_norm_clip = float(optimization["gradient_norm_clip"])
    hidden_width = int(optimization["expert_fit"]["hidden_width"])
    activation_threshold = float(contract["router"]["activation_threshold"])
    gates = contract["fit_gates_per_candidate_without_cross_site_averaging"]
    contract_sha = sha256_file(args.editor_contract)
    capture_sha = sha256_file(args.capture_manifest)
    protected_sha = sha256_file(args.protected_capture_manifest)
    gradient_sha = sha256_file(args.gradient_capture_manifest)

    args.output_dir.mkdir(parents=True)
    candidate_reports = []
    for site_number, site in enumerate(sites):
        torch.manual_seed(seed + site_number)
        key = site_keys[site_number]
        boundary = capture["boundary"][key]
        inputs = capture["inputs"][key]
        outputs = capture["outputs"][key]
        gradients = gradient["gradients"][key]
        targets = outputs.index_select(0, teacher_indices) - outputs
        train_mismatch = train_indices[mismatch[train_indices]]
        positive = train_mismatch[
            (history_target[train_mismatch] == 1) & (task_target[train_mismatch] == 0)
        ]
        negative = train_mismatch[
            (history_target[train_mismatch] == 0) & (task_target[train_mismatch] == 1)
        ]
        if min(len(positive), len(negative)) <= 0:
            raise RuntimeError(f"directional training cells are empty for {key}")
        boundary_center = boundary[train_indices].mean(dim=0)
        context_center = inputs[train_indices].mean(dim=0)
        boundary_basis = _pca_basis(boundary[train_indices], int(site["boundary_rank"]))
        context_basis = _pca_basis(inputs[train_indices], int(site["context_rank"]))
        positive_basis = _gradient_output_basis(
            gradients[positive],
            protected["outputs"][key],
            int(site["positive_output_rank"]),
        )
        negative_basis = _gradient_output_basis(
            gradients[negative],
            protected["outputs"][key],
            int(site["negative_output_rank"]),
        )
        boundary_coordinates = (boundary[train_indices] - boundary_center) @ boundary_basis
        context_coordinates = (inputs[train_indices] - context_center) @ context_basis
        editor = DirectionalGovernanceSiteEditor(
            boundary_basis=boundary_basis,
            context_basis=context_basis,
            positive_output_basis=positive_basis,
            negative_output_basis=negative_basis,
            boundary_center=boundary_center,
            context_center=context_center,
            boundary_scale=safe_scale(boundary_coordinates),
            context_scale=safe_scale(context_coordinates),
            maximum_relative_correction=float(site["max_relative_correction"]),
            activation_threshold=activation_threshold,
            hidden_width=hidden_width,
            head_input_mode="full_state",
            boundary_head_scale=_scalar_rms_scale(boundary[train_indices], boundary_center),
            context_head_scale=_scalar_rms_scale(inputs[train_indices], context_center),
        )
        low_rank = _precompute_low_rank_site_data(
            editor,
            gradients=gradients,
            targets=targets,
            outputs=outputs,
            protected_outputs=protected["outputs"][key],
        )
        head_report = _fit_heads(
            editor,
            boundary=boundary,
            inputs=inputs,
            train_indices=train_indices,
            calibration_indices=calibration_indices,
            history_target=history_target,
            task_target=task_target,
            config=optimization["history_and_task_heads_fit_then_frozen"],
            calibration_config=optimization["head_calibration_fit_then_frozen"],
            batch_size=batch_size,
            gradient_norm_clip=gradient_norm_clip,
            seed=seed + 100 + site_number,
        )
        expert_report = _fit_experts(
            editor,
            low_rank=low_rank,
            metadata=metadata,
            boundary=boundary,
            inputs=inputs,
            outputs=outputs,
            matched_margins=matched_margins,
            margins=margins,
            mismatch=mismatch,
            train_indices=train_indices,
            calibration_indices=calibration_indices,
            protected=protected,
            site_key=key,
            optimization=optimization,
            batch_size=batch_size,
            gradient_norm_clip=gradient_norm_clip,
            seed=seed + 1000 + site_number,
        )
        calibration = _subset_metrics(
            editor,
            low_rank=low_rank,
            boundary=boundary,
            inputs=inputs,
            outputs=outputs,
            indices=calibration_indices,
            mismatch=mismatch,
            history_target=history_target,
            task_target=task_target,
            label_swap=label_swap,
        )
        audit = _subset_metrics(
            editor,
            low_rank=low_rank,
            boundary=boundary,
            inputs=inputs,
            outputs=outputs,
            indices=audit_indices,
            mismatch=mismatch,
            history_target=history_target,
            task_target=task_target,
            label_swap=label_swap,
        )
        protected_metrics = _protected_metrics(editor, low_rank, protected, key)
        parameter_count = sum(parameter.numel() for parameter in editor.parameters())
        gate_checks = {
            "history_head_balanced_accuracy": min(
                float(calibration["history_balanced_accuracy"]),
                float(audit["history_balanced_accuracy"]),
            )
            >= float(gates["history_head_balanced_accuracy_minimum"]),
            "task_head_balanced_accuracy": min(
                float(calibration["task_balanced_accuracy"]),
                float(audit["task_balanced_accuracy"]),
            )
            >= float(gates["task_head_balanced_accuracy_minimum"]),
            "calibration_matched_route_zero": float(
                calibration["matched_route_active_fraction"]
            )
            <= float(gates["calibration_fold_matched_route_active_fraction_maximum"]),
            "calibration_mismatch_route_active": float(
                calibration["mismatch_route_active_fraction"]
            )
            >= float(gates["calibration_fold_mismatch_route_active_fraction_minimum"]),
            "audit_matched_route_zero": float(audit["matched_route_active_fraction"])
            <= float(gates["audit_fold_matched_route_active_fraction_maximum"]),
            "audit_mismatch_route_active": float(audit["mismatch_route_active_fraction"])
            >= float(gates["audit_fold_mismatch_route_active_fraction_minimum"]),
            "both_direction_first_order_gain_positive": all(
                value is not None and float(value) > 0
                for value in audit["first_order_gain_by_direction"].values()
            ),
            "both_label_swap_first_order_gain_positive": all(
                value is not None and float(value) > 0
                for value in audit["first_order_gain_by_label_swap"].values()
            ),
            "protected_forced_on": float(
                protected_metrics["forced_on_mean_relative_correction"]
            )
            <= float(gates["protected_forced_on_mean_relative_correction_maximum"]),
            "trust_region": max(
                float(calibration["maximum_relative_correction"]),
                float(audit["maximum_relative_correction"]),
                float(protected_metrics["forced_on_max_relative_correction"]),
            )
            <= float(gates["maximum_observed_relative_correction"]),
            "parameter_budget": parameter_count
            <= int(contract["architecture"]["total_learned_editor_parameter_upper_bound"]),
        }
        eligible = all(gate_checks.values())
        checkpoint = _checkpoint(
            editor,
            site,
            contract_sha256=contract_sha,
            capture_sha256=capture_sha,
            protected_sha256=protected_sha,
            gradient_sha256=gradient_sha,
        )
        checkpoint_name = f"directional_editor__layer_{site['layer']}__{site['component']}.pt"
        checkpoint_path = args.output_dir / checkpoint_name
        incoming = args.output_dir / f".{checkpoint_name}.incoming"
        torch.save(checkpoint, incoming)
        os.replace(incoming, checkpoint_path)
        candidate_reports.append(
            {
                "candidate_id": f"DSGE-V3-{key}",
                "site": key,
                "checkpoint": checkpoint_name,
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "trainable_parameter_count": parameter_count,
                "head_fit": head_report,
                "expert_fit": expert_report,
                "calibration": calibration,
                "audit": audit,
                "protected": protected_metrics,
                "teacher_reconstruction_is_diagnostic_only": True,
                "gate_checks": gate_checks,
                "fit_eligible": eligible,
                "direct_behavior_characterization_required_even_if_ineligible": True,
            }
        )

    report = {
        "schema_version": 1,
        "stage": "governance_directional_fit",
        "method": "DSGE-V3",
        "fit_complete": True,
        "candidate_count": len(candidate_reports),
        "all_three_candidate_checkpoints_materialized": len(candidate_reports) == 3,
        "eligible_candidate_ids": [
            row["candidate_id"] for row in candidate_reports if row["fit_eligible"]
        ],
        "candidate_reports": candidate_reports,
        "editor_contract_sha256": contract_sha,
        "capture_manifest_sha256": capture_sha,
        "protected_capture_manifest_sha256": protected_sha,
        "gradient_capture_manifest_sha256": gradient_sha,
        "authorization_sha256": sha256_file(args.execution_authorization),
        "fold_rows": {
            "train": len(train_indices),
            "calibration": len(calibration_indices),
            "audit": len(audit_indices),
        },
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "directional_fit_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(_canonical(report))


if __name__ == "__main__":
    main()
