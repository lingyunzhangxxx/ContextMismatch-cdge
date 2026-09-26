#!/usr/bin/env python3
"""Fit method-faithful external-baseline adapters on subspace_fit only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v3.fit_governance_editor import _fold, _load_capture, _teacher_indices
from scripts.benchmark_v4.fit_directional_governance import (
    _load_gradient_capture,
    _validate_row_alignment,
)
from scripts.benchmark_v19.external_baselines import finite_checkpoint


SITE = "27:mlp"
CAPTURE_SITES = ("23:self_attn", "27:mlp", "31:mlp")


def _unit(value: torch.Tensor) -> torch.Tensor:
    return value.float() / torch.clamp(torch.linalg.vector_norm(value.float()), min=1e-12)


def _fit_loreft(
    current: torch.Tensor,
    teacher: torch.Tensor,
    task: torch.Tensor,
    fit: torch.Tensor,
    rank: int,
    ridge: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    delta = teacher[fit] - current[fit]
    _, _, basis = torch.pca_lowrank(delta, q=min(rank + 4, min(delta.shape)), center=True, niter=4)
    basis = basis[:, :rank].contiguous()
    weights = []
    biases = []
    residuals = {}
    for label in (0, 1):
        indices = fit[task[fit] == float(label)]
        x = current[indices].float()
        y = teacher[indices].float() @ basis
        design = torch.cat((x, torch.ones(x.shape[0], 1)), dim=1)
        gram = design.T @ design
        penalty = ridge * torch.eye(gram.shape[0])
        penalty[-1, -1] = 0.0
        solution = torch.linalg.solve(gram + penalty, design.T @ y)
        weights.append(solution[:-1].T.contiguous())
        biases.append(solution[-1].contiguous())
        predicted = x @ solution[:-1] + solution[-1]
        residuals[str(label)] = float((predicted - y).square().mean())
    return basis, torch.stack(weights), torch.stack(biases), residuals


def _fit_reps(
    gradients: torch.Tensor,
    margins: torch.Tensor,
    task: torch.Tensor,
    fit: torch.Tensor,
    *,
    beta: float,
    steps: int,
    learning_rate: float,
) -> tuple[torch.Tensor, dict]:
    vectors = []
    reports = {}
    for label in (0, 1):
        indices = fit[task[fit] == float(label)]
        g = gradients[indices].float()
        m = margins[indices].float()
        vector = torch.zeros(g.shape[1], requires_grad=True)
        first_moment = torch.zeros_like(vector)
        second_moment = torch.zeros_like(vector)
        beta1, beta2, epsilon, weight_decay = 0.9, 0.999, 1e-8, 1e-3
        losses = []
        for step in range(1, steps + 1):
            vector.grad = None
            preference_logit = beta * (m + g @ vector)
            loss = -F.logsigmoid(preference_logit).mean() + 1e-4 * vector.square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite RePS preference loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([vector], 5.0)
            with torch.no_grad():
                gradient_value = vector.grad.float()
                vector.mul_(1.0 - learning_rate * weight_decay)
                first_moment.mul_(beta1).add_(gradient_value, alpha=1.0 - beta1)
                second_moment.mul_(beta2).addcmul_(gradient_value, gradient_value, value=1.0 - beta2)
                corrected_first = first_moment / (1.0 - beta1**step)
                corrected_second = second_moment / (1.0 - beta2**step)
                vector.addcdiv_(corrected_first, corrected_second.sqrt().add(epsilon), value=-learning_rate)
            losses.append(float(loss.detach()))
        vectors.append(vector.detach())
        reports[str(label)] = {
            "initial_loss": losses[0],
            "final_loss": losses[-1],
            "steps": steps,
            "optimizer": "deterministic_cpu_adamw",
            "vector_norm": float(torch.linalg.vector_norm(vector.detach())),
        }
    return torch.stack(vectors), reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--gradient-manifest", type=Path, required=True)
    parser.add_argument("--gradient-editor-contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.output_report.exists():
        raise FileExistsError("refusing existing external-baseline fit outputs")

    protocol = json.loads(args.protocol.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    required = {
        "stage": "qwen3_8b_external_baseline_fit",
        "execution_allowed": True,
        "protocol_sha256": sha256_file(args.protocol),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "gradient_manifest_sha256": sha256_file(args.gradient_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"external baseline fit authorization mismatch: {field}")
    if protocol.get("training_partition") != "subspace_fit" or protocol.get("evaluation_partition") != "operator_dev":
        raise ValueError("external baseline split contract mismatch")

    capture = _load_capture(args.capture_dir, args.capture_manifest, [SITE])
    # The frozen gradient evidence is a three-site capture. Load and validate
    # its complete site scope, then use the preregistered layer-27 view.
    gradient = _load_gradient_capture(
        args.gradient_manifest, list(CAPTURE_SITES), args.gradient_editor_contract
    )
    _validate_row_alignment(capture, gradient)
    metadata = capture["metadata"]
    if any(row.get("partition") != "subspace_fit" for row in metadata):
        raise ValueError("fit capture contains non-training identities")
    folds = torch.tensor([_fold(row["item_id"]) for row in metadata], dtype=torch.long)
    fit = torch.nonzero(folds != 7, as_tuple=False).flatten()
    calibration = torch.nonzero(folds == 7, as_tuple=False).flatten()
    if fit.numel() + calibration.numel() != 6144 or calibration.numel() == 0:
        raise ValueError("invalid fit/calibration partition")
    task = torch.tensor([float(row["target_obedience"]) for row in metadata])
    history = torch.tensor([1.0 if row["history_condition"] == "obedience" else 0.0 for row in metadata])
    mismatch = history != task
    margins = torch.tensor([float(row["task_aligned_margin"]) for row in metadata])
    teacher_indices = _teacher_indices(metadata)
    current = capture["outputs"][SITE].float()
    teacher = current.index_select(0, teacher_indices)
    gradients = gradient["gradients"][SITE].float()

    desired_zero = teacher[fit[task[fit] == 0]].mean(0)
    desired_one = teacher[fit[task[fit] == 1]].mean(0)
    raw_direction = desired_one - desired_zero
    raw_scale = float(torch.linalg.vector_norm(raw_direction))
    caa_direction = _unit(raw_direction)
    candidates = []
    cap = float(protocol["fit"]["maximum_relative_correction"])
    for multiplier in protocol["fit"]["caa_strength_grid"]:
        strength = float(multiplier) * raw_scale
        sign = (2.0 * task[calibration] - 1.0).unsqueeze(1)
        correction = sign * strength * caa_direction
        gains = torch.sum(gradients[calibration] * correction, dim=1)
        mismatch_gain = float(gains[mismatch[calibration]].mean())
        matched_abs = float(gains[~mismatch[calibration]].abs().mean())
        candidates.append({
            "multiplier": float(multiplier), "strength": strength,
            "mismatch_predicted_gain": mismatch_gain,
            "matched_predicted_absolute_change": matched_abs,
            "selection_score": mismatch_gain - 0.25 * matched_abs,
        })
    selected = max(candidates, key=lambda row: (row["selection_score"], -row["strength"]))

    obedience_boundary = capture["boundary"][SITE][fit[history[fit] == 1]].mean(0)
    verification_boundary = capture["boundary"][SITE][fit[history[fit] == 0]].mean(0)
    cast_direction = _unit(obedience_boundary - verification_boundary)
    projected = capture["boundary"][SITE].float() @ cast_direction
    cast_threshold = 0.5 * (
        float(projected[fit[history[fit] == 1]].mean())
        + float(projected[fit[history[fit] == 0]].mean())
    )
    cast_prediction = projected[calibration] >= cast_threshold
    cast_accuracy = float((cast_prediction.float() == history[calibration]).float().mean())

    rank = int(protocol["fit"]["loreft_rank"])
    loreft_basis, loreft_weights, loreft_bias, loreft_report = _fit_loreft(
        current, teacher, task, fit, rank, float(protocol["fit"]["ridge_penalty"])
    )
    reps_vectors, reps_report = _fit_reps(
        gradients, margins, task, fit,
        beta=float(protocol["fit"]["reps_beta"]),
        steps=int(protocol["fit"]["reps_steps"]),
        learning_rate=float(protocol["fit"]["reps_learning_rate"]),
    )
    checkpoint = {
        "schema_version": 1,
        "stage": "qwen3_8b_external_baseline_fit",
        "method": "method-faithful-external-baseline-adapters",
        "site": SITE,
        "hidden_size": current.shape[1],
        "protocol_sha256": sha256_file(args.protocol),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "gradient_manifest_sha256": sha256_file(args.gradient_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "caa_direction": caa_direction,
        "caa_strength": float(selected["strength"]),
        "cast_condition_direction": cast_direction,
        "cast_condition_threshold": cast_threshold,
        "loreft_basis": loreft_basis,
        "loreft_weights": loreft_weights,
        "loreft_bias": loreft_bias,
        "reps_vectors": reps_vectors,
        "maximum_relative_correction": cap,
        "contains_base_model_weights": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not finite_checkpoint(checkpoint):
        raise FloatingPointError("external baseline checkpoint is non-finite")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_checkpoint)
    report = {
        "schema_version": 1,
        "stage": "qwen3_8b_external_baseline_fit",
        "fit_complete": True,
        "training_rows": 6144,
        "fit_rows": int(fit.numel()),
        "calibration_rows": int(calibration.numel()),
        "training_partition": "subspace_fit",
        "operator_dev_accessed": False,
        "site": SITE,
        "caa": {"raw_direction_norm": raw_scale, "candidates": candidates, "selected": selected},
        "cast": {"condition_threshold": cast_threshold, "calibration_accuracy": cast_accuracy},
        "loreft": {"rank": rank, "training_coordinate_mse": loreft_report},
        "reps": reps_report,
        "checkpoint_sha256": sha256_file(args.output_checkpoint),
        "checkpoint_contains_base_model_weights": False,
        "all_metrics_finite": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
