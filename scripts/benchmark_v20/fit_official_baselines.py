#!/usr/bin/env python3
"""Fit official-code-derived baseline modules using frozen subspace_fit evidence only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v3.fit_governance_editor import _fold, _load_capture, _teacher_indices
from scripts.benchmark_v4.fit_directional_governance import _load_gradient_capture, _validate_row_alignment
from scripts.benchmark_v19.external_baselines import _clip_relative
from scripts.benchmark_v20.official_adapters import finite_checkpoint
from scripts.benchmark_v20.official_source import (
    REQUIRED_SOURCE_FILES,
    load_loreft_class,
    load_preference_loss,
    load_reps_class,
    verify_required_sources,
)


SITE = "27:mlp"
CAPTURE_SITES = ("23:self_attn", "27:mlp", "31:mlp")
STAGE = "qwen3_8b_external_baseline_official_fit"


def _unit(value: torch.Tensor) -> torch.Tensor:
    value = value.float()
    norm = torch.linalg.vector_norm(value)
    if not bool(torch.isfinite(norm)) or float(norm) <= 1e-12:
        raise ValueError("cannot normalize a null or non-finite direction")
    return value / norm


def _manifest_source_hashes(manifest: dict) -> dict[str, str]:
    records = manifest.get("required_files")
    if not isinstance(records, dict) or set(records) != set(REQUIRED_SOURCE_FILES):
        raise ValueError("source manifest does not bind the complete required-file set")
    hashes = {}
    for key, relative in REQUIRED_SOURCE_FILES.items():
        record = records[key]
        if record.get("relative_path") != relative:
            raise ValueError(f"source manifest path mismatch: {key}")
        digest = record.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid source digest: {key}")
        hashes[key] = digest
    return hashes


def _pca_pairwise(positive: torch.Tensor, negative: torch.Tensor) -> tuple[torch.Tensor, float]:
    if positive.shape != negative.shape or positive.ndim != 2 or positive.shape[0] < 2:
        raise ValueError("invalid PCA-pairwise contrast tensors")
    import numpy as np
    pos = positive.detach().float().cpu().numpy()
    neg = negative.detach().float().cpu().numpy()
    h = np.empty((pos.shape[0] * 2, pos.shape[1]), dtype=np.float32)
    h[::2], h[1::2] = pos, neg
    center = (h[::2] + h[1::2]) / 2
    train = h.copy()
    train[::2] -= center
    train[1::2] -= center
    # Closed-form equivalent of the fixed upstream CAST
    # PCA(n_components=1, whiten=False) call.  The cluster runtime has no
    # scikit-learn installation, so use the same centered SVD directly.
    centered = train - train.mean(axis=0, keepdims=True)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    if singular_values.size == 0 or not np.isfinite(singular_values).all():
        raise ValueError("invalid CAST PCA singular values")
    total_variance = float(np.square(singular_values).sum())
    if total_variance <= 0.0:
        raise ValueError("null CAST PCA variance")
    direction = right[0].astype(np.float32, copy=True)
    projected = h @ direction
    positive_smaller = np.mean(projected[::2] < projected[1::2])
    positive_larger = np.mean(projected[::2] > projected[1::2])
    if positive_smaller > positive_larger:
        direction *= -1
    return _unit(torch.from_numpy(direction.copy())), float(singular_values[0] ** 2 / total_variance)


def _behavior_pairs(metadata: list[dict], teacher_indices: torch.Tensor, fit: torch.Tensor, teacher: torch.Tensor):
    lookup: dict[tuple, dict[int, int]] = {}
    for index in sorted(set(int(value) for value in teacher_indices.index_select(0, fit).tolist())):
        row = metadata[index]
        key = (
            row["item_id"], row["declared_role"], row["history_style"],
            int(row.get("history_depth", 0)), int(row["history_realization"]), int(row["label_swap"]),
        )
        label = int(float(row["target_obedience"]))
        if label in lookup.setdefault(key, {}) and lookup[key][label] != index:
            raise ValueError("ambiguous task-positive/task-negative CAST pairing")
        lookup[key][label] = index
    complete = [value for value in lookup.values() if set(value) == {0, 1}]
    if len(complete) < 2:
        raise ValueError("insufficient complete behavior pairs for CAST")
    positive = torch.stack([teacher[value[1]] for value in complete])
    negative = torch.stack([teacher[value[0]] for value in complete])
    return positive, negative


def _condition_pairs(metadata: list[dict], boundary: torch.Tensor, fit: torch.Tensor):
    lookup: dict[tuple, dict[int, torch.Tensor]] = {}
    for index in fit.tolist():
        row = metadata[index]
        key = (
            row["declared_role"], row["history_style"],
            int(row.get("history_depth", 0)), int(row["history_realization"]),
        )
        label = 1 if row["history_condition"] == "obedience" else 0
        value = boundary[index]
        existing = lookup.setdefault(key, {}).get(label)
        if existing is not None and not torch.equal(existing, value):
            raise ValueError("same CAST condition key maps to unequal boundary states")
        lookup[key][label] = value
    complete = [value for value in lookup.values() if set(value) == {0, 1}]
    if len(complete) < 2:
        raise ValueError("insufficient complete condition pairs for CAST")
    positive = torch.stack([value[1] for value in complete])
    negative = torch.stack([value[0] for value in complete])
    return positive, negative


def _cast_scores(boundary: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    direction = _unit(direction).to(boundary)
    coefficient = (boundary.float() @ direction) / torch.clamp(direction @ direction, min=1e-12)
    projected = coefficient.unsqueeze(-1) * direction
    transformed = torch.tanh(projected)
    numerator = torch.sum(boundary.float() * transformed, dim=-1)
    denominator = torch.linalg.vector_norm(boundary.float(), dim=-1) * torch.linalg.vector_norm(transformed, dim=-1)
    return numerator / torch.clamp(denominator, min=1e-12)


def _select_threshold(scores: torch.Tensor, labels: torch.Tensor) -> tuple[float, int, float]:
    values = torch.unique(scores.detach().float()).sort().values
    candidates = [float(values[0]) - 1e-7, float(values[-1]) + 1e-7]
    candidates.extend(float((values[i] + values[i + 1]) / 2) for i in range(len(values) - 1))
    best = None
    for threshold in candidates:
        high = scores >= threshold
        for polarity in (1, -1):
            predicted = high if polarity == 1 else ~high
            accuracy = float((predicted.float() == labels.float()).float().mean())
            row = (accuracy, polarity == 1, -abs(threshold), threshold, polarity)
            if best is None or row[:3] > best[:3]:
                best = row
    assert best is not None
    return float(best[3]), int(best[4]), float(best[0])


def _fit_loreft_module(
    cls, current: torch.Tensor, teacher: torch.Tensor, indices: torch.Tensor,
    *, rank: int, steps: int, batch_size: int, learning_rate: float, seed: int,
):
    torch.manual_seed(seed)
    module = cls(
        embed_dim=current.shape[1], low_rank_dimension=rank,
        dtype=torch.float32, dropout=0.0, act_fn=None,
    ).float()
    with torch.no_grad():
        module.learned_source.weight.copy_(module.rotate_layer.weight.T)
        module.learned_source.bias.zero_()
    optimizer = torch.optim.AdamW(
        module.parameters(), lr=learning_rate, weight_decay=1e-4,
        foreach=False, fused=False,
    )
    generator = torch.Generator().manual_seed(seed + 1000)
    losses = []
    module.train()
    order = torch.randperm(indices.numel(), generator=generator)
    cursor = 0
    for step in range(steps):
        if cursor + batch_size > indices.numel():
            order = torch.randperm(indices.numel(), generator=generator)
            cursor = 0
        selected = indices.index_select(0, order[cursor:cursor + batch_size])
        cursor += batch_size
        x = current.index_select(0, selected).float()
        y = teacher.index_select(0, selected).float()
        optimizer.zero_grad(set_to_none=True)
        proposed = module(x)
        correction = proposed - x
        loss = (proposed - y).square().sum(dim=-1).mean() + 1e-5 * correction.square().sum(dim=-1).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite official LoReFT loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(module.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite official LoReFT gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    module.eval()
    state = {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
    return state, {
        "initial_loss": losses[0], "final_loss": losses[-1], "steps": steps,
        "batch_size": batch_size, "optimizer": "torch_adamw_on_official_LoreftIntervention",
        "trainable_parameters": sum(parameter.numel() for parameter in module.parameters()),
    }


def _fit_reps_modules(
    cls, preference_loss, current: torch.Tensor, gradients: torch.Tensor, margins: torch.Tensor,
    task: torch.Tensor, fit: torch.Tensor, *, cap: float, beta: float, steps: int,
    batch_size: int, learning_rate: float, seed: int,
):
    torch.manual_seed(seed)
    # Upstream PreferenceVectorIntervention indexes one projection row but
    # broadcasts its full bias vector.  Its intended one-vector use is rank 1;
    # two task-specific rank-1 instances preserve the official forward path.
    modules = torch.nn.ModuleList([
        cls(
            embed_dim=current.shape[1], low_rank_dimension=1, dropout=0.0,
            intervention_positions_dropout=0.0,
        ).float()
        for _ in range(2)
    ])
    for module in modules:
        with torch.no_grad():
            # Preserve the official Linear weight initialization.  A literal
            # zero weight makes the official null-out branch evaluate 0/0.
            module.proj.bias.zero_()
        module.proj.bias.requires_grad_(False)
    trainable = [parameter for parameter in modules.parameters() if parameter.requires_grad]
    if sum(parameter.numel() for parameter in trainable) != 2 * current.shape[1]:
        raise ValueError("RePS adaptation must train exactly two task-specific projection rows")
    optimizer = torch.optim.AdamW(
        trainable, lr=learning_rate, weight_decay=1e-3,
        foreach=False, fused=False,
    )
    generator = torch.Generator().manual_seed(seed + 2000)
    order = torch.randperm(fit.numel(), generator=generator)
    cursor = 0
    losses = []
    modules.train()
    for step in range(steps):
        if cursor + batch_size > fit.numel():
            order = torch.randperm(fit.numel(), generator=generator)
            cursor = 0
        selected = fit.index_select(0, order[cursor:cursor + batch_size])
        cursor += batch_size
        x = current.index_select(0, selected).float().unsqueeze(1)
        g = gradients.index_select(0, selected).float()
        base_margin = margins.index_select(0, selected).float()
        labels = task.index_select(0, selected).long()
        optimizer.zero_grad(set_to_none=True)
        proposed = torch.empty_like(x)
        for task_id, module in enumerate(modules):
            mask = labels == task_id
            if bool(mask.any()):
                task_x = x[mask]
                subspaces = {
                    "subspaces": [0] * task_x.shape[0],
                    "steering_factor": torch.ones(task_x.shape[0], dtype=torch.float32),
                }
                proposed[mask] = module(task_x, subspaces=subspaces).output
        correction = _clip_relative(x, proposed - x, cap).squeeze(1)
        policy_margin = base_margin + torch.sum(g * correction, dim=-1)
        chosen = 0.5 * policy_margin
        rejected = -0.5 * policy_margin
        zeros = torch.zeros_like(chosen)
        ones = torch.ones_like(labels)
        per_row, _, _ = preference_loss(
            chosen, rejected, zeros, zeros, beta, 0.0, 1.0, ones, ones,
            label_smoothing=0.0, loss_type="dpo", reference_free=True,
        )
        loss = per_row.mean() + 1e-5 * correction.square().sum(dim=-1).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite official RePS preference loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite official RePS gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    modules.eval()
    states = [
        {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}
        for module in modules
    ]
    return states, {
        "initial_loss": losses[0], "final_loss": losses[-1], "steps": steps,
        "batch_size": batch_size,
        "optimizer": "torch_adamw_on_two_official_rank1_PreferenceVectorIntervention_modules",
        "loss": "official_reference_free_dpo_preference_loss",
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "bias_frozen": True,
        "module_count": 2,
        "rank_per_module": 1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--gradient-manifest", type=Path, required=True)
    parser.add_argument("--gradient-editor-contract", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    args = parser.parse_args()
    if args.output_checkpoint.exists() or args.output_report.exists():
        raise FileExistsError("refusing existing official-derived fit outputs")

    protocol = json.loads(args.protocol.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    source_manifest = json.loads(args.source_manifest.read_text())
    source_hashes = _manifest_source_hashes(source_manifest)
    verify_required_sources(args.official_source_root, source_hashes)
    required = {
        "stage": STAGE,
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
            raise ValueError(f"official baseline fit authorization mismatch: {field}")
    if protocol.get("training_partition") != "subspace_fit" or protocol.get("evaluation_partition") != "operator_dev":
        raise ValueError("official baseline split contract mismatch")

    capture = _load_capture(args.capture_dir, args.capture_manifest, [SITE])
    gradient = _load_gradient_capture(args.gradient_manifest, list(CAPTURE_SITES), args.gradient_editor_contract)
    _validate_row_alignment(capture, gradient)
    metadata = capture["metadata"]
    if any(row.get("partition") != "subspace_fit" for row in metadata):
        raise ValueError("fit capture contains non-training identities")
    folds = torch.tensor([_fold(row["item_id"]) for row in metadata], dtype=torch.long)
    fit = torch.nonzero(folds != 7, as_tuple=False).flatten()
    calibration = torch.nonzero(folds == 7, as_tuple=False).flatten()
    if fit.numel() + calibration.numel() != 6144 or calibration.numel() == 0:
        raise ValueError("invalid frozen fit/calibration partition")
    task = torch.tensor([float(row["target_obedience"]) for row in metadata])
    history = torch.tensor([1.0 if row["history_condition"] == "obedience" else 0.0 for row in metadata])
    mismatch = history != task
    margins = torch.tensor([float(row["task_aligned_margin"]) for row in metadata])
    teacher_indices = _teacher_indices(metadata)
    current = capture["outputs"][SITE].float()
    teacher = current.index_select(0, teacher_indices)
    gradients = gradient["gradients"][SITE].float()
    boundary = capture["boundary"][SITE].float()

    sign = (2.0 * task[fit[mismatch[fit]]] - 1.0).unsqueeze(1)
    paired_delta = teacher[fit[mismatch[fit]]] - current[fit[mismatch[fit]]]
    caa_raw = torch.mean(sign * paired_delta, dim=0)
    caa_direction = _unit(caa_raw)
    behavior_positive, behavior_negative = _behavior_pairs(metadata, teacher_indices, fit, teacher)
    cast_behavior_direction, cast_behavior_variance = _pca_pairwise(behavior_positive, behavior_negative)
    condition_positive, condition_negative = _condition_pairs(metadata, boundary, fit)
    cast_condition_direction, cast_condition_variance = _pca_pairwise(condition_positive, condition_negative)
    cast_scores = _cast_scores(boundary[calibration], cast_condition_direction)
    cast_threshold, cast_polarity, cast_accuracy = _select_threshold(cast_scores, history[calibration])

    cap = float(protocol["fit"]["maximum_relative_correction"])
    candidates = {}
    for name, direction, raw_scale in (
        ("caa", caa_direction, float(torch.linalg.vector_norm(caa_raw))),
        ("cast", cast_behavior_direction, float(torch.linalg.vector_norm((behavior_positive - behavior_negative).mean(0)))),
    ):
        rows = []
        for multiplier in protocol["fit"]["strength_grid"]:
            strength = float(multiplier) * raw_scale
            correction = (2.0 * task[calibration] - 1.0).unsqueeze(1) * strength * direction
            gains = torch.sum(gradients[calibration] * correction, dim=1)
            mismatch_gain = float(gains[mismatch[calibration]].mean())
            matched_abs = float(gains[~mismatch[calibration]].abs().mean())
            rows.append({
                "multiplier": float(multiplier), "strength": strength,
                "mismatch_predicted_gain": mismatch_gain,
                "matched_predicted_absolute_change": matched_abs,
                "selection_score": mismatch_gain - 0.25 * matched_abs,
            })
        candidates[name] = {"rows": rows, "selected": max(rows, key=lambda row: (row["selection_score"], -row["strength"]))}

    loreft_cls = load_loreft_class(args.official_source_root, source_hashes["loreft"])
    loreft_states, loreft_reports = [], {}
    for label in (0, 1):
        indices = fit[task[fit] == float(label)]
        state, report = _fit_loreft_module(
            loreft_cls, current, teacher, indices,
            rank=int(protocol["fit"]["loreft_rank"]),
            steps=int(protocol["fit"]["loreft_steps"]),
            batch_size=int(protocol["fit"]["loreft_batch_size"]),
            learning_rate=float(protocol["fit"]["loreft_learning_rate"]),
            seed=int(protocol["fit"]["seed"]) + label,
        )
        loreft_states.append(state)
        loreft_reports[str(label)] = report

    reps_cls = load_reps_class(args.official_source_root, source_hashes["reps_intervention"])
    preference_loss = load_preference_loss(args.official_source_root, source_hashes["reps_loss"])
    reps_states, reps_report = _fit_reps_modules(
        reps_cls, preference_loss, current, gradients, margins, task, fit,
        cap=cap, beta=float(protocol["fit"]["reps_beta"]),
        steps=int(protocol["fit"]["reps_steps"]),
        batch_size=int(protocol["fit"]["reps_batch_size"]),
        learning_rate=float(protocol["fit"]["reps_learning_rate"]),
        seed=int(protocol["fit"]["seed"]),
    )

    checkpoint = {
        "schema_version": 2,
        "stage": STAGE,
        "method": "official-code-derived-task-adaptations",
        "site": SITE,
        "hidden_size": current.shape[1],
        "protocol_sha256": sha256_file(args.protocol),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "gradient_manifest_sha256": sha256_file(args.gradient_manifest),
        "source_manifest_sha256": sha256_file(args.source_manifest),
        "official_source_files_sha256": source_hashes,
        "caa_direction": caa_direction,
        "caa_strength": float(candidates["caa"]["selected"]["strength"]),
        "cast_behavior_direction": cast_behavior_direction,
        "cast_strength": float(candidates["cast"]["selected"]["strength"]),
        "cast_condition_direction": cast_condition_direction,
        "cast_condition_threshold": cast_threshold,
        "cast_condition_polarity": cast_polarity,
        "loreft_rank": int(protocol["fit"]["loreft_rank"]),
        "loreft_state_dicts": loreft_states,
        "reps_state_dicts": reps_states,
        "maximum_relative_correction": cap,
        "official_code_derived_task_adaptation": True,
        "adapter_only": False,
        "unmodified_official_implementation": False,
        "contains_base_model_weights": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not finite_checkpoint(checkpoint):
        raise FloatingPointError("official-derived checkpoint is non-finite")
    args.output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_checkpoint)
    report = {
        "schema_version": 2, "stage": STAGE, "fit_complete": True,
        "training_rows": 6144, "fit_rows": int(fit.numel()), "calibration_rows": int(calibration.numel()),
        "training_partition": "subspace_fit", "operator_dev_accessed": False, "site": SITE,
        "official_code_derived_task_adaptation": True, "adapter_only": False,
        "unmodified_official_implementation": False,
        "caa": {"paired_mismatch_rows": int(mismatch[fit].sum()), **candidates["caa"]},
        "cast": {
            "behavior_pairs": int(behavior_positive.shape[0]), "condition_pairs": int(condition_positive.shape[0]),
            "behavior_explained_variance": cast_behavior_variance,
            "condition_explained_variance": cast_condition_variance,
            "condition_threshold": cast_threshold, "condition_polarity": cast_polarity,
            "calibration_accuracy": cast_accuracy, **candidates["cast"],
        },
        "loreft": loreft_reports, "reps": reps_report,
        "checkpoint_sha256": sha256_file(args.output_checkpoint),
        "checkpoint_contains_base_model_weights": False, "all_metrics_finite": True,
        "final_test_open": False, "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not all(math.isfinite(float(value)) for value in (
        cast_behavior_variance, cast_condition_variance, cast_accuracy,
        *[entry["final_loss"] for entry in loreft_reports.values()], reps_report["final_loss"],
    )):
        raise FloatingPointError("official-derived fit report contains non-finite metrics")
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
