#!/usr/bin/env python3
"""Fit the ADSGE-V4 application veto while keeping the selected V3 editor frozen."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v3.fit_governance_editor import _matched_history
from scripts.benchmark_v4.directional_governance import editor_from_checkpoint as v3_from_checkpoint
from scripts.benchmark_v5.abstaining_directional_governance import (
    APPLICATION_FEATURE_WIDTH,
    ApplicationVetoHead,
    application_features,
)
from scripts.benchmark_v5.abstaining_split import (
    canonical_protected_family,
    exact_source_coverage,
    small_control_fold_overrides,
)


SITE_KEY = "27:mlp"
LAYER_KEY = "27"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fold(group_id: str) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 8


def _protected_fold(row: dict, contract: dict, overrides: dict[tuple[str, str], int]) -> int:
    family = canonical_protected_family(str(row["control_family"]), contract)
    control_id = row.get("control_id")
    if control_id is not None:
        key = (family, str(control_id))
        if key not in overrides:
            raise ValueError(f"missing V4 small-control fold override: {key}")
        return overrides[key]
    return _fold(_group_id(row))


def _group_id(row: dict) -> str:
    for name in ("item_id", "case_id", "control_id", "job_key", "case_key"):
        value = row.get(name)
        if value is not None and str(value):
            return str(value)
    raise ValueError("capture row has no stable grouping identity")


def _load_states(manifest_path: Path, *, expected_rows: int) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("rows") != expected_rows:
        raise ValueError(f"capture row mismatch: {manifest_path}")
    metadata: list[dict] = []
    boundary: list[torch.Tensor] = []
    context: list[torch.Tensor] = []
    observed: set[str] = set()
    for record in manifest.get("shards", []):
        path = manifest_path.parent / str(record["file"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"capture shard SHA mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        rows = list(shard["metadata"])
        if len(rows) != int(record["rows"]):
            raise ValueError(f"capture shard row count mismatch: {path}")
        for row in rows:
            identity = str(row.get("job_key") or row.get("case_key") or _canonical(row))
            if identity in observed:
                raise ValueError(f"duplicate capture identity: {identity}")
            observed.add(identity)
        current_boundary = shard["boundary_states"][LAYER_KEY].float()
        if current_boundary.ndim == 1:
            current_boundary = current_boundary.unsqueeze(0).expand(len(rows), -1).clone()
        elif current_boundary.ndim == 2 and current_boundary.shape[0] == 1:
            current_boundary = current_boundary.expand(len(rows), -1).clone()
        if current_boundary.ndim != 2 or current_boundary.shape[0] != len(rows):
            raise ValueError(f"boundary-state shape mismatch: {path}")
        current_context = shard["component_inputs"][SITE_KEY].float()
        if current_context.ndim != 2 or current_context.shape[0] != len(rows):
            raise ValueError(f"context-state shape mismatch: {path}")
        for name, value in (("boundary", current_boundary), ("context", current_context)):
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError(f"non-finite {name} capture: {path}")
        metadata.extend(rows)
        boundary.append(current_boundary)
        context.append(current_context)
    if len(metadata) != expected_rows or len(observed) != expected_rows:
        raise ValueError("capture is incomplete or contains duplicate rows")
    return {
        "manifest": manifest,
        "metadata": metadata,
        "boundary": torch.cat(boundary, dim=0),
        "context": torch.cat(context, dim=0),
    }


def _extract_features(v3, capture: dict, *, batch_size: int = 512) -> torch.Tensor:
    values = []
    v3.eval()
    with torch.no_grad():
        for offset in range(0, len(capture["metadata"]), batch_size):
            current = application_features(
                v3,
                capture["boundary"][offset : offset + batch_size],
                capture["context"][offset : offset + batch_size],
            )[0]
            values.append(current.detach().cpu())
    result = torch.cat(values, dim=0)
    if result.shape != (len(capture["metadata"]), APPLICATION_FEATURE_WIDTH):
        raise ValueError("V4 applicability feature matrix has the wrong shape")
    return result


@dataclass(frozen=True)
class Dataset:
    features: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor
    folds: torch.Tensor
    sources: tuple[str, ...]
    directions: tuple[str, ...]
    label_swaps: tuple[int, ...]


def _dataset(governance: dict, protected: dict, v3, contract: dict) -> Dataset:
    governance_features = _extract_features(v3, governance)
    protected_features = _extract_features(v3, protected)
    targets = []
    weights = []
    folds = []
    sources: list[str] = []
    directions: list[str] = []
    swaps: list[int] = []
    control_fold_overrides = small_control_fold_overrides(protected["metadata"], contract)
    governance_negative_weight = float(
        contract["fit_labels"]["governance_negative_weight_multiplier"]
    )
    for row in governance["metadata"]:
        mismatch = str(row["history_condition"]) != _matched_history(row)
        targets.append(1.0 if mismatch else 0.0)
        weights.append(1.0 if mismatch else governance_negative_weight)
        folds.append(_fold(_group_id(row)))
        if mismatch:
            sources.append("governance_mismatch")
        else:
            matched_family = (
                "matched_delegated_choice"
                if float(row["target_obedience"]) == 1.0
                else "matched_verification"
            )
            sources.append(f"governance_matched:{matched_family}")
        directions.append(
            "positive" if mismatch and str(row["history_condition"]) == "obedience" else
            "negative" if mismatch else "none"
        )
        swaps.append(int(row["label_swap"]))
    fit_labels = contract["fit_labels"]
    hard_weight = float(fit_labels["fresh_and_explicit_reset_weight_multiplier"])
    protected_weight = float(fit_labels["other_protected_weight_multiplier"])
    for row in protected["metadata"]:
        family = canonical_protected_family(str(row["control_family"]), contract)
        targets.append(0.0)
        weights.append(
            hard_weight
            if family in {"fresh_verification", "explicit_governance_reset"}
            else protected_weight
        )
        folds.append(_protected_fold(row, contract, control_fold_overrides))
        sources.append(f"protected:{family}")
        directions.append("none")
        swaps.append(int(row.get("label_swap", -1)))
    features = torch.cat((governance_features, protected_features), dim=0)
    target = torch.tensor(targets, dtype=torch.float32)
    sample_weights = torch.tensor(weights, dtype=torch.float32)
    fold_tensor = torch.tensor(folds, dtype=torch.long)
    if not bool(torch.isfinite(features).all()) or not bool(torch.isfinite(sample_weights).all()):
        raise FloatingPointError("non-finite V4 fit dataset")
    return Dataset(
        features=features,
        target=target,
        weights=sample_weights,
        folds=fold_tensor,
        sources=tuple(sources),
        directions=tuple(directions),
        label_swaps=tuple(swaps),
    )


def _metrics(logits: torch.Tensor, indices: torch.Tensor, data: Dataset, threshold: float) -> dict:
    active = logits[indices] > threshold
    target = data.target[indices].bool()
    result: dict[str, object] = {
        "rows": len(indices),
        "positive_rows": int(target.sum()),
        "negative_rows": int((~target).sum()),
        "negative_active_fraction": float(active[~target].float().mean()) if bool((~target).any()) else 0.0,
        "mismatch_true_positive_rate": float(active[target].float().mean()) if bool(target.any()) else 0.0,
    }
    directions = {}
    swaps = {}
    source_reports = {}
    for name in ("positive", "negative"):
        mask = torch.tensor(
            [data.directions[int(index)] == name for index in indices], dtype=torch.bool
        )
        directions[name] = {
            "rows": int(mask.sum()),
            "true_positive_rate": float(active[mask].float().mean()) if bool(mask.any()) else 0.0,
        }
    for value in (0, 1):
        mask = target & torch.tensor(
            [data.label_swaps[int(index)] == value for index in indices], dtype=torch.bool
        )
        swaps[str(value)] = {
            "rows": int(mask.sum()),
            "true_positive_rate": float(active[mask].float().mean()) if bool(mask.any()) else 0.0,
        }
    source_names = sorted({data.sources[int(index)] for index in indices})
    for source in source_names:
        mask = torch.tensor(
            [data.sources[int(index)] == source for index in indices], dtype=torch.bool
        )
        current_target = target[mask]
        current_active = active[mask]
        source_reports[source] = {
            "rows": int(mask.sum()),
            "active_fraction": float(current_active.float().mean()) if bool(mask.any()) else 0.0,
            "positive_rows": int(current_target.sum()),
        }
    result["by_direction"] = directions
    result["by_label_swap"] = swaps
    result["by_source"] = source_reports
    return result


def _normalization(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    center = features.float().mean(dim=0)
    scale = torch.sqrt(torch.mean((features.float() - center).square(), dim=0))
    scale = torch.clamp(scale, min=1e-4)
    return center, scale


def _train_candidate(
    *,
    name: str,
    hidden_widths: list[int],
    seed: int,
    data: Dataset,
    train_indices: torch.Tensor,
    calibration_indices: torch.Tensor,
    config: dict,
    gates: dict,
) -> tuple[ApplicationVetoHead, dict]:
    random.seed(seed)
    torch.manual_seed(seed)
    center, scale = _normalization(data.features[train_indices])
    head = ApplicationVetoHead(
        feature_center=center,
        feature_scale=scale,
        hidden_widths=hidden_widths,
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
        fused=False,
    )
    train_target = data.target[train_indices]
    positives = float(train_target.sum())
    negatives = float(len(train_target) - positives)
    if positives <= 0 or negatives <= 0:
        raise ValueError("V4 train fold must contain both application classes")
    positive_weight = negatives / positives
    generator = torch.Generator().manual_seed(seed)
    best_loss = math.inf
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())
    stale = 0
    trace = []
    batch_size = int(config["batch_size"])
    for epoch in range(1, int(config["maximum_epochs"]) + 1):
        permutation = train_indices[
            torch.randperm(len(train_indices), generator=generator)
        ]
        losses = []
        head.train()
        for offset in range(0, len(permutation), batch_size):
            indices = permutation[offset : offset + batch_size]
            logits = head.logits(data.features[indices])
            row_loss = F.binary_cross_entropy_with_logits(
                logits, data.target[indices], reduction="none"
            )
            class_weight = torch.where(
                data.target[indices] > 0.5,
                torch.full_like(row_loss, positive_weight),
                torch.ones_like(row_loss),
            )
            loss = torch.mean(row_loss * class_weight * data.weights[indices])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite V4 application loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                list(head.parameters()), float(config["gradient_norm_clip"])
            )
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite V4 application gradient")
            optimizer.step()
            losses.append(float(loss.detach()))
        head.eval()
        with torch.no_grad():
            calibration_logits = head.logits(data.features[calibration_indices])
            calibration_loss = F.binary_cross_entropy_with_logits(
                calibration_logits, data.target[calibration_indices]
            )
        current = float(calibration_loss)
        if current < best_loss - 1e-7:
            best_loss = current
            best_epoch = epoch
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
        if epoch in {1, int(config["maximum_epochs"])} or epoch % 10 == 0:
            trace.append(
                {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses),
                    "calibration_loss": current,
                }
            )
        if stale >= int(config["early_stopping_patience"]):
            break
    head.load_state_dict(best_state, strict=True)
    head.eval()
    with torch.no_grad():
        logits = head.logits(data.features)
    calibration_negative = calibration_indices[data.target[calibration_indices] == 0]
    if len(calibration_negative) == 0:
        raise ValueError("V4 calibration fold has no negative rows")
    maximum_negative = logits[calibration_negative].max().float()
    threshold_tensor = torch.nextafter(
        maximum_negative, torch.tensor(float("inf"), dtype=torch.float32)
    )
    threshold = float(threshold_tensor)
    if not math.isfinite(threshold):
        raise FloatingPointError("non-finite calibrated V4 threshold")
    head.threshold_logit.copy_(threshold_tensor)
    metrics = _metrics(logits, calibration_indices, data, threshold)
    direction_minimum = min(
        float(value["true_positive_rate"])
        for value in metrics["by_direction"].values()
    )
    swap_minimum = min(
        float(value["true_positive_rate"])
        for value in metrics["by_label_swap"].values()
    )
    eligible = (
        float(metrics["negative_active_fraction"])
        <= float(gates["calibration_negative_active_fraction_maximum"])
        and float(metrics["mismatch_true_positive_rate"])
        >= float(gates["calibration_mismatch_true_positive_rate_minimum"])
        and direction_minimum
        >= float(gates["calibration_each_direction_true_positive_rate_minimum"])
        and swap_minimum
        >= float(gates["calibration_each_label_swap_true_positive_rate_minimum"])
    )
    report = {
        "architecture": name,
        "hidden_widths": hidden_widths,
        "seed": seed,
        "trainable_parameter_count": sum(p.numel() for p in head.parameters()),
        "best_epoch": best_epoch,
        "best_calibration_loss": best_loss,
        "threshold_logit": threshold,
        "maximum_calibration_negative_logit": float(maximum_negative),
        "positive_class_weight": positive_weight,
        "calibration": metrics,
        "calibration_min_direction_tpr": direction_minimum,
        "calibration_min_label_swap_tpr": swap_minimum,
        "calibration_eligible": eligible,
        "trace": trace,
    }
    return head, report


def _validate_inputs(args, contract: dict) -> tuple[dict, dict]:
    if contract.get("method_short_name") != "ADSGE-V4":
        raise ValueError("unexpected V4 contract")
    required_contract = {
        "status": "design_corrected_after_terminal_code_v32_prefit_workdir_test_failure_before_any_v4_fit",
        "code_version_minimum": 33,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_contract.items():
        if contract.get(field) != expected:
            raise ValueError(f"V4 contract mismatch: {field}")
    bindings = contract["bound_v3_evidence"]
    hashes = {
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "selected_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "fit_report_sha256": sha256_file(args.v3_fit_report),
    }
    for field, observed in hashes.items():
        if bindings.get(field) != observed:
            raise ValueError(f"V4 V3 binding mismatch: {field}")
    fit_inputs = contract["bound_fit_inputs"]
    for field, path in (
        ("governance_capture_manifest_sha256", args.capture_manifest),
        ("protected_capture_manifest_sha256", args.protected_capture_manifest),
    ):
        if fit_inputs.get(field) != sha256_file(path):
            raise ValueError(f"V4 capture binding mismatch: {field}")
    fit_report = json.loads(args.v3_fit_report.read_text())
    candidate = next(
        (
            row
            for row in fit_report.get("candidate_reports", [])
            if row.get("candidate_id") == "DSGE-V3-27:mlp"
        ),
        None,
    )
    if candidate is None or candidate.get("fit_eligible") is not True:
        raise ValueError("selected V3 fit candidate is unavailable or ineligible")
    if candidate.get("checkpoint_sha256") != sha256_file(args.v3_checkpoint):
        raise ValueError("selected V3 fit candidate checkpoint mismatch")
    authorization = json.loads(args.execution_authorization.read_text())
    code_root = str(authorization.get("code_root", ""))
    required_authorization = {
        "stage": "governance_abstaining_router_fit",
        "execution_allowed": True,
        "code_root": "/workspace/context-mismatch-qwen3-8b/code-v33",
        "immutable_code_bundle_manifest_sha256": sha256_file(Path(code_root) / "bundle.sha256"),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_manifest),
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_authorization.items():
        if authorization.get(field) != expected:
            raise ValueError(f"V4 fit authorization mismatch: {field}")
    return fit_report, authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing V4 fit output: {args.output_dir}")
    contract = json.loads(args.editor_contract.read_text())
    _, authorization = _validate_inputs(args, contract)
    v3_checkpoint = torch.load(args.v3_checkpoint, map_location="cpu", weights_only=False)
    v3 = v3_from_checkpoint(v3_checkpoint)
    if v3.site.key != SITE_KEY:
        raise ValueError("V4 fit requires selected V3 site 27:mlp")
    for parameter in v3.parameters():
        parameter.requires_grad_(False)
    v3_before = {name: value.detach().clone() for name, value in v3.state_dict().items()}
    v3_sites = v3_checkpoint.get("sites", [])
    v3_state_names = (
        set(v3_sites[0].get("state_dict", {}))
        if isinstance(v3_sites, list) and len(v3_sites) == 1
        else set()
    )
    checkpoint_single_site = len(v3_sites) == 1 and v3.site.key == SITE_KEY
    checkpoint_excludes_base_weights = (
        v3_checkpoint.get("base_model_weights_included") is False
        and not any(
            str(name).startswith(("model.", "base_model."))
            for name in v3_state_names
        )
    )

    governance = _load_states(args.capture_manifest, expected_rows=6144)
    protected = _load_states(args.protected_capture_manifest, expected_rows=4008)
    if governance["manifest"].get("partition") != "subspace_fit":
        raise ValueError("V4 governance capture is not subspace_fit")
    if protected["manifest"].get("partition") != "subspace_fit":
        raise ValueError("V4 protected capture is not subspace_fit")
    expected_raw_family_rows = {
        str(name): int(count)
        for name, count in contract["bound_fit_inputs"][
            "protected_capture_raw_family_rows"
        ].items()
    }
    observed_raw_family_rows = {
        str(name): int(count)
        for name, count in protected["manifest"].get("family_rows", {}).items()
    }
    if observed_raw_family_rows != expected_raw_family_rows:
        raise ValueError("V4 protected capture family-row contract mismatch")
    data = _dataset(governance, protected, v3, contract)
    train_indices = torch.where(data.folds <= 5)[0]
    calibration_indices = torch.where(data.folds == 6)[0]
    audit_indices = torch.where(data.folds == 7)[0]
    if min(len(train_indices), len(calibration_indices), len(audit_indices)) <= 0:
        raise ValueError("V4 split contains an empty fold")

    optimization = contract["optimization"]
    gates = contract["fit_gates"]
    trained: list[tuple[ApplicationVetoHead, dict]] = []
    for architecture in contract["application_veto"]["candidate_architectures"]:
        for seed in contract["application_veto"]["fixed_seeds"]:
            trained.append(
                _train_candidate(
                    name=str(architecture["name"]),
                    hidden_widths=[int(value) for value in architecture["hidden_widths"]],
                    seed=int(seed),
                    data=data,
                    train_indices=train_indices,
                    calibration_indices=calibration_indices,
                    config=optimization,
                    gates=gates,
                )
            )
    trained.sort(
        key=lambda pair: (
            not bool(pair[1]["calibration_eligible"]),
            -float(pair[1]["calibration_min_direction_tpr"]),
            -float(pair[1]["calibration_min_label_swap_tpr"]),
            -float(pair[1]["calibration"]["mismatch_true_positive_rate"]),
            int(pair[1]["trainable_parameter_count"]),
            str(pair[1]["architecture"]),
            int(pair[1]["seed"]),
        )
    )
    selected_head, selected_report = trained[0]
    with torch.no_grad():
        logits = selected_head.logits(data.features)
    audit = _metrics(
        logits,
        audit_indices,
        data,
        float(selected_head.threshold_logit),
    )
    audit_direction_minimum = min(
        float(value["true_positive_rate"]) for value in audit["by_direction"].values()
    )
    audit_swap_minimum = min(
        float(value["true_positive_rate"]) for value in audit["by_label_swap"].values()
    )
    protected_audit_sources = {
        name: value
        for name, value in audit["by_source"].items()
        if name.startswith("protected:")
    }
    governance_matched_audit_sources = {
        name: value
        for name, value in audit["by_source"].items()
        if name.startswith("governance_matched:")
    }
    expected_protected_sources = {
        f"protected:{name}"
        for name in gates["audit_expected_protected_capture_families"]
    }
    expected_governance_matched_sources = {
        f"governance_matched:{name}"
        for name in gates["audit_expected_governance_matched_families"]
    }
    calibration_negative_sources = {
        name
        for name, value in selected_report["calibration"]["by_source"].items()
        if int(value["positive_rows"]) == 0
    }
    expected_calibration_negative_sources = (
        expected_protected_sources | expected_governance_matched_sources
    )
    max_v3_change = 0.0
    for name, before in v3_before.items():
        after = v3.state_dict()[name]
        max_v3_change = max(max_v3_change, float(torch.max(torch.abs(after - before))))
    gate_checks = {
        "calibration_eligible": bool(selected_report["calibration_eligible"]),
        "calibration_negative_source_coverage_exact": exact_source_coverage(
            calibration_negative_sources, expected_calibration_negative_sources
        ),
        "audit_governance_matched_source_coverage_exact": exact_source_coverage(
            set(governance_matched_audit_sources), expected_governance_matched_sources
        ),
        "audit_each_governance_matched_family_exact_abstention": all(
            int(value["rows"]) > 0
            and float(value["active_fraction"])
            <= float(gates["audit_governance_matched_active_fraction_maximum"])
            for value in governance_matched_audit_sources.values()
        ),
        "audit_protected_source_coverage_exact": exact_source_coverage(
            set(protected_audit_sources), expected_protected_sources
        ),
        "audit_each_protected_family_exact_abstention": all(
            int(value["rows"]) > 0 and float(value["active_fraction"]) == 0.0
            for value in protected_audit_sources.values()
        ),
        "audit_mismatch_tpr": float(audit["mismatch_true_positive_rate"])
        >= float(gates["audit_mismatch_true_positive_rate_minimum"]),
        "audit_each_direction_tpr": audit_direction_minimum
        >= float(gates["audit_each_direction_true_positive_rate_minimum"]),
        "audit_each_label_swap_tpr": audit_swap_minimum
        >= float(gates["audit_each_label_swap_true_positive_rate_minimum"]),
        "v3_frozen_exact": max_v3_change
        <= float(gates["v3_frozen_parameter_max_absolute_change"]),
        "checkpoint_contains_exactly_one_site": checkpoint_single_site
        and gates["checkpoint_contains_exactly_one_site"] is True,
        "checkpoint_contains_no_base_weights": checkpoint_excludes_base_weights
        and gates["checkpoint_contains_no_base_weights"] is True,
    }
    fit_eligible = all(gate_checks.values())
    args.output_dir.mkdir(parents=True)
    checkpoint = {
        "schema_version": 1,
        "kind": "abstaining_directional_governance_editor",
        "method": "ADSGE-V4",
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_manifest),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "v3_checkpoint": v3_checkpoint,
        "v3_experts_frozen": True,
        "v3_max_absolute_change": max_v3_change,
        "application_head": {
            "architecture": selected_report["architecture"],
            "hidden_widths": list(selected_head.hidden_widths),
            "seed": selected_report["seed"],
            "feature_center": selected_head.feature_center.detach().cpu(),
            "feature_scale": selected_head.feature_scale.detach().cpu(),
            "threshold_logit": float(selected_head.threshold_logit),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in selected_head.state_dict().items()
            },
        },
        "application_trainable_parameter_count": sum(
            parameter.numel() for parameter in selected_head.parameters()
        ),
        "base_model_weights_included": False,
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    checkpoint_path = args.output_dir / "abstaining_directional_editor__layer_27__mlp.pt"
    incoming = args.output_dir / f".{checkpoint_path.name}.incoming"
    torch.save(checkpoint, incoming)
    os.replace(incoming, checkpoint_path)
    report = {
        "schema_version": 1,
        "stage": "governance_abstaining_router_fit",
        "method": "ADSGE-V4",
        "fit_complete": True,
        "fit_eligible": fit_eligible,
        "selected_candidate": selected_report,
        "candidate_reports": [value for _, value in trained],
        "audit": audit,
        "audit_min_direction_tpr": audit_direction_minimum,
        "audit_min_label_swap_tpr": audit_swap_minimum,
        "gate_checks": gate_checks,
        "v3_max_absolute_change": max_v3_change,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_capture_manifest),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "rows": {
            "total": len(data.target),
            "train": len(train_indices),
            "calibration": len(calibration_indices),
            "audit": len(audit_indices),
            "positive": int(data.target.sum()),
            "negative": int((data.target == 0).sum()),
        },
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "abstaining_router_fit_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(_canonical(report))


if __name__ == "__main__":
    main()
