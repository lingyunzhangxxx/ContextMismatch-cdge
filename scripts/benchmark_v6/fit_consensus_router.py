#!/usr/bin/env python3
"""Fit the frozen V5 two-head consensus router and audit it without retuning."""

from __future__ import annotations

import argparse
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
from scripts.benchmark_v4.directional_governance import (
    editor_from_checkpoint as v3_from_checkpoint,
)
from scripts.benchmark_v5.abstaining_directional_governance import (
    APPLICATION_FEATURE_WIDTH,
    ApplicationVetoHead,
    application_features,
)
from scripts.benchmark_v6.consensus_directional_governance import (
    editor_from_checkpoint as v5_from_checkpoint,
)


SITE_KEY = "27:mlp"
LAYER_KEY = "27"
FAMILY_CROSSWALK = {
    "fresh": "fresh_verification",
    "verification": "matched_verification",
    "obedience_reset": "explicit_governance_reset",
    "supported_user_authority": "supported_user_authority",
    "factual_boundary_memory": "factual_boundary_memory",
}


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _fold(group_id: str) -> int:
    digest = hashlib.sha256(group_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % 8


def _group_id(row: dict) -> str:
    for field in ("item_id", "control_id", "case_id", "job_key", "case_key"):
        value = row.get(field)
        if value is not None and str(value):
            return str(value)
    raise ValueError("capture row lacks a stable group identity")


def _stable_row_identity(row: dict, *, protected: bool) -> str:
    if not protected:
        value = row.get("job_key")
        if not value:
            raise ValueError("governance row lacks job_key")
        return str(value)
    fields = (
        "control_family",
        "benchmark",
        "item_id",
        "control_id",
        "declared_role",
        "history_style",
        "history_realization",
        "label_swap",
    )
    return _canonical({field: row.get(field) for field in fields})


def _small_control_overrides(metadata: list[dict]) -> dict[tuple[str, str], int]:
    sequence = [0, 1, 2, 3, 6, 7]
    identities: dict[str, set[str]] = {
        "supported_user_authority": set(),
        "factual_boundary_memory": set(),
    }
    for row in metadata:
        control_id = row.get("control_id")
        if control_id is None:
            continue
        family = FAMILY_CROSSWALK[str(row["control_family"])]
        if family not in identities:
            raise ValueError(f"unexpected control-id family: {family}")
        identities[family].add(str(control_id))
    overrides: dict[tuple[str, str], int] = {}
    for family, observed in identities.items():
        if len(observed) != 6:
            raise ValueError(f"expected six identities in {family}, observed {len(observed)}")
        ordered = sorted(
            observed, key=lambda value: hashlib.sha256(value.encode("utf-8")).digest()
        )
        for identity, fold in zip(ordered, sequence, strict=True):
            overrides[(family, identity)] = fold
    return overrides


def _protected_fold(row: dict, overrides: dict[tuple[str, str], int]) -> int:
    family = FAMILY_CROSSWALK[str(row["control_family"])]
    control_id = row.get("control_id")
    if control_id is not None:
        return overrides[(family, str(control_id))]
    return _fold(_group_id(row))


def _load_capture(
    manifest_path: Path,
    *,
    expected_rows: int,
    protected: bool,
    allowed_folds: set[int] | None,
) -> tuple[dict, dict, int]:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("rows") != expected_rows:
        raise ValueError(f"capture row mismatch: {manifest_path}")
    all_metadata: list[dict] = []
    records = []
    for record in manifest.get("shards", []):
        path = manifest_path.parent / str(record["file"])
        if sha256_file(path) != record["sha256"]:
            raise ValueError(f"capture shard SHA mismatch: {path}")
        shard = torch.load(path, map_location="cpu", weights_only=False)
        rows = list(shard["metadata"])
        if len(rows) != int(record["rows"]):
            raise ValueError(f"capture shard row mismatch: {path}")
        all_metadata.extend(rows)
        records.append((path, shard, rows))
    if len(all_metadata) != expected_rows:
        raise ValueError("capture metadata is incomplete")
    overrides = _small_control_overrides(all_metadata) if protected else {}

    metadata: list[dict] = []
    boundary: list[torch.Tensor] = []
    context: list[torch.Tensor] = []
    folds: list[int] = []
    observed: set[str] = set()
    excluded = 0
    for path, shard, rows in records:
        current_boundary = shard["boundary_states"][LAYER_KEY].float()
        if current_boundary.ndim == 1:
            current_boundary = current_boundary.unsqueeze(0).expand(len(rows), -1)
        elif current_boundary.ndim == 2 and current_boundary.shape[0] == 1:
            current_boundary = current_boundary.expand(len(rows), -1)
        current_context = shard["component_inputs"][SITE_KEY].float()
        if current_boundary.ndim != 2 or current_boundary.shape[0] != len(rows):
            raise ValueError(f"boundary shape mismatch: {path}")
        if current_context.ndim != 2 or current_context.shape[0] != len(rows):
            raise ValueError(f"context shape mismatch: {path}")
        keep = []
        for index, row in enumerate(rows):
            identity = _stable_row_identity(row, protected=protected)
            if identity in observed:
                raise ValueError(f"duplicate capture identity: {identity}")
            observed.add(identity)
            fold = _protected_fold(row, overrides) if protected else _fold(_group_id(row))
            if allowed_folds is not None and fold not in allowed_folds:
                excluded += 1
                continue
            keep.append(index)
            metadata.append(row)
            folds.append(fold)
        if keep:
            indices = torch.tensor(keep, dtype=torch.long)
            selected_boundary = current_boundary[indices].clone()
            selected_context = current_context[indices].clone()
            if not bool(torch.isfinite(selected_boundary).all()):
                raise FloatingPointError(f"non-finite boundary state: {path}")
            if not bool(torch.isfinite(selected_context).all()):
                raise FloatingPointError(f"non-finite context state: {path}")
            boundary.append(selected_boundary)
            context.append(selected_context)
    if len(observed) != expected_rows or len(metadata) + excluded != expected_rows:
        raise ValueError("capture identity filtering is incomplete")
    return (
        {
            "manifest": manifest,
            "metadata": metadata,
            "boundary": torch.cat(boundary, dim=0),
            "context": torch.cat(context, dim=0),
            "folds": folds,
        },
        manifest,
        excluded,
    )


def _extract_features(v3, capture: dict, batch_size: int = 512) -> torch.Tensor:
    values = []
    v3.eval()
    with torch.no_grad():
        for offset in range(0, len(capture["metadata"]), batch_size):
            features = application_features(
                v3,
                capture["boundary"][offset : offset + batch_size],
                capture["context"][offset : offset + batch_size],
            )[0]
            values.append(features.detach().cpu())
    result = torch.cat(values, dim=0)
    if result.shape != (len(capture["metadata"]), APPLICATION_FEATURE_WIDTH):
        raise ValueError("V5 application feature matrix has the wrong shape")
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("V5 feature matrix contains non-finite values")
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


def _dataset(governance: dict, protected: dict, v3) -> Dataset:
    features = torch.cat(
        (_extract_features(v3, governance), _extract_features(v3, protected)), dim=0
    )
    targets = []
    weights = []
    folds = []
    sources: list[str] = []
    directions: list[str] = []
    swaps: list[int] = []
    for row, fold in zip(governance["metadata"], governance["folds"], strict=True):
        mismatch = str(row["history_condition"]) != _matched_history(row)
        targets.append(1.0 if mismatch else 0.0)
        weights.append(1.0)
        folds.append(fold)
        if mismatch:
            sources.append("governance_mismatch")
            directions.append(
                "positive" if str(row["history_condition"]) == "obedience" else "negative"
            )
        else:
            family = (
                "matched_delegated_choice"
                if float(row["target_obedience"]) == 1.0
                else "matched_verification"
            )
            sources.append(f"governance_matched:{family}")
            directions.append("none")
        swaps.append(int(row["label_swap"]))
    protected_weights = {
        "fresh_verification": 4.0,
        "explicit_governance_reset": 4.0,
        "matched_verification": 2.0,
        "supported_user_authority": 2.0,
        "factual_boundary_memory": 2.0,
    }
    for row, fold in zip(protected["metadata"], protected["folds"], strict=True):
        family = FAMILY_CROSSWALK[str(row["control_family"])]
        targets.append(0.0)
        weights.append(protected_weights[family])
        folds.append(fold)
        sources.append(f"protected:{family}")
        directions.append("none")
        swaps.append(int(row.get("label_swap", -1)))
    return Dataset(
        features=features,
        target=torch.tensor(targets, dtype=torch.float32),
        weights=torch.tensor(weights, dtype=torch.float32),
        folds=torch.tensor(folds, dtype=torch.long),
        sources=tuple(sources),
        directions=tuple(directions),
        label_swaps=tuple(swaps),
    )


def _median_mad(values: torch.Tensor, *, dimension: int | None = None):
    if dimension is None:
        center = torch.median(values.float())
        scale = 1.4826 * torch.median(torch.abs(values.float() - center))
    else:
        center = torch.median(values.float(), dim=dimension).values
        scale = 1.4826 * torch.median(
            torch.abs(values.float() - center.unsqueeze(dimension)), dim=dimension
        ).values
    return center, torch.clamp(scale, min=1e-4)


def _head_subset(data: Dataset, name: str) -> torch.Tensor:
    if name == "all_negative":
        return torch.ones(len(data.target), dtype=torch.bool)
    if name == "matched_state":
        return torch.tensor(
            [
                bool(data.target[index] > 0.5)
                or data.sources[index].startswith("governance_matched:")
                for index in range(len(data.target))
            ],
            dtype=torch.bool,
        )
    raise ValueError(f"unknown V5 head: {name}")


def _train_head(
    *,
    data: Dataset,
    indices: torch.Tensor,
    seed: int,
    generator_seed: int,
    config: dict,
) -> tuple[ApplicationVetoHead, dict, torch.Tensor]:
    random.seed(seed)
    torch.manual_seed(seed)
    center, scale = _median_mad(data.features[indices], dimension=0)
    head = ApplicationVetoHead(
        feature_center=center, feature_scale=scale, hidden_widths=[]
    )
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
        fused=False,
    )
    target = data.target[indices]
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    if min(positives, negatives) <= 0:
        raise ValueError("V5 head training split lacks one class")
    positive_weight = negatives / positives
    generator = torch.Generator().manual_seed(generator_seed)
    trace = []
    batch_size = int(config["batch_size"])
    for epoch in range(1, int(config["epochs"]) + 1):
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        losses = []
        maximum_gradient_norm = 0.0
        head.train()
        for offset in range(0, len(permutation), batch_size):
            batch = permutation[offset : offset + batch_size]
            logits = head.logits(data.features[batch])
            row_loss = F.binary_cross_entropy_with_logits(
                logits, data.target[batch], reduction="none"
            )
            class_weight = torch.where(
                data.target[batch] > 0.5,
                torch.full_like(row_loss, positive_weight),
                torch.ones_like(row_loss),
            )
            loss = torch.mean(row_loss * class_weight * data.weights[batch])
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite V5 application loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                list(head.parameters()), float(config["gradient_norm_clip"])
            )
            if not bool(torch.isfinite(norm)):
                raise FloatingPointError("non-finite V5 application gradient")
            maximum_gradient_norm = max(maximum_gradient_norm, float(norm))
            optimizer.step()
            losses.append(float(loss.detach()))
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config["epochs"]):
            trace.append(
                {
                    "epoch": epoch,
                    "train_loss": sum(losses) / len(losses),
                    "maximum_preclip_gradient_norm": maximum_gradient_norm,
                }
            )
    head.eval()
    with torch.no_grad():
        logits = head.logits(data.features)
    negative_indices = indices[data.target[indices] == 0]
    negative_location, negative_scale = _median_mad(logits[negative_indices])
    report = {
        "seed": seed,
        "generator_seed": generator_seed,
        "rows": len(indices),
        "positive_rows": int(data.target[indices].sum()),
        "negative_rows": int((data.target[indices] == 0).sum()),
        "positive_class_weight": positive_weight,
        "negative_logit_location": float(negative_location),
        "negative_logit_scale": float(negative_scale),
        "maximum_training_negative_logit": float(logits[negative_indices].max()),
        "trace": trace,
    }
    return head, report, logits


def _metrics_from_active(active: torch.Tensor, indices: torch.Tensor, data: Dataset) -> dict:
    logits = active.float()
    target = data.target[indices].bool()
    result: dict[str, object] = {
        "rows": len(indices),
        "positive_rows": int(target.sum()),
        "negative_rows": int((~target).sum()),
        "negative_active_fraction": (
            float(logits[indices][~target].mean()) if bool((~target).any()) else 0.0
        ),
        "mismatch_true_positive_rate": (
            float(logits[indices][target].mean()) if bool(target.any()) else 0.0
        ),
    }
    result["by_direction"] = {}
    for name in ("positive", "negative"):
        mask = torch.tensor(
            [data.directions[int(index)] == name for index in indices], dtype=torch.bool
        )
        result["by_direction"][name] = {
            "rows": int(mask.sum()),
            "true_positive_rate": float(logits[indices][mask].mean()) if bool(mask.any()) else 0.0,
        }
    result["by_label_swap"] = {}
    for value in (0, 1):
        mask = target & torch.tensor(
            [data.label_swaps[int(index)] == value for index in indices], dtype=torch.bool
        )
        result["by_label_swap"][str(value)] = {
            "rows": int(mask.sum()),
            "true_positive_rate": float(logits[indices][mask].mean()) if bool(mask.any()) else 0.0,
        }
    result["by_source"] = {}
    for source in sorted({data.sources[int(index)] for index in indices}):
        mask = torch.tensor(
            [data.sources[int(index)] == source for index in indices], dtype=torch.bool
        )
        source_target = target[mask]
        source_active = logits[indices][mask]
        result["by_source"][source] = {
            "rows": int(mask.sum()),
            "positive_rows": int(source_target.sum()),
            "active_fraction": float(source_active.mean()) if bool(mask.any()) else 0.0,
        }
    return result


def _minimum(metrics: dict, field: str) -> float:
    return min(float(value[field]) for value in metrics.values())


def _finite(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _checkpoint_contains_no_base_weights(checkpoint: dict) -> bool:
    """Audit the complete nested editor payload, not just the two V5 heads."""
    if checkpoint.get("base_model_weights_included") is not False:
        return False
    sites = checkpoint.get("sites")
    if not isinstance(sites, list) or len(sites) != 1:
        return False
    state_dict = sites[0].get("state_dict")
    if not isinstance(state_dict, dict):
        return False
    forbidden_prefixes = ("model.", "base_model.")
    return not any(str(name).startswith(forbidden_prefixes) for name in state_dict)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--router-contract", type=Path, required=True)
    parser.add_argument("--fit-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--train-governance-manifest", type=Path, required=True)
    parser.add_argument("--train-protected-manifest", type=Path, required=True)
    parser.add_argument("--audit-governance-manifest", type=Path, required=True)
    parser.add_argument("--audit-protected-manifest", type=Path, required=True)
    parser.add_argument("--capture-authorization", type=Path)
    parser.add_argument("--governance-capture-authorization", type=Path)
    parser.add_argument("--protected-capture-authorization", type=Path)
    parser.add_argument("--inherited-fit-contract", type=Path)
    parser.add_argument("--capture-archive-receipt", type=Path)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing V5 fit output: {args.output_dir}")

    router = json.loads(args.router_contract.read_text())
    fit_contract = json.loads(args.fit_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    recovery_mode = any(
        value is not None
        for value in (
            args.governance_capture_authorization,
            args.protected_capture_authorization,
            args.inherited_fit_contract,
            args.capture_archive_receipt,
        )
    )
    if recovery_mode:
        if args.capture_authorization is not None or any(
            value is None
            for value in (
                args.governance_capture_authorization,
                args.protected_capture_authorization,
                args.inherited_fit_contract,
                args.capture_archive_receipt,
            )
        ):
            raise ValueError(
                "V5.1 recovery requires exactly the two capture authorizations "
                "and the inherited fit contract"
            )
    elif args.capture_authorization is None:
        raise ValueError("legacy V5 fit requires --capture-authorization")
    required_router = {
        "status": "frozen_before_any_v5_capture_or_fit",
        "method_short_name": "GRC-DGE-V5",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_router.items():
        if router.get(field) != expected:
            raise ValueError(f"V5 router-contract mismatch: {field}")
    inherited_fit_contract = (
        json.loads(args.inherited_fit_contract.read_text()) if recovery_mode else fit_contract
    )
    required_inherited_fit_contract = {
        "status": "frozen_while_v5_capture_job_9113_pending_before_any_capture_output",
        "method_short_name": "GRC-DGE-V5",
        "capture_outputs_examined_before_freeze": False,
        "architecture_search_forbidden": True,
        "seed_search_forbidden": True,
        "epoch_search_forbidden": True,
        "threshold_search_forbidden": True,
    }
    for field, expected in required_inherited_fit_contract.items():
        if inherited_fit_contract.get(field) != expected:
            raise ValueError(f"V5 inherited fit-contract mismatch: {field}")
    if inherited_fit_contract.get("router_contract_sha256") != sha256_file(
        args.router_contract
    ):
        raise ValueError("V5 fit/router contract SHA mismatch")
    if recovery_mode:
        required_recovery_fit_contract = {
            "status": "recovery_lineage_frozen_after_protected_forward_started_before_any_protected_output_was_examined",
            "method_short_name": "GRC-DGE-V5",
            "recovery_revision": "V5.1",
            "code_version_minimum": 54,
            "scientific_hyperparameters_inherited_without_change": True,
            "protected_recovery_manifest_bound_by_later_fit_authorization": True,
            "protected_forward_started_before_recovery_lineage_freeze": True,
            "protected_outputs_examined_before_recovery_lineage_freeze": False,
            "capture_outputs_examined_before_freeze": False,
            "architecture_search_forbidden": True,
            "seed_search_forbidden": True,
            "epoch_search_forbidden": True,
            "threshold_search_forbidden": True,
        }
        for field, expected in required_recovery_fit_contract.items():
            if fit_contract.get(field) != expected:
                raise ValueError(f"V5.1 recovery fit-contract mismatch: {field}")
        if fit_contract.get("router_contract_sha256") != sha256_file(
            args.router_contract
        ):
            raise ValueError("V5.1 recovery fit/router contract SHA mismatch")
        if fit_contract.get("inherited_pre_capture_fit_contract_sha256") != sha256_file(
            args.inherited_fit_contract
        ):
            raise ValueError("V5.1 inherited fit-contract SHA mismatch")
        for section in (
            "training",
            "cross_fit",
            "final_fit",
            "required_audit_sources",
            "gates",
            "safety",
        ):
            if fit_contract.get(section) != inherited_fit_contract.get(section):
                raise ValueError(f"V5.1 scientific contract changed: {section}")
    static = router["bound_static_inputs"]
    for field, path in (
        ("subspace_fit_governance_capture_manifest_sha256", args.train_governance_manifest),
        ("subspace_fit_protected_capture_manifest_sha256", args.train_protected_manifest),
    ):
        if static.get(field) != sha256_file(path):
            raise ValueError(f"V5 training capture SHA mismatch: {field}")
    prior = router["bound_prior_evidence"]
    for field, path in (
        ("v3_checkpoint_sha256", args.v3_checkpoint),
        ("v3_fit_report_sha256", args.v3_fit_report),
    ):
        if prior.get(field) != sha256_file(path):
            raise ValueError(f"V5 V3 evidence mismatch: {field}")
    if recovery_mode:
        governance_capture_authorization = json.loads(
            args.governance_capture_authorization.read_text()
        )
        protected_capture_authorization = json.loads(
            args.protected_capture_authorization.read_text()
        )
        governance_authorization_sha = sha256_file(
            args.governance_capture_authorization
        )
        protected_authorization_sha = sha256_file(
            args.protected_capture_authorization
        )
        if governance_capture_authorization.get("stage") != (
            "governance_consensus_router_audit_capture"
        ):
            raise ValueError("V5.1 governance capture authorization stage mismatch")
        if protected_capture_authorization.get("stage") != (
            "governance_consensus_router_capture_recovery"
        ):
            raise ValueError("V5.1 protected capture authorization stage mismatch")
        if governance_authorization_sha != fit_contract.get(
            "source_governance_authorization_sha256"
        ):
            raise ValueError("V5.1 source governance authorization SHA mismatch")
        if protected_authorization_sha != fit_contract.get(
            "protected_recovery_authorization_sha256"
        ):
            raise ValueError("V5.1 protected recovery authorization SHA mismatch")
        for value, name in (
            (governance_capture_authorization, "governance"),
            (protected_capture_authorization, "protected"),
        ):
            for field, expected in {
                "operator_dev_accessed": False,
                "final_test_open": False,
                "final_test_open_count": 0,
                "production_rollout_approved": False,
            }.items():
                if value.get(field) != expected:
                    raise ValueError(f"V5.1 {name} authorization mismatch: {field}")
        for field, expected in {
            "source_job_id": 9140,
            "recompute_governance_forward": False,
            "engineering_retry_only": True,
            "scientific_inputs_unchanged": True,
        }.items():
            if protected_capture_authorization.get(field) != expected:
                raise ValueError(f"V5.1 protected recovery mismatch: {field}")
        capture_receipt = json.loads(args.capture_archive_receipt.read_text())
        for field, expected in {
            "schema_version": 1,
            "job_id": 9203,
            "run_id": "qwen3-8b-governance-consensus-router-capture-recovery-20260729T134830Z",
            "archive_sha256": "0f51da9d3960a3e72dcebe9f362287f2795066936bcf5b7534c19a9f4e040c2e",
            "cluster_shared_copy_verified": True,
            "host_data_copy_verified": True,
            "local_copy_verified": True,
        }.items():
            if capture_receipt.get(field) != expected:
                raise ValueError(f"V5.1 capture archive receipt mismatch: {field}")
        slurm_record = str(capture_receipt.get("slurm_terminal_record", ""))
        for marker in ("JobId=9203", "JobState=COMPLETED", "ExitCode=0:0"):
            if marker not in slurm_record:
                raise ValueError(f"V5.1 capture receipt lacks terminal marker: {marker}")
    else:
        capture_authorization = json.loads(args.capture_authorization.read_text())
        if capture_authorization.get("stage") != (
            "governance_consensus_router_audit_capture"
        ):
            raise ValueError("V5 capture authorization stage mismatch")
        if sha256_file(args.capture_authorization) != fit_contract.get(
            "capture_authorization_sha256"
        ):
            raise ValueError("V5 fit contract capture authorization mismatch")
        governance_authorization_sha = sha256_file(args.capture_authorization)
        protected_authorization_sha = governance_authorization_sha
    code_root = Path(str(authorization.get("code_root", "")))
    required_authorization = {
        "stage": "governance_consensus_router_fit",
        "execution_allowed": True,
        "code_root": (
            "/workspace/context-mismatch-qwen3-8b/code-v54"
            if recovery_mode
            else "/workspace/context-mismatch-qwen3-8b/code-v35"
        ),
        "immutable_code_bundle_manifest_sha256": sha256_file(code_root / "bundle.sha256"),
        "router_contract_sha256": sha256_file(args.router_contract),
        "fit_contract_sha256": sha256_file(args.fit_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "train_governance_manifest_sha256": sha256_file(args.train_governance_manifest),
        "train_protected_manifest_sha256": sha256_file(args.train_protected_manifest),
        "audit_governance_manifest_sha256": sha256_file(args.audit_governance_manifest),
        "audit_protected_manifest_sha256": sha256_file(args.audit_protected_manifest),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if recovery_mode:
        required_authorization.update(
            {
                "inherited_fit_contract_sha256": sha256_file(
                    args.inherited_fit_contract
                ),
                "governance_capture_authorization_sha256": governance_authorization_sha,
                "protected_capture_authorization_sha256": protected_authorization_sha,
                "capture_archive_receipt_sha256": sha256_file(
                    args.capture_archive_receipt
                ),
            }
        )
    else:
        required_authorization["capture_authorization_sha256"] = (
            governance_authorization_sha
        )
    for field, expected in required_authorization.items():
        if authorization.get(field) != expected:
            raise ValueError(f"V5 fit authorization mismatch: {field}")

    v3_checkpoint = torch.load(args.v3_checkpoint, map_location="cpu", weights_only=False)
    v3 = v3_from_checkpoint(v3_checkpoint)
    if v3.site.key != SITE_KEY:
        raise ValueError("V5 fit requires selected V3 site 27:mlp")
    for parameter in v3.parameters():
        parameter.requires_grad_(False)
    v3_before = {name: value.detach().clone() for name, value in v3.state_dict().items()}
    allowed_folds = set(int(value) for value in fit_contract["training"]["allowed_subspace_hash_folds"])
    if allowed_folds != set(range(6)):
        raise ValueError("unexpected V5 allowed fit folds")
    forbidden_folds = set(
        int(value) for value in fit_contract["training"]["forbidden_subspace_hash_folds"]
    )
    if forbidden_folds != {6, 7}:
        raise ValueError("unexpected V5 forbidden fit folds")
    if set(int(value) for value in fit_contract["cross_fit"]["folds"]) != allowed_folds:
        raise ValueError("V5 cross-fit fold set does not match the allowed training folds")
    train_governance, train_governance_manifest, excluded_governance = _load_capture(
        args.train_governance_manifest,
        expected_rows=6144,
        protected=False,
        allowed_folds=allowed_folds,
    )
    train_protected, train_protected_manifest, excluded_protected = _load_capture(
        args.train_protected_manifest,
        expected_rows=4008,
        protected=True,
        allowed_folds=allowed_folds,
    )
    audit_governance, audit_governance_manifest, _ = _load_capture(
        args.audit_governance_manifest,
        expected_rows=6144,
        protected=False,
        allowed_folds=None,
    )
    audit_protected, audit_protected_manifest, _ = _load_capture(
        args.audit_protected_manifest,
        expected_rows=4008,
        protected=True,
        allowed_folds=None,
    )
    for manifest in (train_governance_manifest, train_protected_manifest):
        if manifest.get("partition") != "subspace_fit":
            raise ValueError("V5 training capture is not subspace_fit")
    for manifest, expected_authorization_sha, name in (
        (audit_governance_manifest, governance_authorization_sha, "governance"),
        (audit_protected_manifest, protected_authorization_sha, "protected"),
    ):
        if manifest.get("partition") != "component_discovery":
            raise ValueError("V5 developmental audit is not component_discovery")
        if manifest.get("authorization_sha256") != expected_authorization_sha:
            raise ValueError(
                f"V5 developmental {name} audit capture authorization mismatch"
            )
        if manifest.get("router_contract_sha256") != sha256_file(args.router_contract):
            raise ValueError("V5 developmental audit router-contract mismatch")
        if manifest.get("final_test_open") is not False:
            raise ValueError("V5 developmental audit unexpectedly opens final_test")
        if manifest.get("final_test_open_count", 0) != 0:
            raise ValueError("V5 developmental audit final_test count mismatch")
        if manifest.get("production_rollout_approved") is not False:
            raise ValueError("V5 developmental audit unexpectedly approves production")
    train_group_ids = {
        _group_id(row)
        for row in train_governance["metadata"] + train_protected["metadata"]
    }
    audit_group_ids = {
        _group_id(row)
        for row in audit_governance["metadata"] + audit_protected["metadata"]
    }
    if train_group_ids & audit_group_ids:
        raise ValueError("V5 fit and component-discovery audit groups are not disjoint")

    train_data = _dataset(train_governance, train_protected, v3)
    audit_data = _dataset(audit_governance, audit_protected, v3)
    allowed_indices = torch.arange(len(train_data.target), dtype=torch.long)
    optimization = fit_contract["training"]
    cross_fit_records: dict[str, list[dict]] = {"all_negative": [], "matched_state": []}
    standardized: dict[str, torch.Tensor] = {
        name: torch.full((len(train_data.target),), float("nan"))
        for name in cross_fit_records
    }
    tails: dict[str, float] = {}
    head_seeds = {
        "all_negative": int(optimization["all_negative_head_seed"]),
        "matched_state": int(optimization["matched_state_head_seed"]),
    }
    for name, seed in head_seeds.items():
        subset = _head_subset(train_data, name)
        held_negative_scores = []
        for held_fold in sorted(allowed_folds):
            train_indices = torch.where(subset & (train_data.folds != held_fold))[0]
            held_indices = torch.where(train_data.folds == held_fold)[0]
            head, record, logits = _train_head(
                data=train_data,
                indices=train_indices,
                seed=seed,
                generator_seed=seed + 1009 * held_fold,
                config=optimization,
            )
            location = float(record["negative_logit_location"])
            scale = float(record["negative_logit_scale"])
            standardized[name][held_indices] = (logits[held_indices] - location) / scale
            held_head_negatives = held_indices[
                subset[held_indices] & (train_data.target[held_indices] == 0)
            ]
            held_negative_scores.append(standardized[name][held_head_negatives])
            record["held_fold"] = held_fold
            record["held_rows_scored"] = len(held_indices)
            record["held_head_negative_rows"] = len(held_head_negatives)
            cross_fit_records[name].append(record)
            del head
        if not bool(torch.isfinite(standardized[name]).all()):
            raise FloatingPointError(f"V5 {name} cross-fit scores are incomplete")
        tails[name] = float(torch.cat(held_negative_scores).max())
    safety_margin = float(fit_contract["cross_fit"]["safety_margin_standardized_units"])
    cross_active = (
        standardized["all_negative"] > tails["all_negative"] + safety_margin
    ) & (standardized["matched_state"] > tails["matched_state"] + safety_margin)
    cross_metrics = _metrics_from_active(cross_active, allowed_indices, train_data)
    cross_direction_minimum = _minimum(cross_metrics["by_direction"], "true_positive_rate")
    cross_swap_minimum = _minimum(cross_metrics["by_label_swap"], "true_positive_rate")

    final_heads: dict[str, ApplicationVetoHead] = {}
    final_records: dict[str, dict] = {}
    for name, seed in head_seeds.items():
        subset = _head_subset(train_data, name)
        train_indices = torch.where(subset)[0]
        head, record, logits = _train_head(
            data=train_data,
            indices=train_indices,
            seed=seed,
            generator_seed=seed,
            config=optimization,
        )
        negative_indices = train_indices[train_data.target[train_indices] == 0]
        maximum_negative = logits[negative_indices].max().float()
        exact_threshold = torch.nextafter(
            maximum_negative, torch.tensor(float("inf"), dtype=torch.float32)
        )
        robust_threshold = torch.tensor(
            float(record["negative_logit_location"])
            + float(record["negative_logit_scale"])
            * (tails[name] + safety_margin),
            dtype=torch.float32,
        )
        threshold = torch.maximum(exact_threshold, robust_threshold)
        if not bool(torch.isfinite(threshold)):
            raise FloatingPointError(f"non-finite V5 {name} threshold")
        head.threshold_logit.copy_(threshold)
        record.update(
            {
                "cross_fit_global_negative_tail": tails[name],
                "safety_margin_standardized_units": safety_margin,
                "exact_training_negative_threshold": float(exact_threshold),
                "robust_tail_threshold": float(robust_threshold),
                "threshold_logit": float(threshold),
            }
        )
        final_heads[name] = head
        final_records[name] = record

    with torch.no_grad():
        audit_all_logit = final_heads["all_negative"].logits(audit_data.features)
        audit_matched_logit = final_heads["matched_state"].logits(audit_data.features)
    audit_active = (
        audit_all_logit > final_heads["all_negative"].threshold_logit
    ) & (audit_matched_logit > final_heads["matched_state"].threshold_logit)
    audit_indices = torch.arange(len(audit_data.target), dtype=torch.long)
    audit_metrics = _metrics_from_active(audit_active, audit_indices, audit_data)
    audit_direction_minimum = _minimum(audit_metrics["by_direction"], "true_positive_rate")
    audit_swap_minimum = _minimum(audit_metrics["by_label_swap"], "true_positive_rate")
    required_sources = fit_contract["required_audit_sources"]
    expected_matched = set(required_sources["governance_matched"])
    expected_protected = set(required_sources["protected"])
    observed_matched = {
        source for source in audit_metrics["by_source"] if source.startswith("governance_matched:")
    }
    observed_protected = {
        source for source in audit_metrics["by_source"] if source.startswith("protected:")
    }
    max_v3_change = 0.0
    for name, before in v3_before.items():
        max_v3_change = max(
            max_v3_change,
            float(torch.max(torch.abs(v3.state_dict()[name] - before))),
        )
    gates = fit_contract["gates"]
    gate_checks = {
        "cross_fit_consensus_negative_exact_abstention": float(
            cross_metrics["negative_active_fraction"]
        )
        <= float(gates["cross_fit_consensus_negative_active_fraction_maximum"]),
        "cross_fit_consensus_mismatch_tpr": float(
            cross_metrics["mismatch_true_positive_rate"]
        )
        >= float(gates["cross_fit_consensus_mismatch_true_positive_rate_minimum"]),
        "cross_fit_consensus_each_direction_tpr": cross_direction_minimum
        >= float(gates["cross_fit_consensus_each_direction_true_positive_rate_minimum"]),
        "cross_fit_consensus_each_label_swap_tpr": cross_swap_minimum
        >= float(gates["cross_fit_consensus_each_label_swap_true_positive_rate_minimum"]),
        "audit_source_coverage_exact": observed_matched == expected_matched
        and observed_protected == expected_protected,
        "audit_each_governance_matched_family_exact_abstention": all(
            int(audit_metrics["by_source"][source]["rows"]) > 0
            and float(audit_metrics["by_source"][source]["active_fraction"])
            <= float(
                gates[
                    "component_audit_consensus_each_governance_matched_family_active_fraction_maximum"
                ]
            )
            for source in expected_matched
        ),
        "audit_each_protected_family_exact_abstention": all(
            int(audit_metrics["by_source"][source]["rows"]) > 0
            and float(audit_metrics["by_source"][source]["active_fraction"])
            <= float(
                gates[
                    "component_audit_consensus_each_protected_family_active_fraction_maximum"
                ]
            )
            for source in expected_protected
        ),
        "audit_mismatch_tpr": float(audit_metrics["mismatch_true_positive_rate"])
        >= float(gates["component_audit_consensus_mismatch_true_positive_rate_minimum"]),
        "audit_each_direction_tpr": audit_direction_minimum
        >= float(gates["component_audit_consensus_each_direction_true_positive_rate_minimum"]),
        "audit_each_label_swap_tpr": audit_swap_minimum
        >= float(gates["component_audit_consensus_each_label_swap_true_positive_rate_minimum"]),
        "v3_frozen_exact": max_v3_change
        <= float(gates["v3_frozen_parameter_max_absolute_change"]),
        "forbidden_subspace_rows_excluded": excluded_governance + excluded_protected > 0
        and set(train_data.folds.tolist()) == allowed_folds,
    }
    fit_eligible = all(gate_checks.values())
    args.output_dir.mkdir(parents=True)
    head_records = {}
    for name, head in final_heads.items():
        head_records[name] = {
            "architecture": "linear",
            "seed": head_seeds[name],
            "feature_center": head.feature_center.detach().cpu(),
            "feature_scale": head.feature_scale.detach().cpu(),
            "threshold_logit": float(head.threshold_logit),
            "state_dict": {
                key: value.detach().cpu() for key, value in head.state_dict().items()
            },
        }
    capture_lineage = (
        {
            "inherited_fit_contract_sha256": sha256_file(
                args.inherited_fit_contract
            ),
            "governance_capture_authorization_sha256": governance_authorization_sha,
            "protected_capture_authorization_sha256": protected_authorization_sha,
            "capture_archive_receipt_sha256": sha256_file(
                args.capture_archive_receipt
            ),
        }
        if recovery_mode
        else {"capture_authorization_sha256": governance_authorization_sha}
    )
    checkpoint = {
        "schema_version": 1,
        "kind": "group_robust_consensus_directional_governance_editor",
        "method": "GRC-DGE-V5",
        "router_contract_sha256": sha256_file(args.router_contract),
        "fit_contract_sha256": sha256_file(args.fit_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "authorization_sha256": sha256_file(args.execution_authorization),
        **capture_lineage,
        "v3_checkpoint": v3_checkpoint,
        "v3_expert_frozen": True,
        "v3_max_absolute_change": max_v3_change,
        "application_heads": head_records,
        "application_trainable_parameter_count": sum(
            parameter.numel() for head in final_heads.values() for parameter in head.parameters()
        ),
        "base_model_weights_included": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    checkpoint_path = args.output_dir / "consensus_directional_editor__layer_27__mlp.pt"
    incoming = args.output_dir / f".{checkpoint_path.name}.incoming"
    torch.save(checkpoint, incoming)
    os.replace(incoming, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = v5_from_checkpoint(loaded)
    checkpoint_contains_no_base_weights = _checkpoint_contains_no_base_weights(
        loaded["v3_checkpoint"]
    ) and not any(
        str(name).startswith(("model.", "base_model."))
        for record in head_records.values()
        for name in record["state_dict"]
    ) and loaded.get("base_model_weights_included") is False
    gate_checks["checkpoint_contains_exactly_two_linear_heads"] = (
        set(loaded["application_heads"]) == {"all_negative", "matched_state"}
        and reloaded.trainable_parameter_count == 2 * (APPLICATION_FEATURE_WIDTH + 1)
    )
    gate_checks["checkpoint_contains_no_base_weights"] = checkpoint_contains_no_base_weights
    fit_eligible = all(gate_checks.values())
    report = {
        "schema_version": 1,
        "stage": "governance_consensus_router_fit",
        "method": "GRC-DGE-V5",
        "fit_complete": True,
        "fit_eligible": fit_eligible,
        "training_rows": len(train_data.target),
        "forbidden_rows_excluded": {
            "governance": excluded_governance,
            "protected": excluded_protected,
            "total": excluded_governance + excluded_protected,
        },
        "cross_fit_heads": cross_fit_records,
        "cross_fit_global_negative_tails": tails,
        "cross_fit_consensus": cross_metrics,
        "cross_fit_min_direction_tpr": cross_direction_minimum,
        "cross_fit_min_label_swap_tpr": cross_swap_minimum,
        "final_heads": final_records,
        "developmental_audit": audit_metrics,
        "audit_min_direction_tpr": audit_direction_minimum,
        "audit_min_label_swap_tpr": audit_swap_minimum,
        "gate_checks": gate_checks,
        "v3_max_absolute_change": max_v3_change,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "router_contract_sha256": sha256_file(args.router_contract),
        "fit_contract_sha256": sha256_file(args.fit_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "train_governance_manifest_sha256": sha256_file(args.train_governance_manifest),
        "train_protected_manifest_sha256": sha256_file(args.train_protected_manifest),
        "audit_governance_manifest_sha256": sha256_file(args.audit_governance_manifest),
        "audit_protected_manifest_sha256": sha256_file(args.audit_protected_manifest),
        **capture_lineage,
        "authorization_sha256": sha256_file(args.execution_authorization),
        "component_discovery_audit_used_for_tuning": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not _finite(report):
        raise FloatingPointError("V5 fit report contains non-finite metrics")
    atomic_write_text(
        args.output_dir / "consensus_router_fit_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(_canonical(report))


if __name__ == "__main__":
    main()
