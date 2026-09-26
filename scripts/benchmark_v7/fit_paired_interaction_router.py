#!/usr/bin/env python3
"""Fit and audit the frozen PAIR-GE V5.2 interaction-aware router."""

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
from scripts.benchmark_v6.fit_consensus_router import (
    FAMILY_CROSSWALK,
    SITE_KEY,
    _checkpoint_contains_no_base_weights,
    _extract_features,
    _finite,
    _group_id,
    _load_capture,
    _median_mad,
    _metrics_from_active,
    _minimum,
)
from scripts.benchmark_v7.paired_interaction_governance import (
    FEATURE_SCHEMA,
    HEAD_NAMES,
    PAIR_INTERACTION_FEATURE_WIDTH,
    PairedInteractionRouter,
    editor_from_checkpoint as pair_ge_from_checkpoint,
    expand_pair_interaction_features,
)


CODE_ROOT = Path("/workspace/context-mismatch-qwen3-8b/code-v55")
CAPTURE_RUN_ID = "qwen3-8b-governance-consensus-router-capture-recovery-20260729T134830Z"
CAPTURE_JOB_ID = 9203
CAPTURE_ARCHIVE_SHA256 = "0f51da9d3960a3e72dcebe9f362287f2795066936bcf5b7534c19a9f4e040c2e"
GOVERNANCE_AUTH_SHA256 = "1503ec72ad6e1db673c7af71691a739e5606f74da178d78b3364e7a83b993940"
PROTECTED_AUTH_SHA256 = "10cc2518844d46ab510141d4f4ffd99afc1b86685f58a8729c0c12649f2e1195"
V5_1_REPORT_SHA256 = "f5c028e07ab0d26bb9f0dc0fb074739434fbe29c68e719bfb591df84305d30b5"
V5_1_RECEIPT_SHA256 = "2389d32f9961162d5d79436775e62fee72a6d808ec2c40f70b2d7b4b14310dd2"


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _pair_key(row: dict) -> str:
    return _canonical(
        {
            "item_id": row["item_id"],
            "declared_role": row["declared_role"],
            "history_style": row["history_style"],
            "history_realization": int(row["history_realization"]),
            "task_requirement": row["task_requirement"],
            "label_swap": int(row["label_swap"]),
        }
    )


@dataclass(frozen=True)
class PairDataset:
    features: torch.Tensor
    target: torch.Tensor
    weights: torch.Tensor
    folds: torch.Tensor
    sources: tuple[str, ...]
    directions: tuple[str, ...]
    label_swaps: tuple[int, ...]
    pair_keys: tuple[str, ...]


def _dataset(governance: dict, protected: dict, v3) -> PairDataset:
    base = torch.cat(
        (_extract_features(v3, governance), _extract_features(v3, protected)), dim=0
    )
    features = expand_pair_interaction_features(base)
    targets: list[float] = []
    weights: list[float] = []
    folds: list[int] = []
    sources: list[str] = []
    directions: list[str] = []
    swaps: list[int] = []
    pair_keys: list[str] = []
    pair_audit: dict[str, list[bool]] = {}
    pair_folds: dict[str, set[int]] = {}
    for row, fold in zip(governance["metadata"], governance["folds"], strict=True):
        mismatch = str(row["history_condition"]) != _matched_history(row)
        targets.append(1.0 if mismatch else 0.0)
        weights.append(1.0)
        folds.append(int(fold))
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
        key = _pair_key(row)
        pair_keys.append(key)
        pair_audit.setdefault(key, []).append(mismatch)
        pair_folds.setdefault(key, set()).add(int(fold))
    if len(pair_audit) * 2 != len(governance["metadata"]):
        raise ValueError("PAIR-GE governance counterfactual pair count mismatch")
    if any(sorted(values) != [False, True] for values in pair_audit.values()):
        raise ValueError("PAIR-GE pair does not contain one matched and one mismatch row")
    if any(len(values) != 1 for values in pair_folds.values()):
        raise ValueError("PAIR-GE counterfactual pair crosses hash folds")

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
        folds.append(int(fold))
        sources.append(f"protected:{family}")
        directions.append("none")
        swaps.append(int(row.get("label_swap", -1)))
        pair_keys.append("")
    result = PairDataset(
        features=features,
        target=torch.tensor(targets, dtype=torch.float32),
        weights=torch.tensor(weights, dtype=torch.float32),
        folds=torch.tensor(folds, dtype=torch.long),
        sources=tuple(sources),
        directions=tuple(directions),
        label_swaps=tuple(swaps),
        pair_keys=tuple(pair_keys),
    )
    if result.features.shape != (len(result.target), PAIR_INTERACTION_FEATURE_WIDTH):
        raise ValueError("PAIR-GE dataset feature matrix shape mismatch")
    return result


def _head_subset(data: PairDataset, head_index: int) -> torch.Tensor:
    if head_index == 0:
        return torch.ones(len(data.target), dtype=torch.bool)
    if head_index == 1:
        return torch.tensor(
            [
                bool(data.target[index] > 0.5)
                or data.sources[index].startswith("governance_matched:")
                for index in range(len(data.target))
            ],
            dtype=torch.bool,
        )
    raise ValueError("PAIR-GE head index must be zero or one")


def _pair_indices(data: PairDataset, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    allowed = set(int(value) for value in indices.tolist())
    grouped: dict[str, dict[bool, int]] = {}
    for index in indices.tolist():
        key = data.pair_keys[index]
        if not key:
            continue
        mismatch = bool(data.target[index] > 0.5)
        if mismatch in grouped.setdefault(key, {}):
            raise ValueError("duplicate PAIR-GE counterfactual cell")
        grouped[key][mismatch] = index
    if any(set(value) != {False, True} for value in grouped.values()):
        raise ValueError("training split cuts a PAIR-GE counterfactual pair")
    if any(index not in allowed for value in grouped.values() for index in value.values()):
        raise ValueError("PAIR-GE pair index escaped the training split")
    ordered = sorted(grouped)
    mismatch = torch.tensor([grouped[key][True] for key in ordered], dtype=torch.long)
    matched = torch.tensor([grouped[key][False] for key in ordered], dtype=torch.long)
    return mismatch, matched


def _group_labels(
    data: PairDataset, indices: torch.Tensor, head_subset: torch.Tensor
) -> dict[str, torch.Tensor]:
    groups: dict[str, list[int]] = {}
    for index in indices.tolist():
        if not bool(head_subset[index]):
            continue
        if bool(data.target[index] > 0.5):
            cell = f"positive:{data.directions[index]}:swap{data.label_swaps[index]}"
        else:
            cell = f"negative:{data.sources[index]}"
        groups.setdefault(f"fold{int(data.folds[index])}:{cell}", []).append(index)
    if not groups:
        raise ValueError("PAIR-GE robust loss has no groups")
    return {
        name: torch.tensor(values, dtype=torch.long) for name, values in sorted(groups.items())
    }


def _head_loss(
    *,
    logits: torch.Tensor,
    data: PairDataset,
    indices: torch.Tensor,
    head_index: int,
    config: dict,
    pair_mismatch: torch.Tensor,
    pair_matched: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    subset = _head_subset(data, head_index)
    head_indices = indices[subset[indices]]
    target = data.target[head_indices]
    positives = float(target.sum())
    negatives = float(len(target) - positives)
    if min(positives, negatives) <= 0:
        raise ValueError("PAIR-GE head training split lacks one class")
    positive_weight = negatives / positives
    row_loss = F.binary_cross_entropy_with_logits(
        logits[head_indices, head_index], target, reduction="none"
    )
    class_weight = torch.where(
        target > 0.5,
        torch.full_like(row_loss, positive_weight),
        torch.ones_like(row_loss),
    )
    weighted = row_loss * class_weight * data.weights[head_indices]
    bce = weighted.mean()

    index_to_position = {
        int(index): position for position, index in enumerate(head_indices.tolist())
    }
    group_losses = []
    for group_indices in _group_labels(data, indices, subset).values():
        positions = torch.tensor(
            [index_to_position[int(index)] for index in group_indices.tolist()],
            dtype=torch.long,
        )
        group_losses.append(weighted[positions].mean())
    stacked_groups = torch.stack(group_losses)
    temperature = float(config["worst_group_temperature"])
    robust = temperature * (
        torch.logsumexp(stacked_groups / temperature, dim=0)
        - math.log(len(stacked_groups))
    )

    rank_difference = (
        logits[pair_mismatch, head_index] - logits[pair_matched, head_index]
    )
    ranking = F.softplus(
        float(config["paired_ranking_margin_logit"]) - rank_difference
    ).mean()

    negative_indices = head_indices[data.target[head_indices] == 0]
    positive_indices = head_indices[data.target[head_indices] > 0.5]
    top_k = min(int(config["negative_tail_top_k"]), len(negative_indices))
    negative_tail = torch.topk(
        logits[negative_indices, head_index], k=top_k, largest=True
    ).values.mean()
    separation = F.softplus(
        float(config["negative_tail_margin_logit"])
        + negative_tail
        - logits[positive_indices, head_index]
    ).mean()
    total = (
        float(config["row_bce_weight"]) * bce
        + float(config["worst_group_weight"]) * robust
        + float(config["paired_ranking_weight"]) * ranking
        + float(config["negative_tail_separation_weight"]) * separation
    )
    return total, {
        "positive_class_weight": positive_weight,
        "row_bce": float(bce.detach()),
        "worst_group_smooth_max": float(robust.detach()),
        "paired_ranking": float(ranking.detach()),
        "negative_tail_separation": float(separation.detach()),
        "negative_tail_top_k_mean": float(negative_tail.detach()),
        "pair_count": len(pair_mismatch),
    }


def _train_router(
    *,
    data: PairDataset,
    indices: torch.Tensor,
    seed: int,
    config: dict,
) -> tuple[PairedInteractionRouter, dict, torch.Tensor]:
    random.seed(seed)
    torch.manual_seed(seed)
    center, scale = _median_mad(data.features[indices], dimension=0)
    router = PairedInteractionRouter(
        feature_center=center,
        feature_scale=scale,
        hidden_widths=config["hidden_widths"],
    )
    optimizer = torch.optim.AdamW(
        router.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        foreach=False,
        fused=False,
    )
    pair_mismatch, pair_matched = _pair_indices(data, indices)
    trace = []
    for epoch in range(1, int(config["epochs"]) + 1):
        router.train()
        logits = router.logits(data.features)
        losses = []
        details = []
        for head_index in range(2):
            loss, detail = _head_loss(
                logits=logits,
                data=data,
                indices=indices,
                head_index=head_index,
                config=config,
                pair_mismatch=pair_mismatch,
                pair_matched=pair_matched,
            )
            losses.append(loss)
            details.append(detail)
        loss = torch.stack(losses).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite PAIR-GE training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(
            list(router.parameters()), float(config["gradient_norm_clip"])
        )
        if not bool(torch.isfinite(norm)):
            raise FloatingPointError("non-finite PAIR-GE gradient norm")
        optimizer.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == int(config["epochs"]):
            trace.append(
                {
                    "epoch": epoch,
                    "total_loss": float(loss.detach()),
                    "maximum_preclip_gradient_norm": float(norm),
                    "heads": {
                        name: details[index] for index, name in enumerate(HEAD_NAMES)
                    },
                }
            )
    router.eval()
    with torch.no_grad():
        logits = router.logits(data.features)
    head_records = {}
    for head_index, name in enumerate(HEAD_NAMES):
        subset = _head_subset(data, head_index)
        negative_indices = indices[subset[indices] & (data.target[indices] == 0)]
        positive_indices = indices[subset[indices] & (data.target[indices] > 0.5)]
        location, negative_scale = _median_mad(logits[negative_indices, head_index])
        head_records[name] = {
            "positive_rows": len(positive_indices),
            "negative_rows": len(negative_indices),
            "negative_logit_location": float(location),
            "negative_logit_scale": float(negative_scale),
            "maximum_training_negative_logit": float(
                logits[negative_indices, head_index].max()
            ),
            "minimum_training_positive_logit": float(
                logits[positive_indices, head_index].min()
            ),
        }
    report = {
        "seed": seed,
        "rows": len(indices),
        "counterfactual_pairs": len(pair_mismatch),
        "head_calibration": head_records,
        "trace": trace,
    }
    return router, report, logits


def _validate_terminal_capture_receipt(path: Path) -> None:
    value = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "job_id": CAPTURE_JOB_ID,
        "run_id": CAPTURE_RUN_ID,
        "archive_sha256": CAPTURE_ARCHIVE_SHA256,
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"PAIR-GE capture receipt mismatch: {field}")
    record = str(value.get("slurm_terminal_record", ""))
    for marker in ("JobId=9203", "JobState=COMPLETED", "ExitCode=0:0"):
        if marker not in record:
            raise ValueError(f"PAIR-GE capture receipt lacks terminal marker: {marker}")


def _validate_v5_1_falsifier(report_path: Path, receipt_path: Path) -> None:
    if sha256_file(report_path) != V5_1_REPORT_SHA256:
        raise ValueError("PAIR-GE V5.1 report SHA mismatch")
    if sha256_file(receipt_path) != V5_1_RECEIPT_SHA256:
        raise ValueError("PAIR-GE V5.1 receipt SHA mismatch")
    report = json.loads(report_path.read_text())
    if report.get("method") != "GRC-DGE-V5" or report.get("fit_eligible") is not False:
        raise ValueError("PAIR-GE requires the terminal V5.1 fit falsifier")
    receipt = json.loads(receipt_path.read_text())
    for field, expected in {
        "job_id": 9205,
        "archive_sha256": "6e038c155b88410a0065d3d504faf505e706751c7f4083956729b252582f0778",
        "cluster_shared_copy_verified": True,
        "host_data_copy_verified": True,
        "local_copy_verified": True,
    }.items():
        if receipt.get(field) != expected:
            raise ValueError(f"PAIR-GE V5.1 receipt mismatch: {field}")


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
    parser.add_argument("--governance-capture-authorization", type=Path, required=True)
    parser.add_argument("--protected-capture-authorization", type=Path, required=True)
    parser.add_argument("--capture-archive-receipt", type=Path, required=True)
    parser.add_argument("--v5-1-fit-report", type=Path, required=True)
    parser.add_argument("--v5-1-fit-receipt", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing PAIR-GE output: {args.output_dir}")

    router_contract = json.loads(args.router_contract.read_text())
    fit_contract = json.loads(args.fit_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    for field, expected in {
        "status": "frozen_after_terminal_v5_1_falsifier_before_any_v5_2_fit",
        "method_short_name": "PAIR-GE-V5.2",
        "code_version_minimum": 55,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if router_contract.get(field) != expected:
            raise ValueError(f"PAIR-GE router contract mismatch: {field}")
    for field, expected in {
        "status": "frozen_before_any_v5_2_fit_execution",
        "method_short_name": "PAIR-GE-V5.2",
        "code_version_minimum": 55,
        "component_discovery_is_developmental_only": True,
        "operator_dev_untouched_confirmatory_selection_required": True,
        "architecture_frozen_before_v5_2_fit": True,
        "seed_frozen_before_v5_2_fit": True,
        "epoch_frozen_before_v5_2_fit": True,
        "threshold_rule_frozen_before_v5_2_fit": True,
    }.items():
        if fit_contract.get(field) != expected:
            raise ValueError(f"PAIR-GE fit contract mismatch: {field}")
    if fit_contract.get("router_contract_sha256") != sha256_file(args.router_contract):
        raise ValueError("PAIR-GE fit/router contract SHA mismatch")
    if fit_contract["training"].get("hidden_widths") != [96, 48]:
        raise ValueError("PAIR-GE hidden widths changed after freeze")
    if fit_contract["training"].get("epochs") != 180:
        raise ValueError("PAIR-GE epoch count changed after freeze")
    if router_contract["application_features"].get("expanded_input_width") != (
        PAIR_INTERACTION_FEATURE_WIDTH
    ):
        raise ValueError("PAIR-GE feature contract width mismatch")

    prior = router_contract["bound_prior_evidence"]
    capture = router_contract["bound_capture_evidence"]
    for field, path in (
        ("v3_checkpoint_sha256", args.v3_checkpoint),
        ("v3_fit_report_sha256", args.v3_fit_report),
        ("v5_1_fit_report_sha256", args.v5_1_fit_report),
        ("v5_1_fit_receipt_sha256", args.v5_1_fit_receipt),
    ):
        if prior.get(field) != sha256_file(path):
            raise ValueError(f"PAIR-GE prior evidence mismatch: {field}")
    for field, path in (
        ("subspace_fit_governance_manifest_sha256", args.train_governance_manifest),
        ("subspace_fit_protected_manifest_sha256", args.train_protected_manifest),
        ("developmental_governance_manifest_sha256", args.audit_governance_manifest),
        ("developmental_protected_manifest_sha256", args.audit_protected_manifest),
        ("capture_receipt_sha256", args.capture_archive_receipt),
    ):
        if capture.get(field) != sha256_file(path):
            raise ValueError(f"PAIR-GE capture evidence mismatch: {field}")
    _validate_terminal_capture_receipt(args.capture_archive_receipt)
    _validate_v5_1_falsifier(args.v5_1_fit_report, args.v5_1_fit_receipt)

    governance_auth_sha = sha256_file(args.governance_capture_authorization)
    protected_auth_sha = sha256_file(args.protected_capture_authorization)
    if governance_auth_sha != GOVERNANCE_AUTH_SHA256:
        raise ValueError("PAIR-GE governance capture authorization mismatch")
    if protected_auth_sha != PROTECTED_AUTH_SHA256:
        raise ValueError("PAIR-GE protected capture authorization mismatch")
    for value, stage, name in (
        (
            json.loads(args.governance_capture_authorization.read_text()),
            "governance_consensus_router_audit_capture",
            "governance",
        ),
        (
            json.loads(args.protected_capture_authorization.read_text()),
            "governance_consensus_router_capture_recovery",
            "protected",
        ),
    ):
        if value.get("stage") != stage:
            raise ValueError(f"PAIR-GE {name} capture stage mismatch")
        for field, expected in {
            "operator_dev_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }.items():
            if value.get(field) != expected:
                raise ValueError(f"PAIR-GE {name} capture safety mismatch: {field}")

    code_root = Path(str(authorization.get("code_root", "")))
    required_authorization = {
        "stage": "governance_pair_interaction_fit",
        "method": "PAIR-GE-V5.2",
        "execution_allowed": True,
        "code_root": str(CODE_ROOT),
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
        "governance_capture_authorization_sha256": governance_auth_sha,
        "protected_capture_authorization_sha256": protected_auth_sha,
        "capture_archive_receipt_sha256": sha256_file(args.capture_archive_receipt),
        "v5_1_fit_report_sha256": sha256_file(args.v5_1_fit_report),
        "v5_1_fit_receipt_sha256": sha256_file(args.v5_1_fit_receipt),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_authorization.items():
        if authorization.get(field) != expected:
            raise ValueError(f"PAIR-GE fit authorization mismatch: {field}")

    v3_checkpoint = torch.load(args.v3_checkpoint, map_location="cpu", weights_only=False)
    v3 = v3_from_checkpoint(v3_checkpoint)
    if v3.site.key != SITE_KEY:
        raise ValueError("PAIR-GE requires selected V3 site 27:mlp")
    for parameter in v3.parameters():
        parameter.requires_grad_(False)
    v3_before = {name: value.detach().clone() for name, value in v3.state_dict().items()}

    allowed_folds = set(
        int(value) for value in fit_contract["training"]["allowed_subspace_hash_folds"]
    )
    forbidden_folds = set(
        int(value) for value in fit_contract["training"]["forbidden_subspace_hash_folds"]
    )
    if allowed_folds != set(range(6)) or forbidden_folds != {6, 7}:
        raise ValueError("PAIR-GE fit-fold contract mismatch")
    if set(int(value) for value in fit_contract["cross_fit"]["folds"]) != allowed_folds:
        raise ValueError("PAIR-GE cross-fit fold mismatch")

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
            raise ValueError("PAIR-GE training capture is not subspace_fit")
    for manifest, auth_sha, name in (
        (audit_governance_manifest, governance_auth_sha, "governance"),
        (audit_protected_manifest, protected_auth_sha, "protected"),
    ):
        if manifest.get("partition") != "component_discovery":
            raise ValueError(f"PAIR-GE developmental {name} capture partition mismatch")
        if manifest.get("authorization_sha256") != auth_sha:
            raise ValueError(f"PAIR-GE developmental {name} authorization mismatch")
        if manifest.get("final_test_open") is not False:
            raise ValueError("PAIR-GE developmental capture opened final test")
        if manifest.get("final_test_open_count", 0) != 0:
            raise ValueError("PAIR-GE developmental capture final-test count mismatch")
        if manifest.get("production_rollout_approved") is not False:
            raise ValueError("PAIR-GE developmental capture approved production")
    train_group_ids = {
        _group_id(row)
        for row in train_governance["metadata"] + train_protected["metadata"]
    }
    audit_group_ids = {
        _group_id(row)
        for row in audit_governance["metadata"] + audit_protected["metadata"]
    }
    if train_group_ids & audit_group_ids:
        raise ValueError("PAIR-GE fit and developmental groups are not disjoint")

    train_data = _dataset(train_governance, train_protected, v3)
    audit_data = _dataset(audit_governance, audit_protected, v3)
    allowed_indices = torch.arange(len(train_data.target), dtype=torch.long)
    optimization = fit_contract["training"]
    seed = int(optimization["seed"])
    standardized = torch.full((len(train_data.target), 2), float("nan"))
    held_records = []
    held_negative_scores: dict[str, list[torch.Tensor]] = {
        name: [] for name in HEAD_NAMES
    }
    for held_fold in sorted(allowed_folds):
        train_indices = torch.where(train_data.folds != held_fold)[0]
        held_indices = torch.where(train_data.folds == held_fold)[0]
        model, record, logits = _train_router(
            data=train_data,
            indices=train_indices,
            seed=seed + 1009 * held_fold,
            config=optimization,
        )
        record["held_fold"] = held_fold
        record["held_rows_scored"] = len(held_indices)
        for head_index, name in enumerate(HEAD_NAMES):
            calibration = record["head_calibration"][name]
            location = float(calibration["negative_logit_location"])
            scale = float(calibration["negative_logit_scale"])
            standardized[held_indices, head_index] = (
                logits[held_indices, head_index] - location
            ) / scale
            subset = _head_subset(train_data, head_index)
            negative_indices = held_indices[
                subset[held_indices] & (train_data.target[held_indices] == 0)
            ]
            if len(negative_indices) == 0:
                raise ValueError(f"PAIR-GE held fold lacks {name} negatives")
            held_negative_scores[name].append(
                standardized[negative_indices, head_index]
            )
            calibration["held_head_negative_rows"] = len(negative_indices)
        held_records.append(record)
        del model
    if not bool(torch.isfinite(standardized).all()):
        raise FloatingPointError("PAIR-GE cross-fit scores are incomplete")
    tails = {
        name: float(torch.cat(held_negative_scores[name]).max())
        for name in HEAD_NAMES
    }
    safety_margin = float(
        fit_contract["cross_fit"]["safety_margin_standardized_units"]
    )
    cross_active = (
        standardized
        > torch.tensor(
            [tails[name] + safety_margin for name in HEAD_NAMES], dtype=torch.float32
        )
    ).all(dim=-1)
    cross_metrics = _metrics_from_active(cross_active, allowed_indices, train_data)
    cross_direction_minimum = _minimum(
        cross_metrics["by_direction"], "true_positive_rate"
    )
    cross_swap_minimum = _minimum(
        cross_metrics["by_label_swap"], "true_positive_rate"
    )

    final_model, final_record, final_logits = _train_router(
        data=train_data,
        indices=allowed_indices,
        seed=seed,
        config=optimization,
    )
    thresholds = []
    for head_index, name in enumerate(HEAD_NAMES):
        subset = _head_subset(train_data, head_index)
        negative_indices = allowed_indices[
            subset[allowed_indices] & (train_data.target[allowed_indices] == 0)
        ]
        maximum_negative = final_logits[negative_indices, head_index].max().float()
        exact_threshold = torch.nextafter(
            maximum_negative, torch.tensor(float("inf"), dtype=torch.float32)
        )
        calibration = final_record["head_calibration"][name]
        robust_threshold = torch.tensor(
            float(calibration["negative_logit_location"])
            + float(calibration["negative_logit_scale"])
            * (tails[name] + safety_margin),
            dtype=torch.float32,
        )
        threshold = torch.maximum(exact_threshold, robust_threshold)
        if not bool(torch.isfinite(threshold)):
            raise FloatingPointError(f"PAIR-GE {name} threshold is non-finite")
        thresholds.append(threshold)
        calibration.update(
            {
                "cross_fit_global_negative_tail": tails[name],
                "safety_margin_standardized_units": safety_margin,
                "exact_training_negative_threshold": float(exact_threshold),
                "robust_tail_threshold": float(robust_threshold),
                "threshold_logit": float(threshold),
            }
        )
    final_model.threshold_logits.copy_(torch.stack(thresholds))

    with torch.no_grad():
        audit_logits = final_model.logits(audit_data.features)
    audit_active = (audit_logits > final_model.threshold_logits).all(dim=-1)
    audit_indices = torch.arange(len(audit_data.target), dtype=torch.long)
    audit_metrics = _metrics_from_active(audit_active, audit_indices, audit_data)
    audit_direction_minimum = _minimum(
        audit_metrics["by_direction"], "true_positive_rate"
    )
    audit_swap_minimum = _minimum(
        audit_metrics["by_label_swap"], "true_positive_rate"
    )
    required_sources = fit_contract["required_audit_sources"]
    expected_matched = set(required_sources["governance_matched"])
    expected_protected = set(required_sources["protected"])
    observed_matched = {
        source
        for source in audit_metrics["by_source"]
        if source.startswith("governance_matched:")
    }
    observed_protected = {
        source
        for source in audit_metrics["by_source"]
        if source.startswith("protected:")
    }
    max_v3_change = max(
        float(torch.max(torch.abs(v3.state_dict()[name] - before)))
        for name, before in v3_before.items()
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
        "developmental_audit_source_coverage_exact": observed_matched == expected_matched
        and observed_protected == expected_protected,
        "developmental_audit_each_governance_matched_family_exact_abstention": all(
            int(audit_metrics["by_source"][source]["rows"]) > 0
            and float(audit_metrics["by_source"][source]["active_fraction"])
            <= float(
                gates[
                    "developmental_audit_consensus_each_governance_matched_family_active_fraction_maximum"
                ]
            )
            for source in expected_matched
        ),
        "developmental_audit_each_protected_family_exact_abstention": all(
            int(audit_metrics["by_source"][source]["rows"]) > 0
            and float(audit_metrics["by_source"][source]["active_fraction"])
            <= float(
                gates[
                    "developmental_audit_consensus_each_protected_family_active_fraction_maximum"
                ]
            )
            for source in expected_protected
        ),
        "developmental_audit_mismatch_tpr": float(
            audit_metrics["mismatch_true_positive_rate"]
        )
        >= float(
            gates["developmental_audit_consensus_mismatch_true_positive_rate_minimum"]
        ),
        "developmental_audit_each_direction_tpr": audit_direction_minimum
        >= float(
            gates[
                "developmental_audit_consensus_each_direction_true_positive_rate_minimum"
            ]
        ),
        "developmental_audit_each_label_swap_tpr": audit_swap_minimum
        >= float(
            gates[
                "developmental_audit_consensus_each_label_swap_true_positive_rate_minimum"
            ]
        ),
        "v3_frozen_exact": max_v3_change
        <= float(gates["v3_frozen_parameter_max_absolute_change"]),
        "forbidden_subspace_rows_excluded": excluded_governance + excluded_protected > 0
        and set(train_data.folds.tolist()) == allowed_folds,
    }

    args.output_dir.mkdir(parents=True)
    router_record = {
        "architecture": "shared_mlp_662_96_48_two_linear_heads",
        "feature_schema": FEATURE_SCHEMA,
        "hidden_widths": list(final_model.hidden_widths),
        "seed": seed,
        "feature_center": final_model.feature_center.detach().cpu(),
        "feature_scale": final_model.feature_scale.detach().cpu(),
        "threshold_logits": final_model.threshold_logits.detach().cpu(),
        "state_dict": {
            key: value.detach().cpu() for key, value in final_model.state_dict().items()
        },
    }
    checkpoint = {
        "schema_version": 1,
        "kind": "paired_interaction_governance_editor",
        "method": "PAIR-GE-V5.2",
        "feature_schema": FEATURE_SCHEMA,
        "router_contract_sha256": sha256_file(args.router_contract),
        "fit_contract_sha256": sha256_file(args.fit_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "v5_1_fit_report_sha256": sha256_file(args.v5_1_fit_report),
        "v5_1_fit_receipt_sha256": sha256_file(args.v5_1_fit_receipt),
        "v3_checkpoint": v3_checkpoint,
        "v3_expert_frozen": True,
        "v3_max_absolute_change": max_v3_change,
        "router": router_record,
        "application_trainable_parameter_count": sum(
            parameter.numel() for parameter in final_model.parameters()
        ),
        "base_model_weights_included": False,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    checkpoint_path = args.output_dir / "pair_ge_editor__layer_27__mlp.pt"
    incoming = args.output_dir / f".{checkpoint_path.name}.incoming"
    torch.save(checkpoint, incoming)
    os.replace(incoming, checkpoint_path)
    loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reloaded = pair_ge_from_checkpoint(loaded)
    checkpoint_contains_no_base_weights = (
        _checkpoint_contains_no_base_weights(loaded["v3_checkpoint"])
        and loaded.get("base_model_weights_included") is False
        and not any(
            str(name).startswith(("model.", "base_model."))
            for name in router_record["state_dict"]
        )
    )
    gate_checks["checkpoint_contains_one_shared_trunk_and_two_output_heads"] = (
        set(reloaded.site_editor.router.output_heads) == set(HEAD_NAMES)
        and tuple(reloaded.site_editor.router.hidden_widths) == (96, 48)
        and reloaded.trainable_parameter_count
        == checkpoint["application_trainable_parameter_count"]
    )
    gate_checks["checkpoint_contains_no_base_weights"] = checkpoint_contains_no_base_weights
    fit_eligible = all(gate_checks.values())
    report = {
        "schema_version": 1,
        "stage": "governance_pair_interaction_fit",
        "method": "PAIR-GE-V5.2",
        "fit_complete": True,
        "fit_eligible": fit_eligible,
        "training_rows": len(train_data.target),
        "training_counterfactual_pairs": sum(bool(value) for value in train_data.pair_keys)
        // 2,
        "feature_width": PAIR_INTERACTION_FEATURE_WIDTH,
        "forbidden_rows_excluded": {
            "governance": excluded_governance,
            "protected": excluded_protected,
            "total": excluded_governance + excluded_protected,
        },
        "cross_fit_models": held_records,
        "cross_fit_global_negative_tails": tails,
        "cross_fit_consensus": cross_metrics,
        "cross_fit_min_direction_tpr": cross_direction_minimum,
        "cross_fit_min_label_swap_tpr": cross_swap_minimum,
        "final_model": final_record,
        "developmental_audit": audit_metrics,
        "developmental_audit_min_direction_tpr": audit_direction_minimum,
        "developmental_audit_min_label_swap_tpr": audit_swap_minimum,
        "developmental_audit_confirmatory": False,
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
        "governance_capture_authorization_sha256": governance_auth_sha,
        "protected_capture_authorization_sha256": protected_auth_sha,
        "capture_archive_receipt_sha256": sha256_file(args.capture_archive_receipt),
        "v5_1_fit_report_sha256": sha256_file(args.v5_1_fit_report),
        "v5_1_fit_receipt_sha256": sha256_file(args.v5_1_fit_receipt),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not _finite(report):
        raise FloatingPointError("PAIR-GE fit report contains non-finite metrics")
    atomic_write_text(
        args.output_dir / "pair_ge_fit_report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(_canonical(report))


if __name__ == "__main__":
    main()
