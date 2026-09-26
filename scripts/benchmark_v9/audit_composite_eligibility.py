#!/usr/bin/env python3
"""Audit the exact frozen C-DGE composite route on the V4 fit captures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v5.abstaining_directional_governance import editor_from_checkpoint
from scripts.benchmark_v5.fit_abstaining_router import _dataset, _load_states


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    return True


def _tensor_tree_equal(left: Any, right: Any) -> bool:
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        return (
            isinstance(left, torch.Tensor)
            and isinstance(right, torch.Tensor)
            and left.dtype == right.dtype
            and tuple(left.shape) == tuple(right.shape)
            and torch.equal(left.cpu(), right.cpu())
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and set(left) == set(right)
            and all(_tensor_tree_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple))
            and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_tensor_tree_equal(a, b) for a, b in zip(left, right))
        )
    return left == right


def route_metrics(active: torch.Tensor, indices: torch.Tensor, data: Any) -> dict:
    """Return the frozen source/direction/swap metrics for one hard route."""
    current_active = active[indices].bool()
    target = data.target[indices].bool()
    result: dict[str, Any] = {
        "rows": int(indices.numel()),
        "positive_rows": int(target.sum()),
        "negative_rows": int((~target).sum()),
        "active_rows": int(current_active.sum()),
        "negative_active_fraction": (
            float(current_active[~target].float().mean()) if bool((~target).any()) else 0.0
        ),
        "mismatch_true_positive_rate": (
            float(current_active[target].float().mean()) if bool(target.any()) else 0.0
        ),
    }
    by_direction: dict[str, dict] = {}
    for name in ("positive", "negative"):
        mask = target & torch.tensor(
            [data.directions[int(index)] == name for index in indices], dtype=torch.bool
        )
        by_direction[name] = {
            "rows": int(mask.sum()),
            "true_positive_rate": (
                float(current_active[mask].float().mean()) if bool(mask.any()) else 0.0
            ),
        }
    by_label_swap: dict[str, dict] = {}
    for value in (0, 1):
        mask = target & torch.tensor(
            [data.label_swaps[int(index)] == value for index in indices], dtype=torch.bool
        )
        by_label_swap[str(value)] = {
            "rows": int(mask.sum()),
            "true_positive_rate": (
                float(current_active[mask].float().mean()) if bool(mask.any()) else 0.0
            ),
        }
    by_source: dict[str, dict] = {}
    for source in sorted({data.sources[int(index)] for index in indices}):
        mask = torch.tensor(
            [data.sources[int(index)] == source for index in indices], dtype=torch.bool
        )
        source_target = target[mask]
        by_source[source] = {
            "rows": int(mask.sum()),
            "positive_rows": int(source_target.sum()),
            "active_rows": int(current_active[mask].sum()),
            "active_fraction": float(current_active[mask].float().mean()),
        }
    result["by_direction"] = by_direction
    result["by_label_swap"] = by_label_swap
    result["by_source"] = by_source
    return result


def _minimum_tpr(report: dict, field: str) -> float:
    values = report[field].values()
    if not values or any(int(value["rows"]) <= 0 for value in values):
        return 0.0
    return min(float(value["true_positive_rate"]) for value in values)


def _source_subset(report: dict, prefix: str) -> dict[str, dict]:
    return {
        name: value
        for name, value in report["by_source"].items()
        if name.startswith(prefix)
    }


def _exact_zero_and_coverage(
    report: dict, *, prefix: str, expected_names: set[str]
) -> tuple[bool, bool]:
    observed = _source_subset(report, prefix)
    expected = {f"{prefix}{name}" for name in expected_names}
    coverage = set(observed) == expected
    exact_zero = coverage and all(
        int(value["rows"]) > 0
        and int(value["active_rows"]) == 0
        and float(value["active_fraction"]) == 0.0
        for value in observed.values()
    )
    return coverage, exact_zero


def _checkpoint_contains_base_weights(checkpoint: dict) -> bool:
    if checkpoint.get("base_model_weights_included") is not False:
        return True
    application = checkpoint.get("application_head", {}).get("state_dict", {})
    if any(str(name).startswith(("model.", "base_model.")) for name in application):
        return True
    embedded = checkpoint.get("v3_checkpoint", {})
    if embedded.get("base_model_weights_included") is not False:
        return True
    for site in embedded.get("sites", []):
        if any(
            str(name).startswith(("model.", "base_model."))
            for name in site.get("state_dict", {})
        ):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--v4-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-fit-report", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing audit output: {args.output}")

    contract = json.loads(args.composite_contract.read_text())
    v4_contract = json.loads(args.v4_contract.read_text())
    fit_report = json.loads(args.v4_fit_report.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    lineage = contract["frozen_lineage"]
    bound_paths = {
        "v3_contract_sha256": args.v3_contract,
        "v3_checkpoint_sha256": args.v3_checkpoint,
        "v3_fit_report_sha256": args.v3_fit_report,
        "v4_contract_sha256": args.v4_contract,
        "v4_checkpoint_sha256": args.v4_checkpoint,
        "v4_fit_report_sha256": args.v4_fit_report,
        "governance_capture_manifest_sha256": args.capture_manifest,
        "protected_capture_manifest_sha256": args.protected_capture_manifest,
    }
    for field, path in bound_paths.items():
        if sha256_file(path) != lineage[field]:
            raise ValueError(f"frozen lineage SHA mismatch: {field}")
    if fit_report.get("fit_complete") is not True or not isinstance(
        fit_report.get("fit_eligible"), bool
    ):
        raise ValueError("the terminal V4 fit result must be complete and boolean")
    if fit_report.get("checkpoint_sha256") != sha256_file(args.v4_checkpoint):
        raise ValueError("V4 checkpoint/report SHA mismatch")
    if fit_report.get("operator_dev_accessed") is not False:
        raise ValueError("V4 fit unexpectedly accessed operator_dev")
    if fit_report.get("protected_behavior_outputs_accessed") is not False:
        raise ValueError("V4 fit unexpectedly accessed protected behavior outputs")

    v4_checkpoint = torch.load(args.v4_checkpoint, map_location="cpu", weights_only=False)
    v3_checkpoint = torch.load(args.v3_checkpoint, map_location="cpu", weights_only=False)
    embedded_v3_exact = _tensor_tree_equal(v4_checkpoint.get("v3_checkpoint"), v3_checkpoint)
    editor = editor_from_checkpoint(v4_checkpoint)
    editor.eval()
    before = {name: value.detach().clone() for name, value in editor.state_dict().items()}

    governance = _load_states(args.capture_manifest, expected_rows=6144)
    protected = _load_states(args.protected_capture_manifest, expected_rows=4008)
    data = _dataset(governance, protected, editor.frozen_v3, v4_contract)
    with torch.no_grad():
        application_logits = editor.site_editor.application_head.logits(data.features)
    threshold = editor.site_editor.application_head.threshold_logit
    application_active = application_logits > threshold
    positive_route = data.features[:, -2]
    negative_route = data.features[:, -1]
    structural_active = (positive_route > 0) | (negative_route > 0)
    composite_active = structural_active & application_active
    if not bool(torch.isfinite(application_logits).all()):
        raise FloatingPointError("non-finite applicability logits")

    splits = {
        "calibration": torch.where(data.folds == 6)[0],
        "audit": torch.where(data.folds == 7)[0],
    }
    reports: dict[str, dict] = {}
    for split, indices in splits.items():
        if not int(indices.numel()):
            raise ValueError(f"empty {split} split")
        reports[split] = {
            "application": route_metrics(application_active, indices, data),
            "structural": route_metrics(structural_active, indices, data),
            "composite": route_metrics(composite_active, indices, data),
        }
        reports[split]["overlap"] = {
            "application_active_rows": int(application_active[indices].sum()),
            "structural_active_rows": int(structural_active[indices].sum()),
            "composite_active_rows": int(composite_active[indices].sum()),
            "application_only_rows": int(
                (application_active[indices] & ~structural_active[indices]).sum()
            ),
            "structural_only_rows": int(
                (structural_active[indices] & ~application_active[indices]).sum()
            ),
        }

    gates = contract["eligibility_gates"]
    expected_matched = set(gates["expected_governance_matched_families"])
    expected_protected = set(gates["expected_protected_families"])
    gate_checks: dict[str, bool] = {}
    for split in ("calibration", "audit"):
        composite = reports[split]["composite"]
        matched_coverage, matched_zero = _exact_zero_and_coverage(
            composite,
            prefix="governance_matched:",
            expected_names=expected_matched,
        )
        protected_coverage, protected_zero = _exact_zero_and_coverage(
            composite,
            prefix="protected:",
            expected_names=expected_protected,
        )
        gate_checks[f"{split}_governance_matched_source_coverage_exact"] = matched_coverage
        gate_checks[f"{split}_governance_matched_composite_exact_abstention"] = matched_zero
        gate_checks[f"{split}_protected_source_coverage_exact"] = protected_coverage
        gate_checks[f"{split}_protected_composite_exact_abstention"] = protected_zero
        gate_checks[f"{split}_composite_mismatch_tpr"] = (
            float(composite["mismatch_true_positive_rate"])
            >= float(gates[f"{split}_composite_mismatch_true_positive_rate_minimum"])
        )
        direction_gate = gates.get(
            f"{split}_composite_each_direction_true_positive_rate_minimum"
        )
        if direction_gate is not None:
            gate_checks[f"{split}_composite_each_direction_tpr"] = (
                _minimum_tpr(composite, "by_direction") >= float(direction_gate)
            )
        label_swap_gate = gates.get(
            f"{split}_composite_each_label_swap_true_positive_rate_minimum"
        )
        if label_swap_gate is not None:
            gate_checks[f"{split}_composite_each_label_swap_tpr"] = (
                _minimum_tpr(composite, "by_label_swap") >= float(label_swap_gate)
            )

    max_change = 0.0
    for name, previous in before.items():
        current = editor.state_dict()[name]
        max_change = max(max_change, float(torch.max(torch.abs(current - previous))))
    gate_checks.update(
        {
            "v3_embedded_checkpoint_exact_match": embedded_v3_exact,
            "frozen_editor_parameter_change_exact_zero": max_change == 0.0,
            "v4_checkpoint_contains_no_base_weights": not _checkpoint_contains_base_weights(
                v4_checkpoint
            ),
            "v4_checkpoint_contains_exactly_one_site": (
                len(v4_checkpoint.get("v3_checkpoint", {}).get("sites", [])) == 1
                and editor.site.key == "27:mlp"
            ),
            "checkpoint_safety_flags_exact": all(
                v4_checkpoint.get(field) == expected
                for field, expected in {
                    "base_model_weights_included": False,
                    "operator_dev_accessed": False,
                    "protected_behavior_outputs_accessed": False,
                    "final_test_open": False,
                    "final_test_open_count": 0,
                    "production_rollout_approved": False,
                }.items()
            ),
            "authorization_safety_flags_exact": all(
                authorization.get(field) == expected
                for field, expected in {
                    "operator_dev_accessed": False,
                    "protected_behavior_outputs_accessed": False,
                    "final_test_open": False,
                    "final_test_open_count": 0,
                    "production_rollout_approved": False,
                }.items()
            ),
        }
    )
    candidate_eligible = all(gate_checks.values())
    report = {
        "schema_version": 1,
        "stage": "governance_composite_eligibility_audit",
        "method": "C-DGE-V4.1",
        "evidence_role": contract["evidence_role"],
        "confirmatory": False,
        "audit_complete": True,
        "candidate_eligible": candidate_eligible,
        "candidate_may_be_locked": False,
        "gate_checks": gate_checks,
        "route_reports": reports,
        "rows": {
            "total": int(data.target.numel()),
            "governance_capture": 6144,
            "protected_capture": 4008,
            "calibration": int(splits["calibration"].numel()),
            "audit": int(splits["audit"].numel()),
        },
        "application_threshold_logit": float(threshold),
        "frozen_editor_max_absolute_change": max_change,
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "v4_contract_sha256": sha256_file(args.v4_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "v4_checkpoint_sha256": sha256_file(args.v4_checkpoint),
        "v4_fit_report_sha256": sha256_file(args.v4_fit_report),
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(args.v3_fit_report),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(
            args.protected_capture_manifest
        ),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not _finite(report):
        raise FloatingPointError("composite eligibility report contains non-finite metrics")
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
