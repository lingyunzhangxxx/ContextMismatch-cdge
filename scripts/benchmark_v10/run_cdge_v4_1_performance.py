#!/usr/bin/env python3
"""Run the frozen 24-cell baseline/V3/C-DGE performance benchmark on one NPU."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v1.histories import prefix_messages
from scripts.benchmark_v1.run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _prefill,
    _split_final_user,
    _suffix_logits_batch,
    _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import _load_model, _module
from scripts.benchmark_v1.run_protected_capture import _prefill_with_last_boundary_capture
from scripts.benchmark_v4.directional_governance import editor_from_checkpoint as v3_editor
from scripts.benchmark_v5.abstaining_directional_governance import editor_from_checkpoint as v4_editor
from scripts.benchmark_v10.cdge_contract import METHOD, validate_composite_prerequisite
from scripts.benchmark_v10.cdge_performance_design import (
    EXPECTED_CELLS,
    EXPECTED_CELL_KEY_SHA256,
    EXPECTED_MEASUREMENTS,
    METHODS,
    TIMED_REPEATS,
    WARMUP_REPEATS,
    cell_key,
    measurement_key,
    selected_performance_jobs,
)


STAGE = "governance_composite_performance"


def _require(value: dict, expected: dict, label: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"{label} mismatch: {field}")


def _same_tree(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return set(left) == set(right) and all(_same_tree(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(_same_tree(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def _validate(args: argparse.Namespace) -> tuple[dict, dict, list[dict]]:
    evaluation = json.loads(args.evaluation_contract.read_text())
    composite = json.loads(args.composite_report.read_text())
    receipt = json.loads(args.composite_receipt.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    design = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    validate_composite_prerequisite(composite, receipt)
    performance = evaluation.get("performance", {})
    _require(
        performance,
        {
            "stage": STAGE,
            "methods": list(METHODS),
            "same_model_and_prompt_tokens_required": True,
            "selection_rule": "lexicographically first operator_dev selection job in each benchmark x task_requirement x label_swap cell",
            "expected_cells": EXPECTED_CELLS,
            "batch_size": 1,
            "warmup_repeats": WARMUP_REPEATS,
            "timed_repeats": TIMED_REPEATS,
            "attn_implementation": "eager",
            "peak_memory_reset_after_model_load_required": True,
            "finite_measurements_required": True,
        },
        "performance contract",
    )
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev design audit failed")
    if design.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("operator-dev design/contract mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model revision mismatch")
    jobs = selected_performance_jobs(load_jsonl(args.manifest), crossover)
    required = {
        "stage": STAGE,
        "method": METHOD,
        "execution_allowed": True,
        "v3_checkpoint_sha256": sha256_file(args.v3_checkpoint),
        "v4_checkpoint_sha256": sha256_file(args.v4_checkpoint),
        "v3_editor_contract_sha256": sha256_file(args.v3_editor_contract),
        "v4_editor_contract_sha256": sha256_file(args.v4_editor_contract),
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "composite_report_sha256": sha256_file(args.composite_report),
        "composite_authorization_sha256": sha256_file(args.composite_authorization),
        "composite_receipt_sha256": sha256_file(args.composite_receipt),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "design_audit_sha256": sha256_file(args.design_audit),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_cells": EXPECTED_CELLS,
        "expected_cell_key_sha256": EXPECTED_CELL_KEY_SHA256,
        "expected_measurements": EXPECTED_MEASUREMENTS,
        "batch_size": 1,
        "warmup_repeats": WARMUP_REPEATS,
        "timed_repeats": TIMED_REPEATS,
        "attn_implementation": "eager",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    _require(authorization, required, "performance authorization")
    return authorization, crossover, jobs


def _synchronize() -> None:
    torch.npu.synchronize()


def _reset_peak(device: torch.device) -> int:
    _synchronize()
    torch.npu.reset_peak_memory_stats(device)
    return int(torch.npu.memory_allocated(device))


@torch.inference_mode()
def _forward_once(
    *,
    method: str,
    model,
    device: torch.device,
    prefix_ids: list[int],
    suffix_ids: list[int],
    label_ids: dict[str, int],
    manager,
) -> tuple[float, float]:
    if method == "baseline":
        base_cache = _prefill(model, device, prefix_ids)
    else:
        specs = list(manager.operators)
        layers = sorted({spec.layer for spec in specs})
        base_cache, captured = _prefill_with_last_boundary_capture(model, device, prefix_ids, layers)
        boundaries = {
            spec: captured[str(spec.layer)].to(device=device, dtype=torch.float32)
            for spec in specs
        }
        manager.install(
            model,
            gate={spec: 1.0 for spec in specs},
            module_resolver=_module,
            boundary_state=boundaries,
            collect_diagnostics=False,
        )
    try:
        logits, extended_cache = _suffix_logits_batch(
            model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
        )
    finally:
        if method != "baseline":
            manager.remove()
    del extended_cache
    del base_cache
    return float(logits[0][0]), float(logits[0][1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v4-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-editor-contract", type=Path, required=True)
    parser.add_argument("--v4-editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--composite-report", type=Path, required=True)
    parser.add_argument("--composite-authorization", type=Path, required=True)
    parser.add_argument("--composite-receipt", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager"])
    args = parser.parse_args()
    for path in (args.output, args.environment_output):
        if path.exists():
            raise FileExistsError(f"refusing existing performance output: {path}")
    authorization, crossover, jobs = _validate(args)
    v3_value = torch.load(args.v3_checkpoint, map_location="cpu", weights_only=False)
    v4_value = torch.load(args.v4_checkpoint, map_location="cpu", weights_only=False)
    if v3_value.get("method") != "DSGE-V3" or v4_value.get("method") != "ADSGE-V4":
        raise ValueError("unexpected editor checkpoint methods")
    if not _same_tree(v3_value, v4_value.get("v3_checkpoint")):
        raise ValueError("V4 frozen V3 expert does not match the formal V3 checkpoint")
    device = _device(args.device)
    if device.type != "npu" or not hasattr(torch, "npu") or torch.npu.device_count() != 1:
        raise RuntimeError("formal performance benchmark requires exactly one visible NPU")
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze base model")
    editors = {
        "DSGE-V3": v3_editor(v3_value).eval(),
        "C-DGE-V4.1": v4_editor(v4_value).eval(),
    }
    label_ids = _label_ids(tokenizer)
    reset_resident = _reset_peak(device)
    environment = {
        "schema_version": 1,
        "stage": STAGE,
        "method": METHOD,
        "methods": list(METHODS),
        "execution_authorization_sha256": sha256_file(args.execution_authorization),
        "expected_cells": EXPECTED_CELLS,
        "expected_cell_key_sha256": EXPECTED_CELL_KEY_SHA256,
        "expected_measurements": EXPECTED_MEASUREMENTS,
        "batch_size": 1,
        "warmup_repeats": WARMUP_REPEATS,
        "timed_repeats": TIMED_REPEATS,
        "attn_implementation": "eager",
        "timed_scope": "end_to_end_prefix_prefill_boundary_capture_if_required_and_suffix_forward",
        "tokenization_excluded_from_timing": True,
        "editor_device_transfer_excluded_from_timing": True,
        "peak_memory_reset_after_model_load": True,
        "post_model_load_resident_npu_memory_bytes": reset_resident,
        "device": str(device),
        "torch_version": torch.__version__,
        "base_model_trainable_parameters": 0,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n")

    completed = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as handle:
        for cell_index, job in enumerate(jobs):
            current_cell = cell_key(job)
            role = job["declared_role"]
            history = job["history_condition"]
            style = job["history_style"]
            depth = job["history_depth"]
            realization = job["history_realization"]
            messages = prefix_messages(role, history, style, depth, realization)
            prefix_ids = _exact_prefix(tokenizer, messages)
            rendered_prefix, suffix_ids = _split_final_user(
                tokenizer,
                _chat_ids(tokenizer, messages + [{"role": "user", "content": job["prompt"]["prompt"]}]),
            )
            if rendered_prefix != prefix_ids:
                raise RuntimeError(f"performance prefix mismatch: {current_cell}")
            order = list(METHODS[cell_index % len(METHODS):] + METHODS[:cell_index % len(METHODS)])
            for order_position, method in enumerate(order):
                manager = None
                if method != "baseline":
                    editors[method].to(device)
                    manager = editors[method].hook_manager().to(device)
                _synchronize()
                for _warmup in range(WARMUP_REPEATS):
                    logits = _forward_once(
                        method=method,
                        model=model,
                        device=device,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        label_ids=label_ids,
                        manager=manager,
                    )
                    if not all(math.isfinite(value) for value in logits):
                        raise FloatingPointError("non-finite warmup logits")
                for repeat in range(TIMED_REPEATS):
                    resident = _reset_peak(device)
                    start = time.perf_counter()
                    logits = _forward_once(
                        method=method,
                        model=model,
                        device=device,
                        prefix_ids=prefix_ids,
                        suffix_ids=suffix_ids,
                        label_ids=label_ids,
                        manager=manager,
                    )
                    _synchronize()
                    latency = time.perf_counter() - start
                    peak = int(torch.npu.max_memory_allocated(device))
                    current_key = measurement_key(current_cell, method, repeat)
                    values = (latency, 1.0 / latency, len(suffix_ids) / latency, *logits)
                    if latency <= 0 or peak <= 0 or not all(math.isfinite(value) for value in values):
                        raise FloatingPointError(f"non-finite performance measurement: {current_key}")
                    record = {
                        "measurement_key": current_key,
                        "cell_key": current_cell,
                        "cell_index": cell_index,
                        "method_order": order,
                        "method_order_position": order_position,
                        "method": method,
                        "repeat": repeat,
                        "benchmark": job["item"]["benchmark"],
                        "item_id": job["item"]["item_id"],
                        "operator_dev_job_key": "__".join(
                            str(value)
                            for value in (
                                job["item"]["item_id"], job["item"]["partition"], role,
                                style, history, depth, realization, job["task_requirement"],
                                job["label_swap"],
                            )
                        ),
                        "task_requirement": job["task_requirement"],
                        "label_swap": int(job["label_swap"]),
                        "declared_role": role,
                        "history_condition": history,
                        "history_style": style,
                        "history_depth": depth,
                        "history_realization": realization,
                        "prefix_tokens": len(prefix_ids),
                        "suffix_tokens": len(suffix_ids),
                        "prefix_token_sha256_int32_le": _token_hash(prefix_ids),
                        "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                        "paired_latency_seconds": latency,
                        "examples_per_second": 1.0 / latency,
                        "suffix_tokens_per_second": len(suffix_ids) / latency,
                        "resident_npu_memory_bytes": resident,
                        "peak_npu_memory_bytes": peak,
                        "incremental_peak_npu_memory_bytes": peak - resident,
                        "selected_label_logits": {"A": logits[0], "B": logits[1]},
                    }
                    handle.write(canonical_json(record) + "\n")
                    completed.add(current_key)
                if manager is not None:
                    del manager
                    editors[method].to("cpu")
                gc.collect()
                torch.npu.empty_cache()
                _synchronize()
    if len(completed) != EXPECTED_MEASUREMENTS:
        raise RuntimeError("performance measurement count mismatch")
    print(json.dumps({
        "authorization_id": authorization.get("authorization_id"),
        "completed_unique_measurements": len(completed),
        "expected_cells": EXPECTED_CELLS,
        "expected_cell_key_sha256": EXPECTED_CELL_KEY_SHA256,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
