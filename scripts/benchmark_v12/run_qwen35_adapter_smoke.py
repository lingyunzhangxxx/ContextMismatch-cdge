#!/usr/bin/env python3
"""Engineering-only Qwen3.5 adapter and exact no-op hook smoke."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v1.histories import prefix_messages
from scripts.benchmark_v1.run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _split_final_user,
    _suffix_logits_batch,
    _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import _layers, _load_model, _module
from scripts.benchmark_v1.run_protected_capture import _prefill_with_last_boundary_capture
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replication-protocol", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing smoke output: {args.output}")

    protocol = json.loads(args.replication_protocol.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    model_contract = json.loads(args.model_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    required_protocol = {
        "status": "frozen_before_any_qwen3_5_9b_cdge_forward",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_protocol.items():
        if protocol.get(field) != expected:
            raise ValueError(f"replication protocol mismatch: {field}")
    if (
        model_contract.get("contract") != "context-mismatch-qwen35-9b-bf16-v1"
        or model_contract.get("non_quantized") is not True
        or model_contract.get("production_rollout_approved") is not False
    ):
        raise ValueError("model contract is not the frozen non-quantized checkpoint")
    if (
        model_contract.get("lineage", {}).get("weight_and_config_file_revision")
        != crossover["base_model"]["revision"]
    ):
        raise ValueError("model revision mismatch")
    architecture = model_contract.get("architecture", {})
    for field, expected in {
        "class": "Qwen3_5ForConditionalGeneration",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
    }.items():
        if architecture.get(field) != expected:
            raise ValueError(f"model architecture mismatch: {field}")
    required_authorization = {
        "stage": "qwen35_cdge_adapter_smoke",
        "method": "C-DGE-V4.1",
        "execution_allowed": True,
        "replication_protocol_sha256": sha256_file(args.replication_protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_contract_sha256": sha256_file(args.model_contract),
        "expected_rows": 1,
        "engineering_only": True,
        "scientific_effect_inference_forbidden": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_authorization.items():
        if authorization.get(field) != expected:
            raise ValueError(f"adapter-smoke authorization mismatch: {field}")

    jobs = sorted(enumerate_jobs(load_jsonl(args.manifest), crossover, "replication"), key=job_key)
    job = jobs[0]
    messages = prefix_messages(
        job["declared_role"],
        job["history_condition"],
        job["history_style"],
        job["history_depth"],
        job["history_realization"],
    )
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if len(_layers(model)) != 32:
        raise RuntimeError("Qwen3.5 text-layer count mismatch")
    module = _module(model, 27, "mlp")
    label_ids = _label_ids(tokenizer)
    prefix_ids = _exact_prefix(tokenizer, messages)
    rendered_prefix, suffix_ids = _split_final_user(
        tokenizer,
        _chat_ids(tokenizer, messages + [{"role": "user", "content": job["prompt"]["prompt"]}]),
    )
    if rendered_prefix != prefix_ids:
        raise RuntimeError("adapter smoke prefix mismatch")
    cache, boundary = _prefill_with_last_boundary_capture(model, device, prefix_ids, [27])
    baseline_logits, baseline_extended = _suffix_logits_batch(
        model, device, cache, len(prefix_ids), [suffix_ids], label_ids, 0
    )
    del baseline_extended
    captures: list[tuple[int, ...]] = []

    def exact_noop(_module_value, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        captures.append(tuple(value.shape))
        return output

    handle = module.register_forward_hook(exact_noop)
    try:
        noop_logits, noop_extended = _suffix_logits_batch(
            model, device, cache, len(prefix_ids), [suffix_ids], label_ids, 0
        )
    finally:
        handle.remove()
    del noop_extended
    values = [*baseline_logits[0], *noop_logits[0]]
    if not all(math.isfinite(float(value)) for value in values):
        raise FloatingPointError("non-finite adapter-smoke logits")
    max_error = max(
        abs(float(baseline_logits[0][0]) - float(noop_logits[0][0])),
        abs(float(baseline_logits[0][1]) - float(noop_logits[0][1])),
    )
    if max_error != 0.0 or len(captures) != 1:
        raise RuntimeError("Qwen3.5 exact no-op hook identity failed")
    boundary_value = boundary.get("27")
    if boundary_value is None or tuple(boundary_value.shape) != (4096,):
        raise RuntimeError("Qwen3.5 boundary capture shape mismatch")

    report = {
        "schema_version": 1,
        "stage": "qwen35_cdge_adapter_smoke",
        "method": "C-DGE-V4.1",
        "engineering_only": True,
        "scientific_effect_inference_forbidden": True,
        "rows": 1,
        "job_key": job_key(job),
        "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
        "text_layers": len(_layers(model)),
        "site": "27:mlp",
        "boundary_shape": list(boundary_value.shape),
        "component_output_shapes": [list(shape) for shape in captures],
        "baseline_selected_logits": [float(value) for value in baseline_logits[0]],
        "noop_selected_logits": [float(value) for value in noop_logits[0]],
        "zero_hook_max_selected_logit_error": max_error,
        "success": True,
        "replication_protocol_sha256": sha256_file(args.replication_protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "model_contract_sha256": sha256_file(args.model_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
