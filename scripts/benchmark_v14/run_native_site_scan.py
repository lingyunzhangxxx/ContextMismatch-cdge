#!/usr/bin/env python3
"""Run a frozen Qwen3.5 model-native smoke, residual, or component scan."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v1.histories import pairwise_prompt, prefix_messages
from scripts.benchmark_v1.run_behavior import (
    _chat_ids, _device, _exact_prefix, _label_ids, _prefill, _split_final_user,
)
from scripts.benchmark_v1.run_mechanism_discovery import (
    _capture_suffix, _layers, _load_model, _margin, _patched_suffix,
)
from scripts.benchmark_v14.native_discovery import (
    load_protocol, scan_factorial_axes, scan_key, select_scan_items,
    smoke_scan_axes,
)


def _scan_configuration(args, protocol, plan):
    rule = protocol["native_discovery"]
    if args.mode == "smoke":
        return smoke_scan_axes(protocol, plan)
    if args.mode == "residual":
        return list(plan["residual_layers"]), ["residual"]
    if args.layer_manifest is None:
        raise ValueError("component scan requires --layer-manifest")
    layer_manifest = json.loads(args.layer_manifest.read_text())
    required = {
        "stage": "qwen35_cdge_v4_2_native_residual_nomination",
        "locked": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if layer_manifest.get(field) != expected:
            raise ValueError(f"layer manifest mismatch: {field}")
    if layer_manifest.get("scan_plan_sha256") != sha256_file(args.scan_plan):
        raise ValueError("layer manifest scan-plan binding mismatch")
    return [int(value) for value in layer_manifest["nominated_layers"]], list(rule["components"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("smoke", "residual", "component"), required=True
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scan-plan", type=Path, required=True)
    parser.add_argument("--layer-manifest", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-contract", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="npu:0")
    args = parser.parse_args()
    if args.output.exists() or args.environment_output.exists():
        raise FileExistsError("native scan output paths must be fresh")
    protocol = load_protocol(args.protocol)
    plan = json.loads(args.scan_plan.read_text())
    model_contract = json.loads(args.model_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    for field, expected in {
        "stage": f"qwen35_cdge_v4_2_native_{args.mode}_scan",
        "method": "C-DGE-V4.2",
        "execution_allowed": True,
        "protocol_sha256": sha256_file(args.protocol),
        "scan_plan_sha256": sha256_file(args.scan_plan),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_contract_sha256": sha256_file(args.model_contract),
        "scientific_efficacy_eligible": args.mode != "smoke",
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if authorization.get(field) != expected:
            raise ValueError(f"native scan authorization mismatch: {field}")
    if sha256_file(args.model_contract) != protocol["frozen_lineage"]["model_contract_sha256"]:
        raise ValueError("native scan model lineage mismatch")
    if model_contract["lineage"]["weight_and_config_file_revision"] != protocol["base_model"]["revision"]:
        raise ValueError("native scan model revision mismatch")
    layers, components = _scan_configuration(args, protocol, plan)
    if args.mode == "component" and authorization.get("layer_manifest_sha256") != sha256_file(args.layer_manifest):
        raise ValueError("component authorization layer-manifest mismatch")
    items = select_scan_items(load_jsonl(args.manifest), protocol, mode=args.mode)
    factorial = scan_factorial_axes(protocol, args.mode)
    roles = factorial["declared_roles"]
    styles = factorial["history_styles"]
    swaps = factorial["label_swaps"]
    regimes = factorial["source_regimes"]
    coefficients = protocol["native_discovery"]["patch_coefficients"]
    expected_keys = {
        scan_key(args.mode, item["item_id"], role, style, swap, source, layer, component, coefficient)
        for item in items for role in roles for style in styles for swap in swaps
        for source in regimes for layer in layers for component in components
        for coefficient in coefficients
    }
    expected_sha = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(expected_keys)).encode("utf-8")
    ).hexdigest()
    if authorization.get("expected_rows") != len(expected_keys) or authorization.get("expected_key_sha256") != expected_sha:
        raise ValueError("native scan authorization row/key contract mismatch")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, "eager")
    if len(_layers(model)) != int(plan["num_hidden_layers"]):
        raise ValueError("loaded model layer count differs from scan plan")
    label_ids = _label_ids(tokenizer)
    module_keys = [(layer, component) for layer in layers for component in components]
    environment = {
        "schema_version": 1,
        "stage": f"qwen35_cdge_v4_2_native_{args.mode}_scan",
        "mode": args.mode,
        "method": "C-DGE-V4.2",
        "protocol_sha256": sha256_file(args.protocol),
        "scan_plan_sha256": sha256_file(args.scan_plan),
        "layer_manifest_sha256": sha256_file(args.layer_manifest) if args.layer_manifest else None,
        "model_contract_sha256": sha256_file(args.model_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "layers": layers,
        "components": components,
        "factorial_axes": factorial,
        "planned_rows": len(expected_keys),
        "expected_key_sha256": expected_sha,
        "scientific_efficacy_eligible": args.mode != "smoke",
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n")
    grouped = defaultdict(list)
    for item in items:
        grouped[int(item["history_realization"])].append(item)
    observed = set()
    with args.output.open("x", buffering=1) as handle:
        for role in roles:
            for style in styles:
                for realization in sorted(grouped):
                    messages = {
                        regime: prefix_messages(role, regime, style, 32, realization)
                        for regime in regimes
                    }
                    prefix_ids = {regime: _exact_prefix(tokenizer, value) for regime, value in messages.items()}
                    caches = {regime: _prefill(model, device, value) for regime, value in prefix_ids.items()}
                    for item in sorted(grouped[realization], key=lambda row: row["item_id"]):
                        for swap in swaps:
                            prompt, correct, foil = pairwise_prompt(item, swap)
                            suffixes = {}
                            for regime in regimes:
                                rendered, suffix = _split_final_user(
                                    tokenizer,
                                    _chat_ids(tokenizer, messages[regime] + [{"role": "user", "content": prompt}]),
                                )
                                if rendered != prefix_ids[regime]:
                                    raise RuntimeError("native scan prefix mismatch")
                                suffixes[regime] = suffix
                            if suffixes[regimes[0]] != suffixes[regimes[1]]:
                                raise RuntimeError("native scan paired suffix mismatch")
                            suffix = suffixes[regimes[0]]
                            captures, baselines = {}, {}
                            for regime in regimes:
                                captured, logits, extended = _capture_suffix(
                                    model, device, copy.deepcopy(caches[regime]),
                                    len(prefix_ids[regime]), suffix, label_ids, module_keys,
                                )
                                captures[regime], baselines[regime] = captured, logits
                                del extended
                            for source in regimes:
                                target = regimes[1] if source == regimes[0] else regimes[0]
                                source_margin = _margin(baselines[source], correct)
                                target_margin = _margin(baselines[target], correct)
                                for module_key in module_keys:
                                    layer, component = module_key
                                    for coefficient in coefficients:
                                        key = scan_key(args.mode, item["item_id"], role, style, swap, source, layer, component, coefficient)
                                        logits, extended = _patched_suffix(
                                            model, device, copy.deepcopy(caches[source]),
                                            len(prefix_ids[source]), suffix, label_ids,
                                            module_key, captures[target][module_key], coefficient,
                                        )
                                        margin = _margin(logits, correct)
                                        row = {
                                            "job_key": key, "mode": args.mode,
                                            "benchmark": item["benchmark"], "item_id": item["item_id"],
                                            "partition": item["partition"], "declared_role": role,
                                            "history_style": style, "history_realization": realization,
                                            "label_swap": swap, "correct_label": correct, "foil_label": foil,
                                            "source_regime": source, "target_regime": target,
                                            "layer": layer, "component": component,
                                            "patch_coefficient": coefficient,
                                            "logit_a": logits[0], "logit_b": logits[1],
                                            "correct_logit_margin": margin,
                                            "source_cached_margin": source_margin,
                                            "target_cached_margin": target_margin,
                                            "patch_effect": margin - source_margin,
                                            "target_gap": target_margin - source_margin,
                                        }
                                        if not all(math.isfinite(float(row[name])) for name in (
                                            "logit_a", "logit_b", "correct_logit_margin",
                                            "source_cached_margin", "target_cached_margin",
                                            "patch_effect", "target_gap",
                                        )):
                                            raise FloatingPointError(f"non-finite native scan row: {key}")
                                        handle.write(canonical_json(row) + "\n")
                                        observed.add(key)
                                        del extended
                            del captures, baselines
                            gc.collect()
                    del caches
                    gc.collect()
                    if device.type == "npu":
                        torch.npu.empty_cache()
    if observed != expected_keys:
        raise RuntimeError("native scan key set mismatch")
    print(json.dumps({"rows": len(observed), "key_sha256": expected_sha, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
