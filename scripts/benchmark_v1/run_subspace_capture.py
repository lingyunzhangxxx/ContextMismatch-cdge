#!/usr/bin/env python3
"""Capture split-isolated Qwen3-8B boundary and component states for operator fitting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

from .common import atomic_write_text, load_jsonl, sha256_file
from .histories import pairwise_prompt, prefix_messages
from .run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _prefill,
    _split_final_user,
    _suffix_logits_batch,
    _token_hash,
)
from .run_mechanism_discovery import _layers, _load_model, _module, _normalize_output, _select_items


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assistant_endpoint_positions(tokenizer, messages: list[dict], depth: int) -> list[int]:
    positions = []
    full_prefix = _exact_prefix(tokenizer, messages)
    for turn in range(1, depth + 1):
        partial = messages[: 1 + 2 * turn]
        partial_ids = _exact_prefix(tokenizer, partial)
        if full_prefix[: len(partial_ids)] != partial_ids:
            raise RuntimeError(f"turn {turn} tokenization is not a prefix of the full history")
        positions.append(len(partial_ids) - 1)
    return positions


@torch.inference_mode()
def _prefill_with_boundary_capture(
    model,
    device: torch.device,
    prefix_ids: list[int],
    layers: list[int],
    endpoint_positions: list[int],
):
    captures = {}
    handles = []

    def make_hook(layer):
        def hook(_module_value, _inputs, output):
            value = _normalize_output(output)
            captures[layer] = (
                value[0, endpoint_positions]
                .detach()
                .to(device="cpu", dtype=torch.float16)
                .contiguous()
            )

        return hook

    for layer in layers:
        handles.append(_layers(model)[layer].register_forward_hook(make_hook(layer)))
    try:
        cache = _prefill(model, device, prefix_ids)
        if set(captures) != set(layers):
            raise RuntimeError(f"missing boundary captures: {sorted(set(layers) - set(captures))}")
        return cache, captures
    finally:
        for handle in handles:
            handle.remove()


def _module_hidden_input(module_inputs: tuple, module_kwargs: dict) -> torch.Tensor:
    """Resolve component input across positional and keyword-only forwards."""
    if module_inputs:
        value = module_inputs[0]
    else:
        value = module_kwargs.get("hidden_states")
    if not isinstance(value, torch.Tensor):
        raise RuntimeError("component forward did not expose a hidden-state tensor")
    return value


@torch.inference_mode()
def _suffix_component_capture(
    model,
    device: torch.device,
    cache,
    prefix_length: int,
    suffix_ids: list[int],
    label_ids: dict[str, int],
    module_keys: list[tuple[int, str]],
):
    inputs = {}
    outputs = {}
    handles = []

    def make_pre_hook(key):
        def hook(_module_value, module_inputs, module_kwargs):
            value = _module_hidden_input(module_inputs, module_kwargs)
            inputs[key] = value[0, -1].detach().to(device="cpu", dtype=torch.float16).contiguous()

        return hook

    def make_output_hook(key):
        def hook(_module_value, _inputs, output):
            value = _normalize_output(output)
            outputs[key] = value[0, -1].detach().to(device="cpu", dtype=torch.float16).contiguous()

        return hook

    for key in module_keys:
        module = _module(model, *key)
        handles.append(module.register_forward_pre_hook(make_pre_hook(key), with_kwargs=True))
        handles.append(module.register_forward_hook(make_output_hook(key)))
    try:
        logits, extended_cache = _suffix_logits_batch(
            model,
            device,
            cache,
            prefix_length,
            [suffix_ids],
            label_ids,
            pad_token_id=0,
        )
        expected = set(module_keys)
        if set(inputs) != expected or set(outputs) != expected:
            raise RuntimeError("component input/output capture is incomplete")
        return inputs, outputs, logits[0], extended_cache
    finally:
        for handle in handles:
            handle.remove()


def _margin(logits: tuple[float, float], correct_label: str) -> float:
    values = {"A": logits[0], "B": logits[1]}
    foil = "B" if correct_label == "A" else "A"
    return values[correct_label] - values[foil]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--component-manifest", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--behavior-analysis", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()

    mechanism = json.loads(args.mechanism_contract.read_text())
    operator_contract = json.loads(args.operator_contract.read_text())
    execution_contract = json.loads(args.execution_contract.read_text())
    component_manifest = json.loads(args.component_manifest.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    behavior_analysis = json.loads(args.behavior_analysis.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if sha256_file(args.mechanism_contract) != operator_contract["mechanism_contract_sha256"]:
        raise ValueError("operator contract does not bind this mechanism contract")
    if sha256_file(args.manifest) != operator_contract["benchmark_manifest_sha256"]:
        raise ValueError("operator contract benchmark manifest mismatch")
    if manifest_report["manifest_sha256"] != operator_contract["benchmark_manifest_sha256"]:
        raise ValueError("manifest report mismatch")
    if not behavior_analysis.get("audit", {}).get("success"):
        raise ValueError("full behavior analysis has not passed")
    if not component_manifest.get("locked"):
        raise ValueError("component manifest is not locked")
    if component_manifest["mechanism_contract_sha256"] != sha256_file(args.mechanism_contract):
        raise ValueError("component manifest mechanism contract mismatch")
    if execution_contract["mechanism_contract_sha256"] != sha256_file(args.mechanism_contract):
        raise ValueError("execution contract mechanism binding mismatch")
    if execution_contract["operator_contract_sha256"] != sha256_file(args.operator_contract):
        raise ValueError("execution contract operator binding mismatch")
    if not site_manifest.get("locked") or site_manifest.get("selection_partition") != "component_discovery":
        raise ValueError("operator sites are not locked from component_discovery")
    if site_manifest["operator_site_execution_contract_sha256"] != sha256_file(
        args.execution_contract
    ):
        raise ValueError("operator site execution binding mismatch")
    if site_manifest["component_layer_manifest_sha256"] != sha256_file(
        args.component_manifest
    ):
        raise ValueError("operator site/component layer binding mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest["revision"] != operator_contract["base_model_revision"]:
        raise ValueError("model revision mismatch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing capture directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = load_jsonl(args.manifest)
    partition = "subspace_fit"
    items = _select_items(manifest, partition, 32)
    roles = mechanism["factorial"]["declared_roles"]
    styles = mechanism["factorial"]["history_styles"]
    regimes = mechanism["factorial"]["source_regimes"]
    label_swaps = mechanism["factorial"]["label_swaps"]
    depth = mechanism["factorial"]["history_depth"]
    selected_sites = list(site_manifest["selected_sites_ordered"])
    layers = [int(value["layer"]) for value in selected_sites]
    module_keys = [(int(value["layer"]), str(value["component"])) for value in selected_sites]
    if len(layers) != 3 or len(set(layers)) != 3 or len(set(module_keys)) != 3:
        raise ValueError("subspace capture requires exactly three locked component sites")
    if set(layers) != {int(value) for value in component_manifest["nominated_layers"]}:
        raise ValueError("operator sites do not cover the residual-nominated layers")
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)

    grouped_items = defaultdict(list)
    for item in items:
        grouped_items[int(item["history_realization"])].append(item)
    expected_rows = len(items) * len(roles) * len(styles) * len(regimes) * len(label_swaps)
    shard_records = []
    total_rows = 0
    total_boundary_rows = 0
    for role in roles:
        for style in styles:
            for realization in sorted(grouped_items):
                shard_name = f"capture__{role}__{style}__r{realization}.pt"
                shard_path = args.output_dir / shard_name
                shard_incoming = args.output_dir / f".{shard_name}.incoming"
                if shard_path.exists() or shard_incoming.exists():
                    raise FileExistsError(f"refusing existing shard target: {shard_path}")
                messages_by_regime = {
                    regime: prefix_messages(role, regime, style, depth, realization)
                    for regime in regimes
                }
                prefix_ids = {
                    regime: _exact_prefix(tokenizer, messages)
                    for regime, messages in messages_by_regime.items()
                }
                boundary_captures = {}
                base_caches = {}
                boundary_metadata = []
                for regime in regimes:
                    endpoint_positions = _assistant_endpoint_positions(
                        tokenizer, messages_by_regime[regime], depth
                    )
                    cache, captured = _prefill_with_boundary_capture(
                        model,
                        device,
                        prefix_ids[regime],
                        layers,
                        endpoint_positions,
                    )
                    base_caches[regime] = cache
                    boundary_captures[regime] = captured
                    boundary_metadata.extend(
                        {
                            "declared_role": role,
                            "history_style": style,
                            "history_realization": realization,
                            "regime": regime,
                            "turn": turn,
                            "prefix_position": position,
                        }
                        for turn, position in enumerate(endpoint_positions, 1)
                    )
                boundary_states = {}
                for layer in layers:
                    boundary_states[str(layer)] = torch.cat(
                        [boundary_captures[regime][layer] for regime in regimes], dim=0
                    )

                metadata = []
                input_values = {f"{layer}:{component}": [] for layer, component in module_keys}
                output_values = {f"{layer}:{component}": [] for layer, component in module_keys}
                for item in sorted(grouped_items[realization], key=lambda row: row["item_id"]):
                    for label_swap in label_swaps:
                        prompt, correct_label, foil_label = pairwise_prompt(item, label_swap)
                        suffixes = {}
                        for regime in regimes:
                            rendered_prefix, suffix_ids = _split_final_user(
                                tokenizer,
                                _chat_ids(
                                    tokenizer,
                                    messages_by_regime[regime]
                                    + [{"role": "user", "content": prompt}],
                                ),
                            )
                            if rendered_prefix != prefix_ids[regime]:
                                raise RuntimeError("capture prefix mismatch")
                            suffixes[regime] = suffix_ids
                        if suffixes[regimes[0]] != suffixes[regimes[1]]:
                            raise RuntimeError("capture paired suffix mismatch")
                        suffix_ids = suffixes[regimes[0]]
                        for regime in regimes:
                            cache = copy.deepcopy(base_caches[regime])
                            inputs, outputs, logits, extended_cache = _suffix_component_capture(
                                model,
                                device,
                                cache,
                                len(prefix_ids[regime]),
                                suffix_ids,
                                label_ids,
                                module_keys,
                            )
                            row = {
                                "benchmark": item["benchmark"],
                                "item_id": item["item_id"],
                                "partition": item["partition"],
                                "declared_role": role,
                                "history_style": style,
                                "history_realization": realization,
                                "regime": regime,
                                "label_swap": label_swap,
                                "correct_label": correct_label,
                                "foil_label": foil_label,
                                "logit_a": logits[0],
                                "logit_b": logits[1],
                                "correct_logit_margin": _margin(logits, correct_label),
                                "prefix_tokens": len(prefix_ids[regime]),
                                "suffix_tokens": len(suffix_ids),
                                "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                            }
                            if not all(math.isfinite(row[key]) for key in ("logit_a", "logit_b", "correct_logit_margin")):
                                raise FloatingPointError("non-finite capture baseline")
                            metadata.append(row)
                            for layer, component in module_keys:
                                key = f"{layer}:{component}"
                                input_values[key].append(inputs[(layer, component)])
                                output_values[key].append(outputs[(layer, component)])
                            del cache, extended_cache
                shard = {
                    "schema_version": 1,
                    "metadata": metadata,
                    "boundary_metadata": boundary_metadata,
                    "boundary_states": boundary_states,
                    "component_inputs": {
                        key: torch.stack(values, dim=0) for key, values in input_values.items()
                    },
                    "component_outputs": {
                        key: torch.stack(values, dim=0) for key, values in output_values.items()
                    },
                }
                torch.save(shard, shard_incoming)
                os.replace(shard_incoming, shard_path)
                shard_records.append(
                    {
                        "file": shard_name,
                        "sha256": _sha256(shard_path),
                        "rows": len(metadata),
                        "boundary_rows": len(boundary_metadata),
                        "role": role,
                        "style": style,
                        "realization": realization,
                    }
                )
                total_rows += len(metadata)
                total_boundary_rows += len(boundary_metadata)
                del shard, base_caches, boundary_captures
                if device.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.empty_cache()
    expected_boundary_rows = len(roles) * len(styles) * len(grouped_items) * len(regimes) * depth
    if total_rows != expected_rows:
        raise RuntimeError(f"capture row mismatch: {total_rows} != {expected_rows}")
    if total_boundary_rows != expected_boundary_rows:
        raise RuntimeError(
            f"boundary capture row mismatch: {total_boundary_rows} != {expected_boundary_rows}"
        )
    capture_manifest = {
        "schema_version": 1,
        "partition": partition,
        "rows": total_rows,
        "boundary_rows": total_boundary_rows,
        "layers": layers,
        "sites": selected_sites,
        "hidden_size": mechanism["base_model"]["hidden_size"],
        "dtype": "float16_capture_from_bfloat16_forward",
        "mechanism_contract_sha256": sha256_file(args.mechanism_contract),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "operator_site_execution_contract_sha256": sha256_file(args.execution_contract),
        "component_manifest_sha256": sha256_file(args.component_manifest),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "behavior_analysis_sha256": sha256_file(args.behavior_analysis),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "shards": shard_records,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "capture_manifest.json",
        json.dumps(capture_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(capture_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
