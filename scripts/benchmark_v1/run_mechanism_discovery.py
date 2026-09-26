#!/usr/bin/env python3
"""Exact paired Qwen3-8B residual/component patching on the scalar cache path."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import (
    canonical_json,
    effective_mechanism_items_per_benchmark,
    load_jsonl,
    sha256_file,
)
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


def _normalize_output(output):
    return output[0] if isinstance(output, tuple) else output


def _replace_output(output, value):
    return (value,) + output[1:] if isinstance(output, tuple) else value


def _layers(model):
    root = getattr(model, "model", None)
    if root is not None and hasattr(root, "layers"):
        return root.layers
    language_model = getattr(root, "language_model", None)
    if language_model is not None and hasattr(language_model, "layers"):
        return language_model.layers
    raise TypeError(f"unsupported model structure: {type(model).__name__}")


def _module(model, layer: int, component: str):
    block = _layers(model)[layer]
    if component == "residual":
        return block
    if component == "self_attn":
        if hasattr(block, "self_attn"):
            return block.self_attn
        if hasattr(block, "linear_attn"):
            return block.linear_attn
        raise TypeError(f"layer {layer} has no supported attention/mixer module")
    if component == "mixer":
        if hasattr(block, "self_attn"):
            return block.self_attn
        if hasattr(block, "linear_attn"):
            return block.linear_attn
        raise TypeError(f"layer {layer} has no supported attention/mixer module")
    if component == "mlp":
        return block.mlp
    raise ValueError(component)


@torch.inference_mode()
def _capture_suffix(
    model,
    device: torch.device,
    cache,
    prefix_length: int,
    suffix_ids: list[int],
    label_ids: dict[str, int],
    module_keys: list[tuple[int, str]],
):
    captures = {}
    handles = []

    def make_hook(key):
        def hook(_module_value, _inputs, output):
            value = _normalize_output(output)
            if value.ndim != 3 or value.shape[0] != 1:
                raise RuntimeError(f"unexpected activation shape for {key}: {tuple(value.shape)}")
            captures[key] = value.detach().clone()

        return hook

    for key in module_keys:
        handles.append(_module(model, *key).register_forward_hook(make_hook(key)))
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
        if set(captures) != set(module_keys):
            raise RuntimeError(f"missing captures: {sorted(set(module_keys) - set(captures))}")
        return captures, logits[0], extended_cache
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def _patched_suffix(
    model,
    device: torch.device,
    cache,
    prefix_length: int,
    suffix_ids: list[int],
    label_ids: dict[str, int],
    module_key: tuple[int, str],
    target: torch.Tensor,
    coefficient: float,
):
    module = _module(model, *module_key)

    def hook(_module_value, _inputs, output):
        value = _normalize_output(output)
        if value.shape != target.shape:
            raise RuntimeError(
                f"paired activation shape mismatch at {module_key}: "
                f"source={tuple(value.shape)} target={tuple(target.shape)}"
            )
        if coefficient == 0.0:
            return output
        paired_target = target.to(device=value.device, dtype=value.dtype)
        edited = paired_target if coefficient == 1.0 else torch.lerp(value, paired_target, coefficient)
        return _replace_output(output, edited)

    handle = module.register_forward_hook(hook)
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
        return logits[0], extended_cache
    finally:
        handle.remove()


def _margin(logits: tuple[float, float], correct_label: str) -> float:
    values = {"A": logits[0], "B": logits[1]}
    foil_label = "B" if correct_label == "A" else "A"
    return values[correct_label] - values[foil_label]


def _completed_keys(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {row["job_key"] for row in load_jsonl(path)}


def _job_key(
    mode: str,
    item_id: str,
    role: str,
    style: str,
    label_swap: int,
    source_regime: str,
    layer: int,
    component: str,
    coefficient: float,
) -> str:
    return "__".join(
        map(
            str,
            (
                mode,
                item_id,
                role,
                style,
                label_swap,
                source_regime,
                layer,
                component,
                coefficient,
            ),
        )
    )


def _select_items(manifest: list[dict], partition: str, items_per_benchmark: int) -> list[dict]:
    selected = []
    benchmarks = sorted({row["benchmark"] for row in manifest})
    for benchmark in benchmarks:
        candidates = sorted(
            (
                row
                for row in manifest
                if row["benchmark"] == benchmark and row["partition"] == partition
            ),
            key=lambda row: row["item_id"],
        )
        if len(candidates) < items_per_benchmark:
            raise ValueError(f"insufficient {partition} items for {benchmark}")
        selected.extend(candidates[:items_per_benchmark])
    return selected


def _load_model(model_path: Path, device: torch.device, attn_implementation: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    kwargs = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "attn_implementation": attn_implementation,
    }
    config_path = model_path / "config.json"
    architectures = []
    if config_path.is_file():
        architectures = json.loads(config_path.read_text()).get("architectures", [])
    model_class = AutoModelForCausalLM
    if "Qwen3_5ForConditionalGeneration" in architectures:
        try:
            from transformers import Qwen3_5ForConditionalGeneration
        except ImportError as exc:
            raise RuntimeError(
                "the installed transformers runtime does not provide "
                "Qwen3_5ForConditionalGeneration"
            ) from exc
        model_class = Qwen3_5ForConditionalGeneration
    try:
        model = model_class.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs["torch_dtype"] = kwargs.pop("dtype")
        model = model_class.from_pretrained(model_path, **kwargs)
    model = model.to(device)
    model.eval()
    return tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--mechanism-erratum", type=Path, required=True)
    parser.add_argument("--behavior-contract", type=Path, required=True)
    parser.add_argument("--behavior-analysis", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mode", choices=["residual", "component"], required=True)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--component-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    mechanism = json.loads(args.mechanism_contract.read_text())
    mechanism_erratum = json.loads(args.mechanism_erratum.read_text())
    mechanism_contract_sha256 = sha256_file(args.mechanism_contract)
    effective_items_per_benchmark = effective_mechanism_items_per_benchmark(
        mechanism,
        mechanism_erratum,
        mechanism_contract_sha256,
    )
    behavior_contract = json.loads(args.behavior_contract.read_text())
    behavior_analysis = json.loads(args.behavior_analysis.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    lineage = mechanism["lineage"]
    if sha256_file(args.behavior_contract) != lineage["behavior_contract_sha256"]:
        raise ValueError("behavior contract SHA mismatch")
    if sha256_file(args.manifest) != lineage["benchmark_manifest_sha256"]:
        raise ValueError("benchmark manifest SHA mismatch")
    if manifest_report["manifest_sha256"] != lineage["benchmark_manifest_sha256"]:
        raise ValueError("manifest report does not certify the frozen manifest")
    if not behavior_analysis.get("audit", {}).get("success"):
        raise ValueError("full behavior gate has not passed")
    if behavior_analysis["audit"]["row_count"] != lineage["required_behavior_rows"]:
        raise ValueError("full behavior row gate failed")
    if behavior_analysis["mismatch_obedience_minus_verification"]["n_pairs"] != lineage["required_behavior_mismatch_pairs"]:
        raise ValueError("full behavior mismatch-pair gate failed")
    if behavior_analysis["reset_minus_obedience"]["n_pairs"] != lineage["required_behavior_reset_pairs"]:
        raise ValueError("full behavior reset-pair gate failed")
    verified = bool(model_manifest.get("verified") or model_manifest.get("success"))
    non_quantized = bool(
        model_manifest.get("non_quantized") or model_manifest.get("quantization") == "none"
    )
    if not verified or not non_quantized:
        raise ValueError("model manifest is unverified or quantized")
    if model_manifest["revision"] != mechanism["base_model"]["revision"]:
        raise ValueError("model revision mismatch")

    manifest = load_jsonl(args.manifest)
    per_benchmark = 1 if args.stage == "smoke" else effective_items_per_benchmark
    items = _select_items(manifest, mechanism["data_split"]["partition"], per_benchmark)
    if args.stage == "smoke":
        roles = ["collaborator"]
        styles = ["lexical_matched"]
    else:
        roles = mechanism["factorial"]["declared_roles"]
        styles = mechanism["factorial"]["history_styles"]
    label_swaps = mechanism["factorial"]["label_swaps"]
    regimes = mechanism["factorial"]["source_regimes"]
    coefficients = mechanism[f"{args.mode}_scan"]["coefficients"]
    if args.mode == "residual":
        layers = mechanism["residual_scan"]["layers"]
        components = ["residual"]
    else:
        if args.component_manifest is None:
            raise ValueError("component mode requires --component-manifest")
        component_manifest = json.loads(args.component_manifest.read_text())
        if not component_manifest.get("locked"):
            raise ValueError("component manifest is not locked")
        if component_manifest["mechanism_contract_sha256"] != sha256_file(args.mechanism_contract):
            raise ValueError("component manifest contract SHA mismatch")
        if component_manifest.get("mechanism_contract_erratum_sha256") != sha256_file(
            args.mechanism_erratum
        ):
            raise ValueError("component manifest contract erratum SHA mismatch")
        layers = component_manifest["nominated_layers"]
        components = mechanism["component_scan"]["components"]
    expected_rows = (
        len(items)
        * len(roles)
        * len(styles)
        * len(label_swaps)
        * len(regimes)
        * len(layers)
        * len(components)
        * len(coefficients)
    )
    if args.stage == "full":
        frozen_expected = mechanism[f"{args.mode}_scan"]["expected_full_rows"]
        if expected_rows != frozen_expected:
            raise ValueError(f"expected-row derivation changed: {expected_rows} != {frozen_expected}")
    expected_job_keys = {
        _job_key(
            args.mode,
            item["item_id"],
            role,
            style,
            label_swap,
            source_regime,
            layer,
            component,
            coefficient,
        )
        for item in items
        for role in roles
        for style in styles
        for label_swap in label_swaps
        for source_regime in regimes
        for layer in layers
        for component in components
        for coefficient in coefficients
    }
    expected_key_sha256 = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(expected_job_keys)).encode("utf-8")
    ).hexdigest()
    if len(expected_job_keys) != expected_rows:
        raise ValueError("mechanism key construction produced a collision")

    completed = _completed_keys(args.output)
    if args.output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.environment_output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.environment_output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    if len(_layers(model)) != mechanism["base_model"]["num_hidden_layers"]:
        raise ValueError("unexpected model layer count")
    label_ids = _label_ids(tokenizer)
    module_keys = [(layer, component) for layer in layers for component in components]
    environment = {
        "mode": args.mode,
        "stage": args.stage,
        "device": str(device),
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "model_revision": mechanism["base_model"]["revision"],
        "mechanism_contract_sha256": mechanism_contract_sha256,
        "mechanism_contract_erratum_sha256": sha256_file(args.mechanism_erratum),
        "effective_items_per_benchmark": per_benchmark,
        "behavior_analysis_sha256": sha256_file(args.behavior_analysis),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "component_manifest_sha256": sha256_file(args.component_manifest) if args.component_manifest else None,
        "label_token_ids": label_ids,
        "layers": layers,
        "components": components,
        "coefficients": coefficients,
        "planned_rows": expected_rows,
        "expected_job_key_set_sha256": expected_key_sha256,
        "batch_size": 1,
        "patch_scope": "all_new_task_suffix_tokens",
        "production_rollout_approved": False,
    }
    if not args.environment_output.exists():
        args.environment_output.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    grouped_items = defaultdict(list)
    for item in items:
        grouped_items[item["history_realization"]].append(item)
    output_handle = args.output.open("a", buffering=1)
    for role in roles:
        for style in styles:
            for realization in sorted(grouped_items):
                messages_by_regime = {
                    regime: prefix_messages(
                        role,
                        regime,
                        style,
                        mechanism["factorial"]["history_depth"],
                        realization,
                    )
                    for regime in regimes
                }
                prefix_ids = {
                    regime: _exact_prefix(tokenizer, messages)
                    for regime, messages in messages_by_regime.items()
                }
                base_caches = {
                    regime: _prefill(model, device, ids)
                    for regime, ids in prefix_ids.items()
                }
                for item in sorted(grouped_items[realization], key=lambda row: row["item_id"]):
                    for label_swap in label_swaps:
                        case_keys = [
                            _job_key(
                                args.mode,
                                item["item_id"],
                                role,
                                style,
                                label_swap,
                                source_regime,
                                layer,
                                component,
                                coefficient,
                            )
                            for source_regime in regimes
                            for layer in layers
                            for component in components
                            for coefficient in coefficients
                        ]
                        if all(key in completed for key in case_keys):
                            continue
                        prompt, correct_label, foil_label = pairwise_prompt(item, label_swap)
                        suffix_by_regime = {}
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
                                raise RuntimeError("rendered prefix differs from cached prefix")
                            suffix_by_regime[regime] = suffix_ids
                        if suffix_by_regime[regimes[0]] != suffix_by_regime[regimes[1]]:
                            raise RuntimeError("paired regimes produced different task suffixes")
                        suffix_ids = suffix_by_regime[regimes[0]]
                        captures = {}
                        baselines = {}
                        for regime in regimes:
                            cache = copy.deepcopy(base_caches[regime])
                            captured, logits, extended_cache = _capture_suffix(
                                model,
                                device,
                                cache,
                                len(prefix_ids[regime]),
                                suffix_ids,
                                label_ids,
                                module_keys,
                            )
                            captures[regime] = captured
                            baselines[regime] = logits
                            del cache, extended_cache
                        for source_regime in regimes:
                            target_regime = "obedience" if source_regime == "verification" else "verification"
                            source_margin = _margin(baselines[source_regime], correct_label)
                            target_margin = _margin(baselines[target_regime], correct_label)
                            for layer in layers:
                                for component in components:
                                    module_key = (layer, component)
                                    for coefficient in coefficients:
                                        key = _job_key(
                                            args.mode,
                                            item["item_id"],
                                            role,
                                            style,
                                            label_swap,
                                            source_regime,
                                            layer,
                                            component,
                                            coefficient,
                                        )
                                        if key in completed:
                                            continue
                                        cache = copy.deepcopy(base_caches[source_regime])
                                        started = time.time()
                                        logits, extended_cache = _patched_suffix(
                                            model,
                                            device,
                                            cache,
                                            len(prefix_ids[source_regime]),
                                            suffix_ids,
                                            label_ids,
                                            module_key,
                                            captures[target_regime][module_key],
                                            coefficient,
                                        )
                                        patched_margin = _margin(logits, correct_label)
                                        record = {
                                            "job_key": key,
                                            "mode": args.mode,
                                            "stage": args.stage,
                                            "benchmark": item["benchmark"],
                                            "item_id": item["item_id"],
                                            "partition": item["partition"],
                                            "declared_role": role,
                                            "history_style": style,
                                            "history_depth": mechanism["factorial"]["history_depth"],
                                            "history_realization": realization,
                                            "label_swap": label_swap,
                                            "correct_label": correct_label,
                                            "foil_label": foil_label,
                                            "source_regime": source_regime,
                                            "target_regime": target_regime,
                                            "layer": layer,
                                            "component": component,
                                            "patch_coefficient": coefficient,
                                            "logit_a": logits[0],
                                            "logit_b": logits[1],
                                            "correct_logit_margin": patched_margin,
                                            "source_cached_logit_a": baselines[source_regime][0],
                                            "source_cached_logit_b": baselines[source_regime][1],
                                            "source_cached_margin": source_margin,
                                            "target_cached_logit_a": baselines[target_regime][0],
                                            "target_cached_logit_b": baselines[target_regime][1],
                                            "target_cached_margin": target_margin,
                                            "patch_effect": patched_margin - source_margin,
                                            "target_gap": target_margin - source_margin,
                                            "prefix_tokens": len(prefix_ids[source_regime]),
                                            "target_prefix_tokens": len(prefix_ids[target_regime]),
                                            "suffix_tokens": len(suffix_ids),
                                            "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                                            "latency_s": round(time.time() - started, 4),
                                            "model_revision": mechanism["base_model"]["revision"],
                                        }
                                        if not all(
                                            math.isfinite(record[name])
                                            for name in (
                                                "logit_a",
                                                "logit_b",
                                                "correct_logit_margin",
                                                "source_cached_margin",
                                                "target_cached_margin",
                                                "patch_effect",
                                                "target_gap",
                                            )
                                        ):
                                            raise FloatingPointError(f"non-finite mechanism row: {key}")
                                        output_handle.write(canonical_json(record) + "\n")
                                        completed.add(key)
                                        del cache, extended_cache
                        del captures, baselines
                        gc.collect()
                del base_caches
                gc.collect()
                if device.type == "npu" and hasattr(torch, "npu"):
                    torch.npu.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()
    output_handle.close()
    if completed != expected_job_keys:
        raise RuntimeError(
            "mechanism job key set mismatch: "
            f"missing={len(expected_job_keys - completed)} "
            f"unexpected={len(completed - expected_job_keys)}"
        )
    print(json.dumps({"planned_rows": expected_rows, "completed_rows": len(completed), "output": str(args.output)}))


if __name__ == "__main__":
    main()
