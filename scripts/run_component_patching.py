#!/usr/bin/env python3
"""Exact paired attention/MLP output patching on the cached-suffix path."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import time
from pathlib import Path

import torch

from run_length_scan import SYSTEM, TASKS, final_prompt, make_history
from run_mechanistic import (
    chat_token_ids,
    decision_record,
    exact_history_prefix_ids,
    input_device,
    label_token_ids,
    load_model,
    prefill_history,
    split_before_final_user,
    tensor_logits,
)


def component_module(model, layer_index, component):
    layer = model.model.language_model.layers[layer_index]
    if component == "mixer":
        if hasattr(layer, "self_attn"):
            return layer.self_attn
        return layer.linear_attn
    if component == "mlp":
        return layer.mlp
    raise ValueError(component)


def normalize_output(output):
    return output[0] if isinstance(output, tuple) else output


@torch.inference_mode()
def suffix_with_component_capture(
    model, suffix_ids, prefix_length, cache, token_ids, layer_indices, components,
):
    captures = {}
    handles = []

    def make_hook(key):
        def hook(_module, _inputs, output):
            value = normalize_output(output)
            captures[key] = (
                value[0, -1].detach().to(device="cpu", dtype=torch.float16).numpy()
            )
        return hook

    for layer_index in layer_indices:
        for component in components:
            key = (layer_index, component)
            handles.append(
                component_module(model, layer_index, component).register_forward_hook(
                    make_hook(key)
                )
            )
    try:
        device = input_device(model)
        ids = torch.tensor([suffix_ids], device=device)
        positions = torch.arange(
            prefix_length, prefix_length + len(suffix_ids), device=device
        ).unsqueeze(0)
        output = model(
            input_ids=ids,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        expected = {
            (layer_index, component)
            for layer_index in layer_indices for component in components
        }
        if set(captures) != expected:
            raise RuntimeError(
                f"component hooks missing: {sorted(expected - set(captures))}"
            )
        logit_a, logit_b = tensor_logits(model, output, token_ids)
        return captures, logit_a, logit_b
    finally:
        for handle in handles:
            handle.remove()


@torch.inference_mode()
def suffix_with_component_patch(
    model, suffix_ids, prefix_length, cache, token_ids, layer_index, component,
    target_activation, coefficient,
):
    module = component_module(model, layer_index, component)

    def hook(_module, _inputs, output):
        value = normalize_output(output)
        edited = value.clone()
        target = torch.as_tensor(
            target_activation, device=edited.device, dtype=edited.dtype
        )
        edited[:, -1, :].lerp_(target, coefficient)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    handle = module.register_forward_hook(hook)
    try:
        device = input_device(model)
        ids = torch.tensor([suffix_ids], device=device)
        positions = torch.arange(
            prefix_length, prefix_length + len(suffix_ids), device=device
        ).unsqueeze(0)
        output = model(
            input_ids=ids,
            position_ids=positions,
            past_key_values=cache,
            use_cache=True,
            logits_to_keep=1,
        )
        return tensor_logits(model, output, token_ids)
    finally:
        handle.remove()


def load_completed(path):
    completed = set()
    if not path.exists():
        return completed
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((
            row["regime"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"], row["layer"],
            row["component"], row["patch_coefficient"],
        ))
    return completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--components", nargs="+", choices=["mixer", "mlp"], default=["mixer", "mlp"])
    parser.add_argument("--coefficients", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--label-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--order-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--history-style", default="lexical_matched", choices=["lexical_matched"])
    parser.add_argument("--gpu-memory-gib", type=int, default=21)
    parser.add_argument("--cpu-memory-gib", type=int, default=80)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model(args)
    token_ids = label_token_ids(tokenizer)
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output) if args.resume else set()
    print(f"resume component patching: {len(completed)} rows", flush=True)
    output_handle = args.output.open("a" if args.resume else "w", buffering=1)

    for realization in range(args.realizations):
        prefixes = {}
        prefix_ids = {}
        base_caches = {}
        for regime in ("verification", "obedience"):
            messages = [{"role": "system", "content": SYSTEM}]
            messages.extend(
                make_history(args.depth, realization, regime, args.history_style)
            )
            prefixes[regime] = messages
            prefix_ids[regime] = exact_history_prefix_ids(tokenizer, messages)
            base_caches[regime] = prefill_history(model, prefix_ids[regime])
        if len(prefix_ids["verification"]) != len(prefix_ids["obedience"]):
            raise RuntimeError("component patching requires token-length matched histories")
        target_captures = {}

        for regime in ("verification", "obedience"):
            target_regime = "obedience" if regime == "verification" else "verification"
            for task in tasks:
                for label_swap in args.label_swaps:
                    for order_swap in args.order_swaps:
                        prompt, correct_label, foil_label = final_prompt(
                            task, label_swap, order_swap
                        )
                        messages = prefixes[regime] + [
                            {"role": "user", "content": prompt}
                        ]
                        item_prefix_ids, suffix_ids = split_before_final_user(
                            tokenizer, chat_token_ids(tokenizer, messages, True)
                        )
                        if item_prefix_ids != prefix_ids[regime]:
                            raise RuntimeError("source component prefix mismatch")
                        target_key = (
                            target_regime, task["id"], label_swap, order_swap
                        )
                        if target_key not in target_captures:
                            target_messages = prefixes[target_regime] + [
                                {"role": "user", "content": prompt}
                            ]
                            target_item_prefix, target_suffix = split_before_final_user(
                                tokenizer,
                                chat_token_ids(tokenizer, target_messages, True),
                            )
                            if target_item_prefix != prefix_ids[target_regime]:
                                raise RuntimeError("target component prefix mismatch")
                            if target_suffix != suffix_ids:
                                raise RuntimeError("paired component suffix mismatch")
                            target_cache = copy.deepcopy(base_caches[target_regime])
                            target_captures[target_key] = suffix_with_component_capture(
                                model, suffix_ids, len(prefix_ids[target_regime]),
                                target_cache, token_ids, args.layers, args.components,
                            )
                            del target_cache
                        targets, target_logit_a, target_logit_b = target_captures[
                            target_key
                        ]

                        for layer in args.layers:
                            for component in args.components:
                                for coefficient in [0.0] + args.coefficients:
                                    key = (
                                        regime, realization, task["id"], label_swap,
                                        order_swap, layer, component, coefficient,
                                    )
                                    if key in completed:
                                        continue
                                    cache = copy.deepcopy(base_caches[regime])
                                    started = time.time()
                                    logit_a, logit_b = suffix_with_component_patch(
                                        model, suffix_ids, len(prefix_ids[regime]),
                                        cache, token_ids, layer, component,
                                        targets[(layer, component)], coefficient,
                                    )
                                    record = {
                                        "regime": regime,
                                        "target_regime": target_regime,
                                        "history_style": args.history_style,
                                        "history_depth": args.depth,
                                        "history_realization": realization,
                                        "task_id": task["id"],
                                        "domain": task["domain"],
                                        "label_swap": label_swap,
                                        "evidence_order_swap": order_swap,
                                        "correct_label": correct_label,
                                        "foil_label": foil_label,
                                        "layer": layer,
                                        "component": component,
                                        "patch_coefficient": coefficient,
                                        "target_cached_logit_a": target_logit_a,
                                        "target_cached_logit_b": target_logit_b,
                                        "latency_s": round(time.time() - started, 3),
                                    }
                                    record.update(
                                        decision_record(logit_a, logit_b, correct_label)
                                    )
                                    output_handle.write(json.dumps(record) + "\n")
                                    print(
                                        f"component {regime}->{target_regime} "
                                        f"{task['id']} l={layer} {component} "
                                        f"c={coefficient} "
                                        f"margin={record['correct_logit_margin']:.3f}",
                                        flush=True,
                                    )
                                    del cache
                                    gc.collect()
        del base_caches, target_captures
        gc.collect()
        torch.cuda.empty_cache()
    output_handle.close()


if __name__ == "__main__":
    main()
