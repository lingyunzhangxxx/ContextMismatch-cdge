#!/usr/bin/env python3
"""Exact paired residual patching for context-mismatch causal mediation."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_length_scan import SYSTEM, TASKS, final_prompt, make_history  # noqa: E402
from run_mechanistic import (  # noqa: E402
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


def load_metadata(input_dir, depth):
    path = input_dir / "activation_metadata.jsonl"
    rows = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]
    rows = [row for row in rows if row["history_depth"] == depth]
    for row in rows:
        row["activation"] = np.load(
            input_dir / row["activation_file"], allow_pickle=False
        )
    return rows


@torch.inference_mode()
def suffix_with_patch(
    model, suffix_ids, prefix_length, cache, token_ids, layer_index,
    target_activation, coefficient,
):
    layer = model.model.language_model.layers[layer_index]

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        edited = value.clone()
        target = torch.as_tensor(
            target_activation, device=edited.device, dtype=edited.dtype
        )
        edited[:, -1, :].lerp_(target, coefficient)
        if isinstance(output, tuple):
            return (edited,) + output[1:]
        return edited

    handle = layer.register_forward_hook(hook)
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


@torch.inference_mode()
def suffix_with_capture(
    model, suffix_ids, prefix_length, cache, token_ids, layer_indices,
):
    captures = {}
    handles = []

    def layer_hook(layer_index):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, tuple) else output
            captures[layer_index] = (
                value[0, -1].detach().to(device="cpu", dtype=torch.float16).numpy()
            )
        return hook

    for layer_index in layer_indices:
        layer = model.model.language_model.layers[layer_index]
        handles.append(layer.register_forward_hook(layer_hook(layer_index)))
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
        missing = set(layer_indices) - set(captures)
        if missing:
            raise RuntimeError(f"cached target hooks missing layers: {sorted(missing)}")
        logit_a, logit_b = tensor_logits(model, output, token_ids)
        return captures, logit_a, logit_b
    finally:
        for handle in handles:
            handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--coefficients", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--label-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--order-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument(
        "--history-style", default="lexical_matched",
        choices=["natural", "lexical_matched"],
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=21)
    parser.add_argument("--cpu-memory-gib", type=int, default=80)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument(
        "--target-source",
        default="full_forward_files",
        choices=["full_forward_files", "cached_suffix"],
        help="obtain target residuals from old full forwards or the exact cached suffix path",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    activation_index = {}
    if args.target_source == "full_forward_files":
        if args.input_dir is None:
            parser.error("--input-dir is required for --target-source full_forward_files")
        rows = load_metadata(args.input_dir, args.depth)
        activation_index = {
            (
                row["regime"], row["history_realization"], row["task_id"],
                row["label_swap"], row["evidence_order_swap"],
            ): row["activation"]
            for row in rows
        }
    tokenizer, model = load_model(args)
    token_ids = label_token_ids(tokenizer)
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = set()
    if args.resume and args.output.exists():
        for line in args.output.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((
                row["regime"], row["history_realization"], row["task_id"],
                row["label_swap"], row["evidence_order_swap"], row["layer"],
                row["patch_coefficient"],
            ))
        print(f"resume patching: {len(completed)} completed rows", flush=True)
    output_handle = args.output.open("a" if args.resume else "w", buffering=1)
    for realization in range(args.realizations):
        prefixes = {}
        prefix_ids_by_regime = {}
        base_caches = {}
        for prefix_regime in ("verification", "obedience"):
            prefix_messages = [{"role": "system", "content": SYSTEM}]
            prefix_messages.extend(
                make_history(
                    args.depth, realization, prefix_regime, args.history_style
                )
            )
            prefixes[prefix_regime] = prefix_messages
            prefix_ids_by_regime[prefix_regime] = exact_history_prefix_ids(
                tokenizer, prefix_messages
            )
            base_caches[prefix_regime] = prefill_history(
                model, prefix_ids_by_regime[prefix_regime]
            )
        if (
            args.target_source == "cached_suffix"
            and len(prefix_ids_by_regime["verification"])
            != len(prefix_ids_by_regime["obedience"])
        ):
            raise RuntimeError(
                "cached target patching requires token-length matched histories"
            )
        cached_targets = {}
        for regime in ("verification", "obedience"):
            target_regime = "obedience" if regime == "verification" else "verification"
            prefix_messages = prefixes[regime]
            prefix_ids = prefix_ids_by_regime[regime]
            for task in tasks:
                for label_swap in args.label_swaps:
                    for order_swap in args.order_swaps:
                        prompt, correct_label, foil_label = final_prompt(
                            task, label_swap, order_swap
                        )
                        messages = prefix_messages + [{"role": "user", "content": prompt}]
                        full_ids = chat_token_ids(tokenizer, messages, True)
                        item_prefix_ids, suffix_ids = split_before_final_user(
                            tokenizer, full_ids
                        )
                        if item_prefix_ids != prefix_ids:
                            raise RuntimeError("chat-template prefix mismatch")
                        target_key = (
                            target_regime, realization, task["id"], label_swap,
                            order_swap,
                        )
                        if args.target_source == "cached_suffix":
                            if target_key not in cached_targets:
                                target_messages = prefixes[target_regime] + [
                                    {"role": "user", "content": prompt}
                                ]
                                target_full_ids = chat_token_ids(
                                    tokenizer, target_messages, True
                                )
                                _, target_suffix_ids = split_before_final_user(
                                    tokenizer, target_full_ids
                                )
                                if target_suffix_ids != suffix_ids:
                                    raise RuntimeError(
                                        "paired cached target has a different suffix"
                                    )
                                target_cache = copy.deepcopy(base_caches[target_regime])
                                target, target_logit_a, target_logit_b = (
                                    suffix_with_capture(
                                        model, suffix_ids,
                                        len(prefix_ids_by_regime[target_regime]),
                                        target_cache, token_ids, args.layers,
                                    )
                                )
                                cached_targets[target_key] = (
                                    target, target_logit_a, target_logit_b
                                )
                                del target_cache
                            target, target_logit_a, target_logit_b = cached_targets[
                                target_key
                            ]
                        else:
                            target = activation_index[target_key]
                            target_logit_a = None
                            target_logit_b = None
                        for layer in args.layers:
                            for coefficient in [0.0] + args.coefficients:
                                key = (
                                    regime, realization, task["id"], label_swap,
                                    order_swap, layer, coefficient,
                                )
                                if key in completed:
                                    continue
                                cache = copy.deepcopy(base_caches[regime])
                                started = time.time()
                                logit_a, logit_b = suffix_with_patch(
                                    model, suffix_ids, len(prefix_ids), cache, token_ids,
                                    layer, target[layer], coefficient,
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
                                    "patch_coefficient": coefficient,
                                    "target_source": args.target_source,
                                    "target_cached_logit_a": target_logit_a,
                                    "target_cached_logit_b": target_logit_b,
                                    "latency_s": round(time.time() - started, 3),
                                }
                                record.update(
                                    decision_record(logit_a, logit_b, correct_label)
                                )
                                output_handle.write(json.dumps(record) + "\n")
                                print(
                                    f"patch {regime}->{target_regime} {task['id']} "
                                    f"layer={layer} c={coefficient} "
                                    f"margin={record['correct_logit_margin']:.3f}",
                                    flush=True,
                                )
                                del cache
                                gc.collect()
        del base_caches, cached_targets
        gc.collect()
        torch.cuda.empty_cache()
    output_handle.close()


if __name__ == "__main__":
    main()
