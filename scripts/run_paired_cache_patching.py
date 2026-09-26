#!/usr/bin/env python3
"""Paired counterfactual cache-state patching for lexical-matched histories."""

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
    forward_suffix,
    label_token_ids,
    load_model,
    prefill_history,
    split_before_final_user,
    text_model,
)


GLOBAL_INTERVENTIONS = [
    "intact",
    "paired_recurrent",
    "paired_conv",
    "paired_kv",
    "paired_recurrent_kv",
    "paired_all",
]
STAGE_INTERVENTIONS = [f"paired_stage_{index}" for index in range(8)]


def copy_tensor(destination, source, label):
    if destination.shape != source.shape:
        raise RuntimeError(
            f"paired cache shape mismatch for {label}: "
            f"{tuple(destination.shape)} != {tuple(source.shape)}"
        )
    destination.copy_(source)


def patch_cache(source_cache, target_cache, layer_types, intervention):
    if intervention == "intact":
        return
    selected_layers = set(range(len(layer_types)))
    components = set()
    if intervention == "paired_recurrent":
        components = {"recurrent"}
    elif intervention == "paired_conv":
        components = {"conv"}
    elif intervention == "paired_kv":
        components = {"kv"}
    elif intervention == "paired_recurrent_kv":
        components = {"recurrent", "kv"}
    elif intervention == "paired_all":
        components = {"recurrent", "conv", "kv"}
    elif intervention.startswith("paired_stage_"):
        stage = int(intervention.rsplit("_", 1)[1])
        selected_layers = set(range(stage * 4, stage * 4 + 4))
        components = {"recurrent", "conv", "kv"}
    else:
        raise ValueError(f"unknown paired cache intervention: {intervention}")

    for index, layer_type in enumerate(layer_types):
        if index not in selected_layers:
            continue
        source_layer = source_cache.layers[index]
        target_layer = target_cache.layers[index]
        if layer_type == "linear_attention":
            if "recurrent" in components:
                if not (
                    source_layer.is_recurrent_states_initialized
                    and target_layer.is_recurrent_states_initialized
                ):
                    raise RuntimeError(f"uninitialized recurrent state at layer {index}")
                copy_tensor(
                    source_layer.recurrent_states,
                    target_layer.recurrent_states,
                    f"layer {index} recurrent",
                )
            if "conv" in components:
                if not (
                    source_layer.is_conv_states_initialized
                    and target_layer.is_conv_states_initialized
                ):
                    raise RuntimeError(f"uninitialized conv state at layer {index}")
                copy_tensor(
                    source_layer.conv_states,
                    target_layer.conv_states,
                    f"layer {index} conv",
                )
        elif layer_type == "full_attention" and "kv" in components:
            if not (source_layer.is_initialized and target_layer.is_initialized):
                raise RuntimeError(f"uninitialized KV state at layer {index}")
            copy_tensor(source_layer.keys, target_layer.keys, f"layer {index} keys")
            copy_tensor(source_layer.values, target_layer.values, f"layer {index} values")


def load_completed(output):
    if not output.exists():
        return set()
    completed = set()
    for line in output.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        completed.add((
            row["regime"], row["history_realization"], row["task_id"],
            row["label_swap"], row["evidence_order_swap"], row["intervention"],
        ))
    return completed


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=32)
    parser.add_argument("--realizations", type=int, default=3)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--label-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--order-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument(
        "--interventions",
        nargs="+",
        choices=GLOBAL_INTERVENTIONS + STAGE_INTERVENTIONS,
        default=GLOBAL_INTERVENTIONS + STAGE_INTERVENTIONS,
    )
    parser.add_argument(
        "--history-style",
        default="lexical_matched",
        choices=["lexical_matched"],
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=21)
    parser.add_argument("--cpu-memory-gib", type=int, default=80)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    tokenizer, model = load_model(args)
    token_ids = label_token_ids(tokenizer)
    layer_types = list(text_model(model).config.layer_types)
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed(args.output) if args.resume else set()
    print(f"resume paired cache: {len(completed)} completed rows", flush=True)
    output_handle = args.output.open("a" if args.resume else "w", buffering=1)

    for realization in range(args.realizations):
        prefix_messages = {}
        prefix_ids = {}
        base_caches = {}
        for regime in ("verification", "obedience"):
            messages = [{"role": "system", "content": SYSTEM}]
            messages.extend(
                make_history(args.depth, realization, regime, args.history_style)
            )
            prefix_messages[regime] = messages
            prefix_ids[regime] = exact_history_prefix_ids(tokenizer, messages)
            base_caches[regime] = prefill_history(model, prefix_ids[regime])
        if len(prefix_ids["verification"]) != len(prefix_ids["obedience"]):
            raise RuntimeError("paired cache patching requires token-length matched histories")

        for regime in ("verification", "obedience"):
            target_regime = "obedience" if regime == "verification" else "verification"
            for task in tasks:
                for label_swap in args.label_swaps:
                    for order_swap in args.order_swaps:
                        prompt, correct_label, foil_label = final_prompt(
                            task, label_swap, order_swap
                        )
                        messages = prefix_messages[regime] + [
                            {"role": "user", "content": prompt}
                        ]
                        full_ids = chat_token_ids(tokenizer, messages, True)
                        item_prefix_ids, suffix_ids = split_before_final_user(
                            tokenizer, full_ids
                        )
                        if item_prefix_ids != prefix_ids[regime]:
                            raise RuntimeError("chat-template prefix mismatch")
                        for intervention in args.interventions:
                            key = (
                                regime, realization, task["id"], label_swap,
                                order_swap, intervention,
                            )
                            if key in completed:
                                continue
                            cache = copy.deepcopy(base_caches[regime])
                            patch_cache(
                                cache, base_caches[target_regime], layer_types,
                                intervention,
                            )
                            started = time.time()
                            logit_a, logit_b = forward_suffix(
                                model, suffix_ids, len(prefix_ids[regime]), cache,
                                token_ids,
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
                                "intervention": intervention,
                                "prefix_tokens": len(prefix_ids[regime]),
                                "suffix_tokens": len(suffix_ids),
                                "latency_s": round(time.time() - started, 3),
                            }
                            record.update(
                                decision_record(logit_a, logit_b, correct_label)
                            )
                            output_handle.write(json.dumps(record) + "\n")
                            print(
                                f"paired-cache {regime}->{target_regime} "
                                f"{task['id']} {intervention} "
                                f"margin={record['correct_logit_margin']:.3f}",
                                flush=True,
                            )
                            del cache
                            gc.collect()
        del base_caches
        gc.collect()
        torch.cuda.empty_cache()
    output_handle.close()


if __name__ == "__main__":
    main()
