#!/usr/bin/env python3
"""Mechanistic activation and hybrid-cache interventions for Qwen3.5/3.6.

This runner deliberately records only the final decision-token residual from
each layer. It never requests full-sequence hidden states. Cache interventions
separate Gated DeltaNet recurrent/conv state from full-attention KV state while
passing absolute position IDs explicitly.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch

# GPTQModel 7.3.2 imports optional model definitions eagerly.  Its Solar Open
# 2 definition expects this helper from a newer Transformers release, although
# the Qwen3.5/3.6 path used here never calls it.  Install a fail-closed symbol
# before GPTQModel can be imported by ``from_pretrained``.
import transformers.masking_utils as masking_utils

if not hasattr(masking_utils, "create_recurrent_attention_mask"):
    def _unavailable_recurrent_attention_mask(*_args, **_kwargs):
        raise RuntimeError(
            "create_recurrent_attention_mask compatibility stub was called; "
            "it is not implemented for this Transformers version"
        )

    masking_utils.create_recurrent_attention_mask = (  # type: ignore[attr-defined]
        _unavailable_recurrent_attention_mask
    )

from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_length_scan import (  # noqa: E402
    SYSTEM,
    TASKS,
    final_prompt,
    make_history,
)


def parse_int_choices(values: list[int] | None) -> list[int]:
    return values if values is not None else [0, 1]


def load_model(args):
    max_memory = {
        0: f"{args.gpu_memory_gib}GiB",
        1: f"{args.gpu_memory_gib}GiB",
        "cpu": f"{args.cpu_memory_gib}GiB",
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model_path,
        device_map="balanced",
        max_memory=max_memory,
        dtype="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    return tokenizer, model


def text_model(model):
    return model.model.language_model


def input_device(model):
    return text_model(model).embed_tokens.weight.device


def chat_token_ids(tokenizer, messages, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    # Transformers 5.x may return a low-level ``tokenizers.Encoding`` for the
    # multimodal Qwen3.5 tokenizer, whereas earlier versions returned a list.
    # Normalize without decoding/re-encoding so the experimental prompt stays
    # byte-for-byte identical to the chat-template output.
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.reshape(-1).tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    if not all(isinstance(token_id, int) for token_id in encoded):
        raise TypeError(
            f"chat template returned non-integer token IDs: {type(encoded)!r}"
        )
    return list(encoded)


def split_before_final_user(tokenizer, full_ids: list[int]) -> tuple[list[int], list[int]]:
    """Split a rendered chat at the final user-role header.

    Qwen3.5 inserts an empty ``<think>...</think>`` block when a conversation
    ending in an assistant message is rendered alone, but omits that block when
    another user turn follows.  Therefore a separately rendered history is not
    an exact token prefix of the full prompt.  Splitting the already-rendered
    full prompt avoids that template artifact.
    """
    marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    matches = [
        index
        for index in range(len(full_ids) - len(marker) + 1)
        if full_ids[index : index + len(marker)] == marker
    ]
    if not matches:
        raise RuntimeError("final user-role marker not found in rendered chat")
    boundary = matches[-1]
    if boundary == 0:
        raise RuntimeError("rendered chat has no pre-boundary history")
    return full_ids[:boundary], full_ids[boundary:]


def exact_history_prefix_ids(tokenizer, prefix_messages) -> list[int]:
    """Render history in a continued conversation and return its exact prefix."""
    probe_messages = list(prefix_messages) + [
        {"role": "user", "content": "Boundary probe; no task content."}
    ]
    probe_ids = chat_token_ids(tokenizer, probe_messages, True)
    prefix_ids, _ = split_before_final_user(tokenizer, probe_ids)
    return prefix_ids


def label_token_ids(tokenizer) -> dict[str, int]:
    result = {}
    for label in ("A", "B"):
        ids = tokenizer.encode(label, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"label {label!r} is not one token: {ids}")
        result[label] = ids[0]
    return result


def messages_for_item(
    task, regime, depth, realization, label_swap, order_swap,
    history_style="natural",
):
    messages = [{"role": "system", "content": SYSTEM}]
    messages.extend(make_history(depth, realization, regime, history_style))
    prompt, correct_label, foil_label = final_prompt(task, label_swap, order_swap)
    messages.append({"role": "user", "content": prompt})
    return messages, correct_label, foil_label


def tensor_logits(model, output, token_ids):
    logits = output.logits[0, -1]
    a = float(logits[token_ids["A"]].float().cpu())
    b = float(logits[token_ids["B"]].float().cpu())
    return a, b


def decision_record(logit_a, logit_b, correct_label):
    answer = "A" if logit_a >= logit_b else "B"
    correct_logit = logit_a if correct_label == "A" else logit_b
    foil_logit = logit_b if correct_label == "A" else logit_a
    margin = correct_logit - foil_logit
    return {
        "logit_a": logit_a,
        "logit_b": logit_b,
        "answer": answer,
        "correct": answer == correct_label,
        "correct_logit_margin": margin,
        "correct_probability_binary": 1.0 / (1.0 + math.exp(max(-700, min(700, -margin)))),
    }


@torch.inference_mode()
def forward_full(model, input_ids, token_ids, capture=False):
    captures = []
    handles = []
    if capture:
        layers = text_model(model).layers
        captures = [None] * (len(layers) + 1)

        def layer_hook(index):
            def hook(_module, _inputs, output):
                value = output[0] if isinstance(output, tuple) else output
                captures[index] = value[:, -1, :].detach().to("cpu", dtype=torch.float16)
            return hook

        for index, layer in enumerate(layers):
            handles.append(layer.register_forward_hook(layer_hook(index)))
        handles.append(
            text_model(model).norm.register_forward_hook(layer_hook(len(layers)))
        )
    try:
        output = model(
            input_ids=input_ids,
            use_cache=False,
            logits_to_keep=1,
        )
        logit_a, logit_b = tensor_logits(model, output, token_ids)
    finally:
        for handle in handles:
            handle.remove()
    if capture:
        if any(value is None for value in captures):
            missing = [index for index, value in enumerate(captures) if value is None]
            raise RuntimeError(f"activation hooks missing layers: {missing}")
        activation = torch.cat(captures, dim=0).numpy()
    else:
        activation = None
    return logit_a, logit_b, activation


def item_key(regime, depth, realization, task_id, label_swap, order_swap):
    return (
        f"{regime}__d{depth}__r{realization}__{task_id}"
        f"__label{label_swap}__order{order_swap}"
    )


def run_collect(args, tokenizer, model, token_ids):
    out = args.output_dir / "activations"
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = args.output_dir / "activation_metadata.jsonl"
    completed = set()
    if args.resume and metadata_path.exists():
        completed = {
            json.loads(line)["key"]
            for line in metadata_path.read_text().splitlines()
            if line.strip()
        }
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    label_swaps = parse_int_choices(args.label_swaps)
    order_swaps = parse_int_choices(args.order_swaps)
    mode = "a" if args.resume else "w"
    with metadata_path.open(mode, buffering=1) as metadata_handle:
        for depth in args.depths:
            for realization in range(args.realizations):
                for task in tasks:
                    for label_swap in label_swaps:
                        for order_swap in order_swaps:
                            for regime in ("verification", "obedience"):
                                key = item_key(
                                    regime, depth, realization, task["id"],
                                    label_swap, order_swap,
                                )
                                if key in completed:
                                    print(f"skip completed {key}", flush=True)
                                    continue
                                messages, correct_label, foil_label = messages_for_item(
                                    task, regime, depth, realization, label_swap,
                                    order_swap, args.history_style,
                                )
                                ids = chat_token_ids(tokenizer, messages, True)
                                input_ids = torch.tensor([ids], device=input_device(model))
                                started = time.time()
                                logit_a, logit_b, activation = forward_full(
                                    model, input_ids, token_ids, capture=True
                                )
                                activation_path = out / f"{key}.npy"
                                np.save(activation_path, activation, allow_pickle=False)
                                record = {
                                    "key": key,
                                    "regime": regime,
                                    "history_style": args.history_style,
                                    "history_depth": depth,
                                    "history_realization": realization,
                                    "task_id": task["id"],
                                    "domain": task["domain"],
                                    "label_swap": label_swap,
                                    "evidence_order_swap": order_swap,
                                    "correct_label": correct_label,
                                    "foil_label": foil_label,
                                    "prompt_tokens": len(ids),
                                    "prompt_sha256": hashlib.sha256(
                                        np.asarray(ids, dtype=np.int32).tobytes()
                                    ).hexdigest(),
                                    "activation_file": str(activation_path.relative_to(args.output_dir)),
                                    "activation_shape": list(activation.shape),
                                    "activation_dtype": str(activation.dtype),
                                    "latency_s": round(time.time() - started, 3),
                                    "messages": messages,
                                }
                                record.update(decision_record(logit_a, logit_b, correct_label))
                                metadata_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                print(
                                    f"collect {key} correct={record['correct']} "
                                    f"margin={record['correct_logit_margin']:.3f} "
                                    f"tokens={len(ids)} latency={record['latency_s']}",
                                    flush=True,
                                )
                                del input_ids, activation
                                gc.collect()


def reset_cache(cache, intervention, layer_types):
    drop_kv = intervention in {
        "kv_drop", "recurrent_zero_kv_drop", "delta_reset_kv_drop"
    }
    recurrent_zero = intervention in {"recurrent_zero", "recurrent_zero_kv_drop"}
    conv_zero = intervention == "conv_zero"
    delta_reset = intervention in {"delta_reset", "delta_reset_kv_drop"}
    for index, layer_type in enumerate(layer_types):
        layer = cache.layers[index]
        if layer_type == "linear_attention":
            if delta_reset:
                layer.reset()
            else:
                if recurrent_zero and getattr(layer, "is_recurrent_states_initialized", False):
                    layer.recurrent_states.zero_()
                if conv_zero and getattr(layer, "is_conv_states_initialized", False):
                    layer.conv_states.zero_()
        elif layer_type == "full_attention" and drop_kv:
            if getattr(layer, "is_initialized", False):
                layer.keys = layer.keys[..., :0, :].contiguous()
                layer.values = layer.values[..., :0, :].contiguous()


def cache_bytes(cache, layer_types):
    totals = {"kv": 0, "conv": 0, "recurrent": 0}
    for index, layer_type in enumerate(layer_types):
        layer = cache.layers[index]
        if layer_type == "full_attention" and getattr(layer, "is_initialized", False):
            totals["kv"] += layer.keys.numel() * layer.keys.element_size()
            totals["kv"] += layer.values.numel() * layer.values.element_size()
        elif layer_type == "linear_attention":
            if getattr(layer, "is_conv_states_initialized", False):
                totals["conv"] += layer.conv_states.numel() * layer.conv_states.element_size()
            if getattr(layer, "is_recurrent_states_initialized", False):
                totals["recurrent"] += (
                    layer.recurrent_states.numel() * layer.recurrent_states.element_size()
                )
    return totals


@torch.inference_mode()
def prefill_history(model, prefix_ids):
    ids = torch.tensor([prefix_ids], device=input_device(model))
    output = text_model(model)(input_ids=ids, use_cache=True)
    return output.past_key_values


@torch.inference_mode()
def forward_suffix(model, suffix_ids, prefix_length, cache, token_ids):
    device = input_device(model)
    ids = torch.tensor([suffix_ids], device=device)
    absolute_positions = torch.arange(
        prefix_length, prefix_length + len(suffix_ids), device=device
    ).unsqueeze(0)
    output = model(
        input_ids=ids,
        position_ids=absolute_positions,
        past_key_values=cache,
        use_cache=True,
        logits_to_keep=1,
    )
    return tensor_logits(model, output, token_ids)


def run_cache(args, tokenizer, model, token_ids):
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    label_swaps = parse_int_choices(args.label_swaps)
    order_swaps = parse_int_choices(args.order_swaps)
    layer_types = list(text_model(model).config.layer_types)
    output_path = args.output_dir / "cache_interventions.jsonl"
    completed = set()
    if args.resume and output_path.exists():
        for line in output_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            completed.add((
                row["regime"], row["history_depth"],
                row["history_realization"], row["task_id"],
                row["label_swap"], row["evidence_order_swap"],
                row["intervention"],
            ))
        print(f"resume cache: {len(completed)} completed rows", flush=True)
    mode = "a" if args.resume else "w"
    with output_path.open(mode, buffering=1) as output_handle:
        for depth in args.depths:
            for realization in range(args.realizations):
                for regime in ("verification", "obedience"):
                    prefix_messages = [{"role": "system", "content": SYSTEM}]
                    prefix_messages.extend(
                        make_history(depth, realization, regime, args.history_style)
                    )
                    prefix_ids = exact_history_prefix_ids(tokenizer, prefix_messages)
                    print(
                        f"prefill regime={regime} depth={depth} realization={realization} "
                        f"tokens={len(prefix_ids)}",
                        flush=True,
                    )
                    base_cache = prefill_history(model, prefix_ids)
                    state_bytes = cache_bytes(base_cache, layer_types)
                    print(f"cache bytes={state_bytes}", flush=True)
                    for task in tasks:
                        for label_swap in label_swaps:
                            for order_swap in order_swaps:
                                messages, correct_label, foil_label = messages_for_item(
                                    task, regime, depth, realization, label_swap,
                                    order_swap, args.history_style,
                                )
                                full_ids = chat_token_ids(tokenizer, messages, True)
                                item_prefix_ids, suffix_ids = split_before_final_user(
                                    tokenizer, full_ids
                                )
                                if item_prefix_ids != prefix_ids:
                                    raise RuntimeError(
                                        "chat-template prefix mismatch; cache boundary is invalid"
                                    )
                                for intervention in args.cache_interventions:
                                    key = (
                                        regime, depth, realization, task["id"],
                                        label_swap, order_swap, intervention,
                                    )
                                    if key in completed:
                                        continue
                                    cache = copy.deepcopy(base_cache)
                                    reset_cache(cache, intervention, layer_types)
                                    started = time.time()
                                    logit_a, logit_b = forward_suffix(
                                        model, suffix_ids, len(prefix_ids), cache, token_ids
                                    )
                                    record = {
                                        "regime": regime,
                                        "history_style": args.history_style,
                                        "history_depth": depth,
                                        "history_realization": realization,
                                        "task_id": task["id"],
                                        "domain": task["domain"],
                                        "label_swap": label_swap,
                                        "evidence_order_swap": order_swap,
                                        "correct_label": correct_label,
                                        "foil_label": foil_label,
                                        "intervention": intervention,
                                        "prefix_tokens": len(prefix_ids),
                                        "suffix_tokens": len(suffix_ids),
                                        "absolute_position_start": len(prefix_ids),
                                        "cache_bytes_before": state_bytes,
                                        "latency_s": round(time.time() - started, 3),
                                    }
                                    record.update(decision_record(logit_a, logit_b, correct_label))
                                    output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                                    print(
                                        f"cache {regime} d={depth} {task['id']} l={label_swap} "
                                        f"o={order_swap} {intervention} correct={record['correct']} "
                                        f"margin={record['correct_logit_margin']:.3f}",
                                        flush=True,
                                    )
                                    del cache
                                    gc.collect()
                    del base_cache
                    gc.collect()
                    torch.cuda.empty_cache()


def write_environment(args, tokenizer, model):
    text_config = text_model(model).config
    environment = {
        "model_path": str(args.model_path),
        "model_class": type(model).__name__,
        "transformers_version": __import__("transformers").__version__,
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
        "device_map": getattr(model, "hf_device_map", None),
        "num_hidden_layers": text_config.num_hidden_layers,
        "hidden_size": text_config.hidden_size,
        "layer_types": list(text_config.layer_types),
        "label_token_ids": label_token_ids(tokenizer),
        "arguments": vars(args),
    }
    serializable = json.loads(json.dumps(environment, default=str))
    (args.output_dir / "environment.json").write_text(
        json.dumps(serializable, indent=2) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("generated/mechanistic"))
    parser.add_argument("--modes", nargs="+", choices=["collect", "cache"], default=["collect"])
    parser.add_argument("--depths", nargs="+", type=int, default=[64])
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument(
        "--history-style",
        default="natural",
        choices=["natural", "lexical_matched"],
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="append only missing activation/cache keys instead of replacing output",
    )
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--label-swaps", nargs="+", type=int, choices=[0, 1])
    parser.add_argument("--order-swaps", nargs="+", type=int, choices=[0, 1])
    parser.add_argument(
        "--cache-interventions",
        nargs="+",
        default=["intact", "recurrent_zero", "conv_zero", "delta_reset", "kv_drop", "recurrent_zero_kv_drop", "delta_reset_kv_drop"],
        choices=["intact", "recurrent_zero", "conv_zero", "delta_reset", "kv_drop", "recurrent_zero_kv_drop", "delta_reset_kv_drop"],
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=21)
    parser.add_argument("--cpu-memory-gib", type=int, default=80)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args)
    token_ids = label_token_ids(tokenizer)
    write_environment(args, tokenizer, model)
    if "collect" in args.modes:
        run_collect(args, tokenizer, model, token_ids)
    if "cache" in args.modes:
        run_cache(args, tokenizer, model, token_ids)


if __name__ == "__main__":
    main()
