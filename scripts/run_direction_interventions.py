#!/usr/bin/env python3
"""Held-out-domain negative/positive governance-direction interventions.

Directions are estimated from saved activation pairs without the evaluated task
domain. History is prefetched once; each intervention runs only the task suffix.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import sys
import time
from collections import defaultdict
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


def load_metadata(input_dir):
    path = input_dir / "activation_metadata.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    for row in rows:
        row["activation"] = np.load(
            input_dir / row["activation_file"], allow_pickle=False
        ).astype(np.float32)
    return rows


def lodo_directions(rows, layers):
    task_ids = sorted({row["task_id"] for row in rows})
    directions = {}
    for held_out in task_ids:
        train = [row for row in rows if row["task_id"] != held_out]
        for layer in layers:
            obey = np.stack([
                row["activation"][layer] for row in train if row["regime"] == "obedience"
            ])
            verify = np.stack([
                row["activation"][layer] for row in train if row["regime"] == "verification"
            ])
            directions[(held_out, layer)] = obey.mean(0) - verify.mean(0)
    return directions


@torch.inference_mode()
def suffix_with_direction(
    model, suffix_ids, prefix_length, cache, token_ids, layer_index, direction, coefficient
):
    layer = model.model.language_model.layers[layer_index]

    def hook(_module, _inputs, output):
        value = output[0] if isinstance(output, tuple) else output
        edited = value.clone()
        vector = torch.as_tensor(direction, device=edited.device, dtype=edited.dtype)
        edited[:, -1, :] += coefficient * vector
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, default=Path("generated/mechanistic"))
    parser.add_argument("--output", type=Path, default=Path("generated/mechanistic/direction_interventions.jsonl"))
    parser.add_argument("--layers", nargs="+", type=int, required=True)
    parser.add_argument("--alphas", nargs="+", type=float, default=[0.5, 1.0, 2.0])
    parser.add_argument("--depth", type=int, default=64)
    parser.add_argument("--realizations", type=int, default=1)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--label-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument("--order-swaps", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    parser.add_argument(
        "--evaluation-history-style",
        default="natural",
        choices=["natural", "lexical_matched"],
        help="history style to intervene on; fitted directions come from --input-dir",
    )
    parser.add_argument("--gpu-memory-gib", type=int, default=21)
    parser.add_argument("--cpu-memory-gib", type=int, default=80)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    args = parser.parse_args()

    metadata = [
        row for row in load_metadata(args.input_dir)
        if row["history_depth"] == args.depth
    ]
    directions = lodo_directions(metadata, args.layers)
    tokenizer, model = load_model(args)
    token_ids = label_token_ids(tokenizer)
    tasks = [task for task in TASKS if not args.tasks or task["id"] in args.tasks]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_handle = args.output.open("w", buffering=1)
    for realization in range(args.realizations):
        for regime in ("verification", "obedience"):
            prefix_messages = [{"role": "system", "content": SYSTEM}]
            prefix_messages.extend(
                make_history(
                    args.depth, realization, regime, args.evaluation_history_style
                )
            )
            prefix_ids = exact_history_prefix_ids(tokenizer, prefix_messages)
            base_cache = prefill_history(model, prefix_ids)
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
                        for layer in args.layers:
                            direction = directions[(task["id"], layer)]
                            # Negative steering for mismatch; reverse steering establishes sufficiency.
                            sign = -1.0 if regime == "obedience" else 1.0
                            for alpha in [0.0] + args.alphas:
                                cache = copy.deepcopy(base_cache)
                                started = time.time()
                                logit_a, logit_b = suffix_with_direction(
                                    model,
                                    suffix_ids,
                                    len(prefix_ids),
                                    cache,
                                    token_ids,
                                    layer,
                                    direction,
                                    sign * alpha,
                                )
                                record = {
                                    "regime": regime,
                                    "history_style": args.evaluation_history_style,
                                    "history_depth": args.depth,
                                    "history_realization": realization,
                                    "task_id": task["id"],
                                    "domain": task["domain"],
                                    "label_swap": label_swap,
                                    "evidence_order_swap": order_swap,
                                    "correct_label": correct_label,
                                    "foil_label": foil_label,
                                    "layer": layer,
                                    "alpha": alpha,
                                    "signed_coefficient": sign * alpha,
                                    "direction_fit_excludes_task": task["id"],
                                    "direction_norm": float(np.linalg.norm(direction)),
                                    "latency_s": round(time.time() - started, 3),
                                }
                                record.update(decision_record(logit_a, logit_b, correct_label))
                                output_handle.write(json.dumps(record) + "\n")
                                print(
                                    f"direction {regime} {task['id']} layer={layer} "
                                    f"alpha={alpha} margin={record['correct_logit_margin']:.3f}",
                                    flush=True,
                                )
                                del cache
                                gc.collect()
            del base_cache
            gc.collect()
            torch.cuda.empty_cache()
    output_handle.close()


if __name__ == "__main__":
    main()
