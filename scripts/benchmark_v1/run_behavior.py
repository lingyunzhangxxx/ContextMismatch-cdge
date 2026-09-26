#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import struct
import sys
import time
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .common import DEFAULT_CONTRACT, canonical_json, load_jsonl, sha256_file
from .histories import pairwise_prompt, prefix_messages


def _device(name: str) -> torch.device:
    if name != "auto":
        if name.startswith("npu"):
            import torch_npu  # noqa: F401
        return torch.device(name)
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu:0")
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _chat_ids(tokenizer, messages: list[dict]) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(encoded, Mapping):
        encoded = encoded["input_ids"]
    if hasattr(encoded, "ids"):
        encoded = encoded.ids
    if isinstance(encoded, torch.Tensor):
        encoded = encoded.reshape(-1).tolist()
    if encoded and isinstance(encoded[0], list):
        encoded = encoded[0]
    return [int(value) for value in encoded]


def _split_final_user(tokenizer, full_ids: list[int]) -> tuple[list[int], list[int]]:
    marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    matches = [
        index
        for index in range(len(full_ids) - len(marker) + 1)
        if full_ids[index : index + len(marker)] == marker
    ]
    if not matches:
        raise RuntimeError("final user marker absent from rendered prompt")
    boundary = matches[-1]
    return full_ids[:boundary], full_ids[boundary:]


def _exact_prefix(tokenizer, messages: list[dict]) -> list[int]:
    probe = messages + [{"role": "user", "content": "Boundary probe; no task content."}]
    prefix, _ = _split_final_user(tokenizer, _chat_ids(tokenizer, probe))
    return prefix


def _token_hash(token_ids: list[int]) -> str:
    digest = hashlib.sha256()
    for value in token_ids:
        digest.update(struct.pack("<i", value))
    return digest.hexdigest()


def _label_ids(tokenizer) -> dict[str, int]:
    result = {}
    for label in ("A", "B"):
        values = tokenizer.encode(label, add_special_tokens=False)
        if len(values) != 1:
            raise ValueError(f"label {label} is not a single token: {values}")
        result[label] = int(values[0])
    return result


def _jobs(contract: dict, manifest: list[dict], stage: str) -> list[dict]:
    stage_contract = contract["behavior_stages"][stage]
    if stage == "smoke":
        chosen = []
        for benchmark in sorted({row["benchmark"] for row in manifest}):
            rows = [row for row in manifest if row["benchmark"] == benchmark and row["partition"] == stage_contract["partition"]]
            chosen.extend(sorted(rows, key=lambda row: row["item_id"])[: stage_contract["items_per_benchmark"]])
        depths = [contract["factorial"]["history_depth"]]
    elif stage == "depth_scan":
        chosen = []
        for benchmark in sorted({row["benchmark"] for row in manifest}):
            rows = [row for row in manifest if row["benchmark"] == benchmark and row["partition"] == stage_contract["partition"]]
            chosen.extend(sorted(rows, key=lambda row: row["item_id"])[: stage_contract["items_per_benchmark"]])
        depths = stage_contract["depths"]
    elif stage == "full_behavior":
        chosen = list(manifest)
        depths = [contract["factorial"]["history_depth"]]
    else:
        raise ValueError(stage)

    if stage == "full_behavior":
        conditions = contract["factorial"]["governance_conditions"]
        roles = contract["factorial"]["declared_roles"]
        styles = contract["factorial"]["history_styles"]
        label_swaps = contract["endpoint"]["label_swaps"]
    else:
        conditions = stage_contract["conditions"]
        roles = stage_contract["declared_roles"]
        styles = stage_contract["history_styles"]
        label_swaps = stage_contract["label_swaps"]
    jobs = []
    for item in chosen:
        for depth in depths:
            for role in roles:
                for condition in conditions:
                    condition_styles = ["none"] if condition == "fresh" else styles
                    for style in condition_styles:
                        for label_swap in label_swaps:
                            jobs.append(
                                {
                                    "item": item,
                                    "depth": depth,
                                    "role": role,
                                    "condition": condition,
                                    "style": style,
                                    "label_swap": label_swap,
                                    "realization": 0 if condition == "fresh" else item["history_realization"],
                                }
                            )
    return jobs


def _job_key(job: dict) -> str:
    item = job["item"]
    return "__".join(
        str(value)
        for value in (
            item["item_id"],
            job["depth"],
            job["role"],
            job["condition"],
            job["style"],
            job["label_swap"],
            job["realization"],
        )
    )


@torch.inference_mode()
def _prefill(model, device: torch.device, prefix_ids: list[int]):
    ids = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    output = model(input_ids=ids, use_cache=True, return_dict=True)
    return output.past_key_values


def _expand_cache_batch(cache, batch_size: int):
    expanded = copy.deepcopy(cache)
    if batch_size == 1:
        return expanded
    if hasattr(expanded, "batch_repeat_interleave"):
        expanded.batch_repeat_interleave(batch_size)
        return expanded
    if isinstance(expanded, (tuple, list)):
        return type(expanded)(
            type(layer)(tensor.repeat_interleave(batch_size, dim=0) for tensor in layer)
            for layer in expanded
        )
    raise TypeError(f"cache type does not support batch expansion: {type(expanded)!r}")


@torch.inference_mode()
def _suffix_logits_batch(
    model,
    device: torch.device,
    base_cache,
    prefix_length: int,
    suffix_batch: list[list[int]],
    label_ids: dict[str, int],
    pad_token_id: int,
):
    batch_size = len(suffix_batch)
    max_length = max(map(len, suffix_batch))
    input_rows = []
    suffix_masks = []
    position_rows = []
    for suffix_ids in suffix_batch:
        padding = max_length - len(suffix_ids)
        input_rows.append([pad_token_id] * padding + suffix_ids)
        suffix_masks.append([0] * padding + [1] * len(suffix_ids))
        position_rows.append([0] * padding + list(range(prefix_length, prefix_length + len(suffix_ids))))
    ids = torch.tensor(input_rows, device=device, dtype=torch.long)
    positions = torch.tensor(position_rows, device=device, dtype=torch.long)
    prefix_mask = torch.ones((batch_size, prefix_length), device=device, dtype=torch.long)
    suffix_mask = torch.tensor(suffix_masks, device=device, dtype=torch.long)
    attention_mask = torch.cat([prefix_mask, suffix_mask], dim=1)
    cache = _expand_cache_batch(base_cache, batch_size)
    kwargs = {
        "input_ids": ids,
        "position_ids": positions,
        "attention_mask": attention_mask,
        "past_key_values": cache,
        "use_cache": True,
        "return_dict": True,
    }
    try:
        output = model(logits_to_keep=1, **kwargs)
    except TypeError:
        output = model(**kwargs)
    logits = output.logits[:, -1].float()
    result = [
        (float(row[label_ids["A"]].cpu()), float(row[label_ids["B"]].cpu()))
        for row in logits
    ]
    return result, cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--stage", choices=["smoke", "depth_scan", "full_behavior"], required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if sha256_file(args.manifest) != manifest_report["manifest_sha256"]:
        raise ValueError("benchmark manifest hash mismatch")
    manifest_verified = bool(model_manifest.get("verified") or model_manifest.get("success"))
    manifest_non_quantized = bool(
        model_manifest.get("non_quantized") or model_manifest.get("quantization") == "none"
    )
    if not manifest_verified or model_manifest["revision"] != contract["base_model"]["revision"]:
        raise ValueError("model manifest is unverified or has the wrong revision")
    if not manifest_non_quantized:
        raise ValueError("quantized checkpoints are forbidden")

    manifest = load_jsonl(args.manifest)
    jobs = _jobs(contract, manifest, args.stage)
    completed = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        for row in load_jsonl(args.output):
            completed.add(row["job_key"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    device = _device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    label_ids = _label_ids(tokenizer)
    load_kwargs = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "trust_remote_code": False,
        "attn_implementation": args.attn_implementation,
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    except TypeError:
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(args.model_path, **load_kwargs)
    model = model.to(device)
    model.eval()
    environment = {
        "stage": args.stage,
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_model_class": type(model).__name__,
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "contract_sha256": sha256_file(args.contract),
        "label_token_ids": label_ids,
        "planned_rows": len(jobs),
        "batch_size": args.batch_size,
        "batching_strategy": "exact_suffix_length_buckets",
        "production_rollout_approved": False,
    }
    if args.environment_output.exists() and not args.resume:
        raise FileExistsError(f"refusing to overwrite {args.environment_output}")
    if not args.environment_output.exists():
        args.environment_output.parent.mkdir(parents=True, exist_ok=True)
        args.environment_output.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for job in jobs:
        key = (job["role"], job["condition"], job["style"], job["depth"], job["realization"])
        grouped[key].append(job)
    output_handle = args.output.open("a", buffering=1)
    for prefix_key in sorted(grouped):
        role, condition, style, depth, realization = prefix_key
        history_style = "natural" if style == "none" else style
        messages = prefix_messages(role, condition, history_style, depth, realization)
        prefix_ids = _exact_prefix(tokenizer, messages)
        base_cache = _prefill(model, device, prefix_ids)
        pending = []
        for job in grouped[prefix_key]:
            key = _job_key(job)
            if key in completed:
                continue
            item = job["item"]
            prompt, correct_label, foil_label = pairwise_prompt(item, job["label_swap"])
            full_messages = messages + [{"role": "user", "content": prompt}]
            rendered_prefix, suffix_ids = _split_final_user(tokenizer, _chat_ids(tokenizer, full_messages))
            if rendered_prefix != prefix_ids:
                raise RuntimeError(f"prefix mismatch for {key}")
            if len(prefix_ids) + len(suffix_ids) > contract["base_model"]["max_position_embeddings"]:
                raise ValueError(f"context length exceeded for {key}")
            pending.append((job, key, item, correct_label, foil_label, suffix_ids))
        by_suffix_length: dict[int, list[tuple]] = defaultdict(list)
        for value in pending:
            by_suffix_length[len(value[-1])].append(value)
        for suffix_length in sorted(by_suffix_length):
            same_length = sorted(by_suffix_length[suffix_length], key=lambda value: value[1])
            for offset in range(0, len(same_length), args.batch_size):
                batch = same_length[offset : offset + args.batch_size]
                if len({len(value[-1]) for value in batch}) != 1:
                    raise RuntimeError("a batch crossed exact suffix-length buckets")
                started = time.time()
                logits, cache = _suffix_logits_batch(
                    model,
                    device,
                    base_cache,
                    len(prefix_ids),
                    [value[-1] for value in batch],
                    label_ids,
                    int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id),
                )
                batch_latency = time.time() - started
                for (job, key, item, correct_label, foil_label, suffix_ids), (logit_a, logit_b) in zip(batch, logits):
                    values = {"A": logit_a, "B": logit_b}
                    margin = values[correct_label] - values[foil_label]
                    answer = "A" if logit_a >= logit_b else "B"
                    record = {
                        "job_key": key,
                        "stage": args.stage,
                        "benchmark": item["benchmark"],
                        "item_id": item["item_id"],
                        "partition": item["partition"],
                        "declared_role": role,
                        "condition": condition,
                        "history_style": style,
                        "history_depth": depth,
                        "history_realization": realization,
                        "label_swap": job["label_swap"],
                        "correct_label": correct_label,
                        "foil_label": foil_label,
                        "logit_a": logit_a,
                        "logit_b": logit_b,
                        "correct_logit_margin": margin,
                        "correct_probability_binary": 1.0
                        / (1.0 + math.exp(max(-700.0, min(700.0, -margin)))),
                        "answer": answer,
                        "correct": answer == correct_label,
                        "followed_unsupported_preference": answer == foil_label,
                        "prefix_tokens": len(prefix_ids),
                        "suffix_tokens": len(suffix_ids),
                        "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                        "latency_s": round(batch_latency / len(batch), 4),
                        "batch_latency_s": round(batch_latency, 4),
                        "batch_size_observed": len(batch),
                        "model_revision": contract["base_model"]["revision"],
                        "benchmark_manifest_sha256": environment["benchmark_manifest_sha256"],
                    }
                    if not all(math.isfinite(record[name]) for name in ("logit_a", "logit_b", "correct_logit_margin")):
                        raise FloatingPointError(f"non-finite result for {key}")
                    output_handle.write(canonical_json(record) + "\n")
                    completed.add(key)
                del cache
        del base_cache
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        elif device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()
    output_handle.close()
    print(json.dumps({"planned_rows": len(jobs), "completed_rows": len(completed), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
