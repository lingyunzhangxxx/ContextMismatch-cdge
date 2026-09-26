#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path

from transformers import AutoTokenizer

from .common import DEFAULT_CONTRACT, atomic_write_text, load_jsonl, sha256_file
from .histories import pairwise_prompt, prefix_messages


def ids(tokenizer, messages: list[dict], add_generation_prompt: bool) -> list[int]:
    value = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
    if isinstance(value, Mapping):
        value = value["input_ids"]
    if hasattr(value, "ids"):
        value = value.ids
    if value and isinstance(value[0], list):
        value = value[0]
    return [int(token) for token in value]


def split_final_user(tokenizer, full: list[int]) -> tuple[list[int], list[int]]:
    marker = tokenizer.encode("<|im_start|>user\n", add_special_tokens=False)
    matches = [index for index in range(len(full) - len(marker) + 1) if full[index : index + len(marker)] == marker]
    if not matches:
        raise RuntimeError("final user marker absent")
    return full[: matches[-1]], full[matches[-1] :]


def exact_prefix(tokenizer, messages: list[dict]) -> list[int]:
    full = ids(tokenizer, messages + [{"role": "user", "content": "Boundary probe; no task content."}], True)
    return split_final_user(tokenizer, full)[0]


def token_hash(values: list[int]) -> str:
    digest = hashlib.sha256()
    for value in values:
        digest.update(struct.pack("<i", value))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    manifest = load_jsonl(args.manifest)
    model = contract["base_model"]
    tokenizer_source = args.tokenizer_path or model["repository"]
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        revision=None if args.tokenizer_path else model["revision"],
        trust_remote_code=False,
    )
    label_token_ids = {}
    for label in contract["endpoint"]["labels"]:
        encoded = tokenizer.encode(label, add_special_tokens=False)
        if len(encoded) != 1:
            raise ValueError(f"label {label} is not one token: {encoded}")
        label_token_ids[label] = encoded[0]

    prefix_records = []
    prefix_lookup = {}
    for role in contract["factorial"]["declared_roles"]:
        for condition in contract["factorial"]["governance_conditions"]:
            styles = ["none"] if condition == "fresh" else contract["factorial"]["history_styles"]
            realizations = [0] if condition == "fresh" else range(contract["manifest"]["history_realizations"])
            for style in styles:
                for realization in realizations:
                    history_style = "natural" if style == "none" else style
                    messages = prefix_messages(role, condition, history_style, contract["factorial"]["history_depth"], realization)
                    prefix = exact_prefix(tokenizer, messages)
                    key = (role, condition, style, realization)
                    prefix_lookup[key] = (messages, prefix)
                    prefix_records.append(
                        {
                            "declared_role": role,
                            "condition": condition,
                            "history_style": style,
                            "history_realization": realization,
                            "prefix_tokens": len(prefix),
                            "prefix_sha256_int32_le": token_hash(prefix),
                        }
                    )

    canonical_messages, _ = prefix_lookup[("collaborator", "fresh", "none", 0)]
    suffix_records = []
    max_total = 0
    max_item = None
    for item in manifest:
        for label_swap in contract["endpoint"]["label_swaps"]:
            prompt, _, _ = pairwise_prompt(item, label_swap)
            full = ids(tokenizer, canonical_messages + [{"role": "user", "content": prompt}], True)
            _, suffix = split_final_user(tokenizer, full)
            suffix_records.append((item["item_id"], label_swap, len(suffix), token_hash(suffix)))
            current = len(suffix) + max(record["prefix_tokens"] for record in prefix_records)
            if current > max_total:
                max_total = current
                max_item = item["item_id"]

    probe_item = manifest[0]
    probe_prompt, _, _ = pairwise_prompt(probe_item, 0)
    probe_suffix_hashes = set()
    prefix_identity_errors = []
    for key, (messages, prefix) in prefix_lookup.items():
        full = ids(tokenizer, messages + [{"role": "user", "content": probe_prompt}], True)
        rendered_prefix, suffix = split_final_user(tokenizer, full)
        if rendered_prefix != prefix:
            prefix_identity_errors.append(key)
        probe_suffix_hashes.add(token_hash(suffix))

    length_balance = {}
    for style in contract["factorial"]["history_styles"]:
        lengths = {}
        for condition in ("neutral_length", "verification", "obedience"):
            selected = [
                record["prefix_tokens"]
                for record in prefix_records
                if record["declared_role"] == "collaborator"
                and record["history_style"] == style
                and record["condition"] == condition
            ]
            lengths[condition] = sorted(set(selected))
        flat = [value for values in lengths.values() for value in values]
        length_balance[style] = {
            "prefix_token_counts": lengths,
            "max_relative_range": (max(flat) - min(flat)) / (sum(flat) / len(flat)),
        }
    report = {
        "success": max_total <= model["max_position_embeddings"] and len(probe_suffix_hashes) == 1 and not prefix_identity_errors,
        "contract_sha256": sha256_file(args.contract),
        "manifest_sha256": sha256_file(args.manifest),
        "tokenizer_repository": model["repository"],
        "tokenizer_revision": model["revision"],
        "tokenizer_class": type(tokenizer).__name__,
        "label_token_ids": label_token_ids,
        "prefix_cell_count": len(prefix_records),
        "suffix_cell_count": len(suffix_records),
        "probe_suffix_unique_hash_count_across_all_prefixes": len(probe_suffix_hashes),
        "prefix_identity_errors": [list(value) for value in prefix_identity_errors],
        "max_total_tokens": max_total,
        "max_total_item_id": max_item,
        "context_limit": model["max_position_embeddings"],
        "length_balance": length_balance,
        "prefix_records": prefix_records,
    }
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n", args.allow_overwrite)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
