#!/usr/bin/env python3
"""Pressure-free pairwise competence control on the frozen six-benchmark manifest."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import time
from pathlib import Path

import torch

from .common import canonical_json, load_jsonl, sha256_file
from .histories import pairwise_no_pressure_prompt, prefix_messages
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
from .run_mechanism_discovery import _load_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--control-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()

    contract = json.loads(args.control_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if sha256_file(args.manifest) != contract["benchmark_manifest_sha256"]:
        raise ValueError("competence-control manifest SHA mismatch")
    if manifest_report["manifest_sha256"] != contract["benchmark_manifest_sha256"]:
        raise ValueError("manifest report mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest["revision"] != contract["base_model_revision"]:
        raise ValueError("model revision mismatch")
    if args.output.exists() or args.environment_output.exists():
        raise FileExistsError("refusing to overwrite competence-control output")
    manifest = load_jsonl(args.manifest)
    if len(manifest) != contract["items"]:
        raise ValueError("competence-control item count mismatch")
    expected = len(manifest) * len(contract["declared_roles"]) * len(contract["label_swaps"])
    if expected != contract["expected_rows"]:
        raise ValueError("competence-control expected row derivation changed")
    expected_keys = {
        "__".join((item["item_id"], role, str(label_swap)))
        for item in manifest
        for role in contract["declared_roles"]
        for label_swap in contract["label_swaps"]
    }
    expected_key_sha256 = hashlib.sha256(
        "".join(f"{key}\n" for key in sorted(expected_keys)).encode("utf-8")
    ).hexdigest()

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)
    environment = {
        "device": str(device),
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "model_revision": contract["base_model_revision"],
        "control_contract_sha256": sha256_file(args.control_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "label_token_ids": label_ids,
        "planned_rows": expected,
        "expected_job_key_set_sha256": expected_key_sha256,
        "batch_size": 1,
        "production_rollout_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.environment_output.write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n")
    completed = 0
    output_handle = args.output.open("x", buffering=1)
    for role in contract["declared_roles"]:
        messages = prefix_messages(role, "fresh", "natural", 0, 0)
        prefix_ids = _exact_prefix(tokenizer, messages)
        base_cache = _prefill(model, device, prefix_ids)
        for item in sorted(manifest, key=lambda row: (row["benchmark"], row["item_id"])):
            for label_swap in contract["label_swaps"]:
                prompt, correct_label, foil_label = pairwise_no_pressure_prompt(item, label_swap)
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(tokenizer, messages + [{"role": "user", "content": prompt}]),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError("competence-control prefix mismatch")
                cache = copy.deepcopy(base_cache)
                started = time.time()
                logits, extended_cache = _suffix_logits_batch(
                    model,
                    device,
                    cache,
                    len(prefix_ids),
                    [suffix_ids],
                    label_ids,
                    pad_token_id=0,
                )
                logit_a, logit_b = logits[0]
                values = {"A": logit_a, "B": logit_b}
                margin = values[correct_label] - values[foil_label]
                answer = "A" if logit_a >= logit_b else "B"
                record = {
                    "job_key": "__".join((item["item_id"], role, str(label_swap))),
                    "condition": "pressure_free_pairwise",
                    "benchmark": item["benchmark"],
                    "item_id": item["item_id"],
                    "partition": item["partition"],
                    "declared_role": role,
                    "label_swap": label_swap,
                    "correct_label": correct_label,
                    "foil_label": foil_label,
                    "logit_a": logit_a,
                    "logit_b": logit_b,
                    "correct_logit_margin": margin,
                    "answer": answer,
                    "correct": answer == correct_label,
                    "prefix_tokens": len(prefix_ids),
                    "suffix_tokens": len(suffix_ids),
                    "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                    "latency_s": round(time.time() - started, 4),
                    "model_revision": contract["base_model_revision"],
                }
                if not all(math.isfinite(record[key]) for key in ("logit_a", "logit_b", "correct_logit_margin")):
                    raise FloatingPointError("non-finite competence-control row")
                output_handle.write(canonical_json(record) + "\n")
                completed += 1
                del cache, extended_cache
        del base_cache
        gc.collect()
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()
    output_handle.close()
    if completed != expected:
        raise RuntimeError(f"competence-control incomplete: {completed} != {expected}")
    print(json.dumps({"planned_rows": expected, "completed_rows": completed}))


if __name__ == "__main__":
    main()
