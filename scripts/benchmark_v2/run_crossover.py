#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v1.histories import prefix_messages
from scripts.benchmark_v1.run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _prefill,
    _split_final_user,
    _suffix_logits_batch,
    _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import _load_model
from scripts.benchmark_v2.crossover import audit_design, enumerate_jobs, job_key, key_hash


def _require_sha_bound_execution(
    *,
    stage: str,
    contract_path: Path,
    manifest_path: Path,
    design_audit_path: Path,
    authorization_path: Path | None,
    expected_key_sha256: str,
    expected_rows: int,
) -> dict | None:
    design_audit = json.loads(design_audit_path.read_text())
    if not design_audit.get("audit", {}).get("success"):
        raise ValueError("crossover design audit did not pass")
    if design_audit["stage"] != stage:
        raise ValueError("design audit stage mismatch")
    if design_audit["contract_sha256"] != sha256_file(contract_path):
        raise ValueError("design audit contract binding mismatch")
    if design_audit["manifest_sha256"] != sha256_file(manifest_path):
        raise ValueError("design audit manifest binding mismatch")
    if design_audit["audit"]["expected_key_sha256"] != expected_key_sha256:
        raise ValueError("design audit key hash mismatch")
    if int(design_audit["audit"]["row_count"]) != expected_rows:
        raise ValueError("design audit row count mismatch")
    if stage == "smoke":
        return None
    if authorization_path is None:
        raise ValueError("non-smoke stages require a SHA-bound execution authorization")
    authorization = json.loads(authorization_path.read_text())
    required = {
        "stage": stage,
        "contract_sha256": sha256_file(contract_path),
        "manifest_sha256": sha256_file(manifest_path),
        "design_audit_sha256": sha256_file(design_audit_path),
        "expected_key_sha256": expected_key_sha256,
        "expected_rows": expected_rows,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"authorization field mismatch: {field}")
    if not authorization.get("execution_allowed"):
        raise ValueError("execution authorization is closed")
    if authorization.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    if stage == "final_test":
        if authorization.get("final_test_open") is not True:
            raise ValueError("final_test remains closed")
        if authorization.get("final_test_open_count") != 1:
            raise ValueError("final_test authorization is not one-time")
    elif authorization.get("final_test_open") not in (None, False):
        raise ValueError("development authorization unexpectedly opens final_test")
    return authorization


def _record_margin(values: dict[str, float], positive: str, negative: str) -> float:
    return float(values[positive]) - float(values[negative])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text())
    if args.stage not in contract["stages"]:
        raise ValueError(f"unknown stage: {args.stage}")
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if sha256_file(args.manifest) != manifest_report["manifest_sha256"]:
        raise ValueError("benchmark manifest hash mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != contract["base_model"]["revision"]:
        raise ValueError("model revision mismatch")
    if not (model_manifest.get("non_quantized") or model_manifest.get("quantization") == "none"):
        raise ValueError("quantized checkpoints are forbidden")

    manifest = load_jsonl(args.manifest)
    jobs = enumerate_jobs(manifest, contract, args.stage)
    design = audit_design(jobs)
    if not design["success"]:
        raise ValueError("in-memory crossover design audit failed")
    expected_key_sha256 = key_hash(jobs)
    expected_rows = int(contract["stages"][args.stage]["expected_rows"])
    authorization = _require_sha_bound_execution(
        stage=args.stage,
        contract_path=args.contract,
        manifest_path=args.manifest,
        design_audit_path=args.design_audit,
        authorization_path=args.execution_authorization,
        expected_key_sha256=expected_key_sha256,
        expected_rows=expected_rows,
    )

    completed = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        completed = {row["job_key"] for row in load_jsonl(args.output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)
    environment = {
        "schema_version": 1,
        "stage": args.stage,
        "device": str(device),
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "contract_sha256": sha256_file(args.contract),
        "manifest_sha256": sha256_file(args.manifest),
        "manifest_report_sha256": sha256_file(args.manifest_report),
        "design_audit_sha256": sha256_file(args.design_audit),
        "authorization_sha256": sha256_file(args.execution_authorization)
        if args.execution_authorization
        else None,
        "expected_key_sha256": expected_key_sha256,
        "planned_rows": expected_rows,
        "label_token_ids": label_ids,
        "batch_size": args.batch_size,
        "batching_strategy": "exact_suffix_length_buckets",
        "final_test_open": bool(authorization and authorization.get("final_test_open")),
        "production_rollout_approved": False,
    }
    if args.environment_output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.environment_output}")
        previous = json.loads(args.environment_output.read_text())
        for field in ("stage", "contract_sha256", "manifest_sha256", "expected_key_sha256"):
            if previous.get(field) != environment.get(field):
                raise ValueError(f"resume environment mismatch: {field}")
    else:
        atomic_write_text(
            args.environment_output,
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
        )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for job in jobs:
        grouped[
            (
                job["declared_role"],
                job["history_condition"],
                job["history_style"],
                job["history_depth"],
                job["history_realization"],
            )
        ].append(job)

    with args.output.open("a", buffering=1) as output_handle:
        for prefix_key in sorted(grouped):
            role, history, style, depth, realization = prefix_key
            messages = prefix_messages(role, history, style, depth, realization)
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache = _prefill(model, device, prefix_ids)
            pending = []
            for job in grouped[prefix_key]:
                current_key = job_key(job)
                if current_key in completed:
                    continue
                prompt_spec = job["prompt"]
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(
                        tokenizer,
                        messages + [{"role": "user", "content": prompt_spec["prompt"]}],
                    ),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"prefix mismatch for {current_key}")
                if len(prefix_ids) + len(suffix_ids) > contract["base_model"]["max_position_embeddings"]:
                    raise ValueError(f"context length exceeded for {current_key}")
                pending.append((job, current_key, suffix_ids))
            by_length: dict[int, list[tuple]] = defaultdict(list)
            for value in pending:
                by_length[len(value[-1])].append(value)
            for suffix_length in sorted(by_length):
                same_length = sorted(by_length[suffix_length], key=lambda value: value[1])
                for offset in range(0, len(same_length), args.batch_size):
                    batch = same_length[offset : offset + args.batch_size]
                    started = time.time()
                    logits, extended_cache = _suffix_logits_batch(
                        model,
                        device,
                        base_cache,
                        len(prefix_ids),
                        [value[-1] for value in batch],
                        label_ids,
                        int(
                            tokenizer.pad_token_id
                            if tokenizer.pad_token_id is not None
                            else tokenizer.eos_token_id
                        ),
                    )
                    latency = time.time() - started
                    for (job, current_key, suffix_ids), (logit_a, logit_b) in zip(batch, logits):
                        spec = job["prompt"]
                        values = {"A": float(logit_a), "B": float(logit_b)}
                        task_margin = _record_margin(
                            values, spec["task_correct_label"], spec["task_foil_label"]
                        )
                        factual_margin = _record_margin(
                            values, spec["factual_correct_label"], spec["factual_foil_label"]
                        )
                        user_margin = _record_margin(
                            values, spec["user_selected_label"], spec["factual_correct_label"]
                        )
                        answer = "A" if logit_a >= logit_b else "B"
                        record = {
                            "job_key": current_key,
                            "stage": args.stage,
                            "benchmark": job["item"]["benchmark"],
                            "item_id": job["item"]["item_id"],
                            "partition": job["item"]["partition"],
                            "declared_role": role,
                            "history_condition": history,
                            "history_style": style,
                            "history_depth": depth,
                            "history_realization": realization,
                            "task_requirement": job["task_requirement"],
                            "target_obedience": spec["target_obedience"],
                            "authority_status": spec["authority_status"],
                            "label_swap": job["label_swap"],
                            "task_correct_label": spec["task_correct_label"],
                            "task_foil_label": spec["task_foil_label"],
                            "factual_correct_label": spec["factual_correct_label"],
                            "factual_foil_label": spec["factual_foil_label"],
                            "user_selected_label": spec["user_selected_label"],
                            "logit_a": logit_a,
                            "logit_b": logit_b,
                            "task_aligned_margin": task_margin,
                            "factual_margin": factual_margin,
                            "user_choice_margin": user_margin,
                            "task_aligned_probability_binary": 1.0
                            / (1.0 + math.exp(max(-700.0, min(700.0, -task_margin)))),
                            "answer": answer,
                            "task_aligned_correct": answer == spec["task_correct_label"],
                            "factual_correct": answer == spec["factual_correct_label"],
                            "followed_user_selection": answer == spec["user_selected_label"],
                            "candidate_payload_sha256": spec["candidate_payload_sha256"],
                            "requirement_rule_sha256": spec["requirement_rule_sha256"],
                            "prompt_sha256": spec["prompt_sha256"],
                            "prefix_tokens": len(prefix_ids),
                            "suffix_tokens": len(suffix_ids),
                            "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                            "latency_s": round(latency / len(batch), 4),
                            "batch_latency_s": round(latency, 4),
                            "batch_size_observed": len(batch),
                            "model_revision": contract["base_model"]["revision"],
                        }
                        numeric = (
                            "logit_a",
                            "logit_b",
                            "task_aligned_margin",
                            "factual_margin",
                            "user_choice_margin",
                        )
                        if not all(math.isfinite(float(record[name])) for name in numeric):
                            raise FloatingPointError(f"non-finite output for {current_key}")
                        output_handle.write(canonical_json(record) + "\n")
                        completed.add(current_key)
                    del extended_cache
            del base_cache
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            elif device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    result = {
        "planned_rows": expected_rows,
        "completed_unique_rows": len(completed),
        "expected_key_sha256": expected_key_sha256,
        "output": str(args.output),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if len(completed) != expected_rows:
        raise SystemExit("crossover run ended without all expected rows")


if __name__ == "__main__":
    main()
