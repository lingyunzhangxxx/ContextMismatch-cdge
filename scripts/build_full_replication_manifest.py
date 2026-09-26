#!/usr/bin/env python3
"""Freeze all prompts and CUDA baselines for the Ascend full mechanism runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return rows


def signed_margin(row: dict[str, object]) -> float:
    logit_a = float(row["logit_a"])
    logit_b = float(row["logit_b"])
    return logit_a - logit_b if row["correct_label"] == "A" else logit_b - logit_a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if contract["status"] != "frozen_no_submission":
        raise RuntimeError("unexpected frozen-contract status")
    if contract["production_rollout_approved"] is not False:
        raise RuntimeError("production boundary changed")
    selection = contract["shared_selection"]
    regimes = [str(value) for value in selection["regimes"]]
    realizations = [int(value) for value in selection["history_realizations"]]
    tasks = [str(value) for value in selection["tasks"]]
    label_swaps = [int(value) for value in selection["label_swaps"]]
    order_swaps = [int(value) for value in selection["evidence_order_swaps"]]
    expected_samples = (
        len(regimes)
        * len(realizations)
        * len(tasks)
        * len(label_swaps)
        * len(order_swaps)
    )

    def selected(row: dict[str, object]) -> bool:
        return (
            row["history_style"] == selection["history_style"]
            and int(row["history_depth"]) == int(selection["history_depth"])
            and int(row["history_realization"]) in realizations
            and str(row["task_id"]) in tasks
            and int(row["label_swap"]) in label_swaps
            and int(row["evidence_order_swap"]) in order_swaps
            and str(row["regime"]) in regimes
        )

    full_rows = [row for row in read_jsonl(args.metadata) if selected(row)]
    if len(full_rows) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} full rows, observed {len(full_rows)}"
        )
    full_index: dict[tuple[object, ...], dict[str, object]] = {}
    for row in full_rows:
        key = (
            str(row["regime"]),
            int(row["history_realization"]),
            str(row["task_id"]),
            int(row["label_swap"]),
            int(row["evidence_order_swap"]),
        )
        if key in full_index:
            raise RuntimeError(f"duplicate full baseline: {key}")
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise RuntimeError(f"missing exact messages: {key}")
        if not all(
            math.isfinite(float(row[name])) for name in ("logit_a", "logit_b")
        ):
            raise RuntimeError(f"non-finite full baseline: {key}")
        full_index[key] = row

    expected_keys = {
        (regime, realization, task, label_swap, order_swap)
        for regime in regimes
        for realization in realizations
        for task in tasks
        for label_swap in label_swaps
        for order_swap in order_swaps
    }
    if set(full_index) != expected_keys:
        missing = sorted(expected_keys - set(full_index))
        extra = sorted(set(full_index) - expected_keys)
        raise RuntimeError(f"full factorial mismatch: missing={missing} extra={extra}")

    cached_rows = [
        row
        for row in read_jsonl(args.cached)
        if selected(row)
        and float(row["patch_coefficient"]) == 0.0
        and row["target_source"] == selection["target_source"]
    ]
    residual_layers = [int(value) for value in contract["residual_run"]["layers"]]
    cached_by_key: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in cached_rows:
        key = (
            str(row["regime"]),
            int(row["history_realization"]),
            str(row["task_id"]),
            int(row["label_swap"]),
            int(row["evidence_order_swap"]),
        )
        cached_by_key.setdefault(key, []).append(row)
    if set(cached_by_key) != expected_keys:
        raise RuntimeError("cached and full sample keys differ")

    cached_index: dict[tuple[object, ...], dict[str, object]] = {}
    for key, rows in cached_by_key.items():
        observed_layers = sorted(int(row["layer"]) for row in rows)
        if observed_layers != residual_layers:
            raise RuntimeError(
                f"coefficient-0 layer coverage mismatch for {key}: {observed_layers}"
            )
        observed_logits = {
            (float(row["logit_a"]), float(row["logit_b"])) for row in rows
        }
        if len(observed_logits) != 1:
            raise RuntimeError(f"coefficient-0 layer variance for {key}")
        cached_index[key] = min(rows, key=lambda row: int(row["layer"]))

    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    expected_label_ids = environment.get("label_token_ids")
    if expected_label_ids != {"A": 32, "B": 33}:
        raise RuntimeError(f"unexpected CUDA label IDs: {expected_label_ids}")

    samples = []
    for realization in realizations:
        for task in tasks:
            for label_swap in label_swaps:
                for order_swap in order_swaps:
                    for regime in regimes:
                        key = (regime, realization, task, label_swap, order_swap)
                        row = full_index[key]
                        cached = cached_index[key]
                        samples.append(
                            {
                                "key": row["key"],
                                "regime": regime,
                                "history_style": row["history_style"],
                                "history_depth": row["history_depth"],
                                "history_realization": realization,
                                "task_id": task,
                                "domain": row["domain"],
                                "label_swap": label_swap,
                                "evidence_order_swap": order_swap,
                                "correct_label": row["correct_label"],
                                "foil_label": row["foil_label"],
                                "messages": row["messages"],
                                "messages_canonical_sha256": canonical_sha(
                                    row["messages"]
                                ),
                                "cuda_prompt": {
                                    "token_count": int(row["prompt_tokens"]),
                                    "token_sha256_int32_le": row["prompt_sha256"],
                                },
                                "cuda_full": {
                                    "logit_a": row["logit_a"],
                                    "logit_b": row["logit_b"],
                                    "correct_margin": signed_margin(row),
                                    "decision": row["answer"],
                                },
                                "cuda_cached": {
                                    "logit_a": cached["logit_a"],
                                    "logit_b": cached["logit_b"],
                                    "correct_margin": signed_margin(cached),
                                    "decision": cached["answer"],
                                    "invariant_layers": residual_layers,
                                },
                            }
                        )

    manifest = {
        "manifest_version": "context-mismatch-full-replication-manifest-v1",
        "production_rollout_approved": False,
        "contract_version": contract["contract_version"],
        "contract_sha256": sha256_file(args.contract),
        "expected_label_token_ids": expected_label_ids,
        "sample_count": len(samples),
        "sources": {
            "activation_metadata": {
                "path": args.metadata.as_posix(),
                "sha256": sha256_file(args.metadata),
            },
            "cached_patching": {
                "path": args.cached.as_posix(),
                "sha256": sha256_file(args.cached),
            },
            "cuda_environment": {
                "path": args.environment.as_posix(),
                "sha256": sha256_file(args.environment),
                "torch": environment.get("torch_version"),
                "transformers": environment.get("transformers_version"),
            },
        },
        "samples": samples,
    }
    if len(samples) != expected_samples:
        raise RuntimeError("manifest construction lost samples")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "samples": len(samples),
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
