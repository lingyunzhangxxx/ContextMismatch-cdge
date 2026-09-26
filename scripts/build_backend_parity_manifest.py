#!/usr/bin/env python3
"""Freeze the CUDA baselines and exact prompts for the Ascend parity gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def signed_margin(row: dict[str, object]) -> float:
    a = float(row["logit_a"])
    b = float(row["logit_b"])
    return a - b if row["correct_label"] == "A" else b - a


def canonical_sha(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--cached", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    selection = contract["selection"]
    expected_samples = int(selection["expected_samples"])
    regimes = set(selection["regimes"])

    def selected(row: dict[str, object]) -> bool:
        return (
            row["history_style"] == selection["history_style"]
            and int(row["history_depth"]) == int(selection["history_depth"])
            and int(row["history_realization"])
            == int(selection["history_realization"])
            and int(row["label_swap"]) == int(selection["label_swap"])
            and int(row["evidence_order_swap"])
            == int(selection["evidence_order_swap"])
            and row["regime"] in regimes
        )

    full_rows = [row for row in jsonl(args.metadata) if selected(row)]
    if len(full_rows) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} full rows, observed {len(full_rows)}"
        )
    full_index = {}
    for row in full_rows:
        key = (str(row["task_id"]), str(row["regime"]))
        if key in full_index:
            raise RuntimeError(f"duplicate full baseline: {key}")
        if not isinstance(row.get("messages"), list) or not row["messages"]:
            raise RuntimeError(f"missing exact messages: {key}")
        if not math.isfinite(float(row["logit_a"])) or not math.isfinite(
            float(row["logit_b"])
        ):
            raise RuntimeError(f"non-finite full baseline: {key}")
        full_index[key] = row

    cached_candidates = [
        row
        for row in jsonl(args.cached)
        if selected(row)
        and float(row["patch_coefficient"]) == 0.0
        and row["target_source"]
        == contract["cached_cuda_baseline"]["required_target_source"]
    ]
    cached_by_key: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in cached_candidates:
        key = (str(row["task_id"]), str(row["regime"]))
        cached_by_key.setdefault(key, []).append(row)
    if set(cached_by_key) != set(full_index):
        raise RuntimeError("cached and full sample keys differ")

    cached_index = {}
    for key, rows in cached_by_key.items():
        observed = {
            (float(row["logit_a"]), float(row["logit_b"])) for row in rows
        }
        if len(observed) != 1:
            raise RuntimeError(f"coefficient-0 layer variance for {key}: {observed}")
        if len({int(row["layer"]) for row in rows}) < 2:
            raise RuntimeError(f"insufficient layers to establish invariance: {key}")
        cached_index[key] = min(rows, key=lambda row: int(row["layer"]))

    environment = json.loads(args.environment.read_text(encoding="utf-8"))
    expected_label_ids = environment.get("label_token_ids")
    if expected_label_ids != {"A": 32, "B": 33}:
        raise RuntimeError(f"unexpected CUDA label IDs: {expected_label_ids}")

    samples = []
    for row in full_rows:
        key = (str(row["task_id"]), str(row["regime"]))
        cached = cached_index[key]
        full_margin = signed_margin(row)
        cached_margin = signed_margin(cached)
        samples.append(
            {
                "key": row["key"],
                "task_id": row["task_id"],
                "domain": row["domain"],
                "regime": row["regime"],
                "history_style": row["history_style"],
                "history_depth": row["history_depth"],
                "history_realization": row["history_realization"],
                "label_swap": row["label_swap"],
                "evidence_order_swap": row["evidence_order_swap"],
                "correct_label": row["correct_label"],
                "foil_label": row["foil_label"],
                "messages": row["messages"],
                "messages_canonical_sha256": canonical_sha(row["messages"]),
                "cuda_prompt": {
                    "token_count": row["prompt_tokens"],
                    "token_sha256_int32_le": row["prompt_sha256"],
                },
                "cuda_full": {
                    "logit_a": row["logit_a"],
                    "logit_b": row["logit_b"],
                    "correct_margin": full_margin,
                    "decision": row["answer"],
                },
                "cuda_cached": {
                    "logit_a": cached["logit_a"],
                    "logit_b": cached["logit_b"],
                    "correct_margin": cached_margin,
                    "decision": cached["answer"],
                    "coefficient": cached["patch_coefficient"],
                    "invariant_layers": sorted(
                        int(candidate["layer"])
                        for candidate in cached_by_key[key]
                    ),
                },
            }
        )

    task_ids = {str(sample["task_id"]) for sample in samples}
    if len(task_ids) != int(selection["task_count"]):
        raise RuntimeError(f"expected six tasks, observed {sorted(task_ids)}")
    for task_id in task_ids:
        observed = {
            str(sample["regime"])
            for sample in samples
            if sample["task_id"] == task_id
        }
        if observed != regimes:
            raise RuntimeError(f"regime imbalance for {task_id}: {observed}")

    manifest = {
        "manifest_version": "context-mismatch-backend-parity-manifest-v1",
        "production_rollout_approved": False,
        "contract": contract,
        "contract_sha256": sha256_file(args.contract),
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
        "expected_label_token_ids": expected_label_ids,
        "sample_count": len(samples),
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "samples": len(samples),
                "tasks": sorted(task_ids),
                "sha256": sha256_file(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
