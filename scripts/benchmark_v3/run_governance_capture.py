#!/usr/bin/env python3
"""Capture bidirectional crossover states for the adaptive governance editor."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v1.histories import prefix_messages
from scripts.benchmark_v1.run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _split_final_user,
    _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import _load_model
from scripts.benchmark_v1.run_protected_capture import _prefill_with_last_boundary_capture
from scripts.benchmark_v1.run_subspace_capture import _suffix_component_capture
from scripts.benchmark_v2.crossover import (
    audit_design,
    enumerate_jobs,
    job_key,
    key_hash,
)


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _selected_jobs(
    manifest: list[dict], crossover: dict, mode: str, crossover_stage: str = "replication"
) -> list[dict]:
    jobs = enumerate_jobs(manifest, crossover, crossover_stage)
    if mode == "full":
        return jobs
    first_items: dict[str, str] = {}
    for job in sorted(jobs, key=lambda row: (row["item"]["benchmark"], row["item"]["item_id"])):
        first_items.setdefault(job["item"]["benchmark"], job["item"]["item_id"])
    selected = [
        job for job in jobs if first_items[job["item"]["benchmark"]] == job["item"]["item_id"]
    ]
    if len(selected) != 192:
        raise RuntimeError(f"smoke capture selected {len(selected)} rows, expected 192")
    return selected


def _site_key(layer: int, component: str) -> str:
    return f"{layer}:{component}"


def _validate_authorization(
    path: Path,
    *,
    mode: str,
    editor_contract: Path,
    crossover_contract: Path,
    site_manifest: Path,
    manifest: Path,
    design_audit: Path,
    expected_rows: int,
    expected_key_sha256: str,
    authorization_stage: str,
    router_contract: Path | None,
) -> dict:
    authorization = json.loads(path.read_text())
    required = {
        "schema_version": 1,
        "stage": authorization_stage,
        "execution_allowed": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "expected_rows": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "editor_contract_sha256": sha256_file(editor_contract),
        "crossover_contract_sha256": sha256_file(crossover_contract),
        "operator_site_manifest_sha256": sha256_file(site_manifest),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "design_audit_sha256": sha256_file(design_audit),
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"capture authorization mismatch: {field}")
    if router_contract is not None:
        router = json.loads(router_contract.read_text())
        if router.get("status") != "frozen_before_any_v5_capture_or_fit":
            raise ValueError("V5 router contract is not frozen")
        if authorization.get("router_contract_sha256") != sha256_file(router_contract):
            raise ValueError("capture authorization V5 router-contract SHA mismatch")
    code_root = authorization.get("code_root", "")
    if not code_root.startswith("/workspace/context-mismatch-qwen3-8b/code-v"):
        raise ValueError("capture authorization code root is invalid")
    bundle = Path(code_root) / "bundle.sha256"
    if bundle.exists() and authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(bundle):
        raise ValueError("capture authorization bundle SHA mismatch")
    return authorization


def _write_shard(
    output_dir: Path,
    name: str,
    *,
    metadata: list[dict],
    boundary_states: dict[str, torch.Tensor],
    component_inputs: dict[str, list[torch.Tensor]],
    component_outputs: dict[str, list[torch.Tensor]],
    prefix: dict,
) -> dict:
    final = output_dir / name
    incoming = output_dir / f".{name}.incoming"
    if final.exists() or incoming.exists():
        raise FileExistsError(f"refusing existing capture shard: {final}")
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "prefix": prefix,
        "boundary_states": boundary_states,
        "component_inputs": {
            key: torch.stack(values, dim=0) for key, values in component_inputs.items()
        },
        "component_outputs": {
            key: torch.stack(values, dim=0) for key, values in component_outputs.items()
        },
    }
    torch.save(payload, incoming)
    os.replace(incoming, final)
    return {
        "file": name,
        "rows": len(metadata),
        "sha256": sha256_file(final),
        **prefix,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument(
        "--crossover-stage",
        choices=["discovery", "replication"],
        default="replication",
    )
    parser.add_argument("--authorization-stage")
    parser.add_argument("--router-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()

    for output in (args.output_dir,):
        if output.exists():
            raise FileExistsError(f"refusing existing output directory: {output}")
    editor = json.loads(args.editor_contract.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    sites_manifest = json.loads(args.operator_site_manifest.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    design_audit = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if editor.get("status") != "frozen_before_any_code_v22_forward":
        raise ValueError("adaptive editor contract is not frozen")
    if editor.get("final_test_open") or editor.get("production_rollout_approved"):
        raise ValueError("editor contract unexpectedly opens final or production")
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if design_audit.get("stage") != args.crossover_stage or not design_audit.get("audit", {}).get("success"):
        raise ValueError(f"{args.crossover_stage} design audit is not valid")
    if design_audit.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("replication design contract SHA mismatch")
    if design_audit.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("replication design manifest SHA mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model revision mismatch")
    selected = {
        (int(site["layer"]), site["component"])
        for site in sites_manifest["selected_sites_ordered"]
    }
    locked = {
        (int(site["layer"]), site["component"])
        for site in editor["locked_sites"]
    }
    if selected != locked:
        raise ValueError("editor sites do not match the locked operator sites")
    if editor["bound_inputs"]["operator_site_manifest_sha256"] != sha256_file(args.operator_site_manifest):
        raise ValueError("editor contract site-manifest binding mismatch")

    manifest = load_jsonl(args.manifest)
    jobs = _selected_jobs(manifest, crossover, args.mode, args.crossover_stage)
    design = audit_design(jobs)
    if not design["success"]:
        raise ValueError("in-memory capture design audit failed")
    expected_rows = 192 if args.mode == "smoke" else 6144
    expected_key_sha256 = key_hash(jobs)
    authorization_stage = args.authorization_stage or f"governance_capture_{args.mode}"
    _validate_authorization(
        args.execution_authorization,
        mode=args.mode,
        editor_contract=args.editor_contract,
        crossover_contract=args.crossover_contract,
        site_manifest=args.operator_site_manifest,
        manifest=args.manifest,
        design_audit=args.design_audit,
        expected_rows=expected_rows,
        expected_key_sha256=expected_key_sha256,
        authorization_stage=authorization_stage,
        router_contract=args.router_contract,
    )

    args.output_dir.mkdir(parents=True)
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)
    ordered_sites = sorted(locked)
    layers = sorted({layer for layer, _ in ordered_sites})
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

    records = []
    observed_keys = []
    for prefix_key in sorted(grouped):
        role, history, style, depth, realization = prefix_key
        messages = prefix_messages(role, history, style, depth, realization)
        prefix_ids = _exact_prefix(tokenizer, messages)
        base_cache, boundary_states = _prefill_with_last_boundary_capture(
            model, device, prefix_ids, layers
        )
        metadata = []
        inputs = {_site_key(*site): [] for site in ordered_sites}
        outputs = {_site_key(*site): [] for site in ordered_sites}
        for job in sorted(grouped[prefix_key], key=job_key):
            current_key = job_key(job)
            prompt = job["prompt"]
            rendered_prefix, suffix_ids = _split_final_user(
                tokenizer,
                _chat_ids(tokenizer, messages + [{"role": "user", "content": prompt["prompt"]}]),
            )
            if rendered_prefix != prefix_ids:
                raise RuntimeError(f"prefix mismatch for {current_key}")
            cache = copy.deepcopy(base_cache)
            captured_inputs, captured_outputs, logits, extended_cache = _suffix_component_capture(
                model,
                device,
                cache,
                len(prefix_ids),
                suffix_ids,
                label_ids,
                ordered_sites,
            )
            values = {"A": float(logits[0]), "B": float(logits[1])}
            task_margin = values[prompt["task_correct_label"]] - values[prompt["task_foil_label"]]
            factual_margin = values[prompt["factual_correct_label"]] - values[prompt["factual_foil_label"]]
            row = {
                "job_key": current_key,
                "benchmark": job["item"]["benchmark"],
                "item_id": job["item"]["item_id"],
                "partition": job["item"]["partition"],
                "declared_role": role,
                "history_condition": history,
                "history_style": style,
                "history_depth": depth,
                "history_realization": realization,
                "task_requirement": job["task_requirement"],
                "target_obedience": prompt["target_obedience"],
                "label_swap": job["label_swap"],
                "task_correct_label": prompt["task_correct_label"],
                "task_foil_label": prompt["task_foil_label"],
                "factual_correct_label": prompt["factual_correct_label"],
                "factual_foil_label": prompt["factual_foil_label"],
                "logit_a": values["A"],
                "logit_b": values["B"],
                "task_aligned_margin": task_margin,
                "factual_margin": factual_margin,
                "prefix_tokens": len(prefix_ids),
                "suffix_tokens": len(suffix_ids),
                "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
            }
            if not all(
                math.isfinite(float(row[field]))
                for field in ("logit_a", "logit_b", "task_aligned_margin", "factual_margin")
            ):
                raise FloatingPointError(f"non-finite capture output for {current_key}")
            metadata.append(row)
            observed_keys.append(current_key)
            for site in ordered_sites:
                key = _site_key(*site)
                inputs[key].append(captured_inputs[site])
                outputs[key].append(captured_outputs[site])
            del cache, extended_cache
        prefix = {
            "declared_role": role,
            "history_condition": history,
            "history_style": style,
            "history_depth": depth,
            "history_realization": realization,
        }
        name = f"capture__{role}__{history}__{style}__r{realization}.pt"
        records.append(
            _write_shard(
                args.output_dir,
                name,
                metadata=metadata,
                boundary_states=boundary_states,
                component_inputs=inputs,
                component_outputs=outputs,
                prefix=prefix,
            )
        )
        del base_cache
        gc.collect()
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()

    if len(observed_keys) != expected_rows or len(set(observed_keys)) != expected_rows:
        raise RuntimeError("capture row identity is incomplete or duplicated")
    observed_sha = __import__("hashlib").sha256(
        ("\n".join(sorted(observed_keys)) + "\n").encode("utf-8")
    ).hexdigest()
    if observed_sha != expected_key_sha256:
        raise RuntimeError("capture key SHA mismatch")
    pair_groups: dict[tuple, set[str]] = defaultdict(set)
    for job in jobs:
        pair_groups[
            (
                job["item"]["item_id"],
                job["declared_role"],
                job["history_style"],
                job["history_realization"],
                job["task_requirement"],
                job["label_swap"],
            )
        ].add(job["history_condition"])
    if any(values != {"verification", "obedience"} for values in pair_groups.values()):
        raise RuntimeError("capture counterfactual history pairs are incomplete")
    partitions = {str(job["item"]["partition"]) for job in jobs}
    if len(partitions) != 1:
        raise RuntimeError(f"capture spans multiple partitions: {sorted(partitions)}")
    capture_partition = next(iter(partitions))
    capture_manifest = {
        "schema_version": 1,
        "stage": authorization_stage,
        "crossover_stage": args.crossover_stage,
        "partition": capture_partition,
        "rows": expected_rows,
        "unique_job_keys": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "observed_key_sha256": observed_sha,
        "counterfactual_pairs": len(pair_groups),
        "shard_count": len(records),
        "sites": editor["locked_sites"],
        "hidden_size": int(crossover.get("base_model", {}).get("hidden_size", 4096)),
        "capture_position": "last_decision_token",
        "dtype": "float16_capture_from_bfloat16_forward",
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "router_contract_sha256": (
            sha256_file(args.router_contract) if args.router_contract is not None else None
        ),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "shards": records,
        "complete": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "capture_manifest.json",
        json.dumps(capture_manifest, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        args.output_dir / "environment.json",
        json.dumps(
            {
                "device": str(device),
                "torch_version": torch.__version__,
                "model_class": type(model).__name__,
                "label_token_ids": label_ids,
                "production_rollout_approved": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(_canonical(capture_manifest))


if __name__ == "__main__":
    main()
