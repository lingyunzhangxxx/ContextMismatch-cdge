#!/usr/bin/env python3
"""Capture true downstream A/B margin gradients for DSGE-V3 fitting."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
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
    _expand_cache_batch,
    _label_ids,
    _split_final_user,
    _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import (
    _load_model,
    _module,
    _normalize_output,
)
from scripts.benchmark_v2.crossover import audit_design, enumerate_jobs, job_key, key_hash


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _selected_jobs(manifest: list[dict], crossover: dict, mode: str) -> list[dict]:
    jobs = enumerate_jobs(manifest, crossover, "replication")
    if mode == "full":
        return jobs
    first_items: dict[str, str] = {}
    for job in sorted(jobs, key=lambda row: (row["item"]["benchmark"], row["item"]["item_id"])):
        first_items.setdefault(job["item"]["benchmark"], job["item"]["item_id"])
    selected = [
        job for job in jobs if first_items[job["item"]["benchmark"]] == job["item"]["item_id"]
    ]
    if len(selected) != 192:
        raise RuntimeError(f"gradient smoke selected {len(selected)} rows, expected 192")
    return selected


def _replace_output(output, value):
    return (value,) + output[1:] if isinstance(output, tuple) else value


def _gradient_safe_prefill(model, device: torch.device, prefix_ids: list[int]):
    """Build a detached cache without creating inference-mode tensors."""
    ids = torch.tensor([prefix_ids], device=device, dtype=torch.long)
    with torch.no_grad():
        output = model(input_ids=ids, use_cache=True, return_dict=True)
    return output.past_key_values


def _suffix_margin_gradient_capture(
    model,
    device: torch.device,
    base_cache,
    prefix_length: int,
    suffix_ids: list[int],
    label_ids: dict[str, int],
    module_keys: list[tuple[int, str]],
    correct_label: str,
    foil_label: str,
) -> tuple[dict[tuple[int, str], torch.Tensor], tuple[float, float]]:
    """Differentiate the true task-aligned selected-token margin at all sites."""
    if correct_label not in {"A", "B"} or foil_label not in {"A", "B"}:
        raise ValueError("gradient labels must be A or B")
    if correct_label == foil_label:
        raise ValueError("gradient correct and foil labels must differ")
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise ValueError("base-model parameter gradients must be disabled")
    captures: dict[tuple[int, str], torch.Tensor] = {}
    handles = []

    def make_hook(key):
        def hook(_module_value, _inputs, output):
            value = _normalize_output(output)
            if value.ndim != 3 or value.shape[0] != 1:
                raise RuntimeError(f"unexpected activation shape for {key}: {tuple(value.shape)}")
            # The first site has no differentiable upstream source because all
            # base parameters are frozen.  Promote that activation to a leaf;
            # later sites remain connected to it and only retain their grads.
            if not value.requires_grad:
                value.requires_grad_(True)
            value.retain_grad()
            captures[key] = value
            return _replace_output(output, value)

        return hook

    for key in module_keys:
        handles.append(_module(model, *key).register_forward_hook(make_hook(key)))
    try:
        ids = torch.tensor([suffix_ids], device=device, dtype=torch.long)
        positions = torch.arange(
            prefix_length,
            prefix_length + len(suffix_ids),
            device=device,
            dtype=torch.long,
        ).unsqueeze(0)
        attention_mask = torch.ones(
            (1, prefix_length + len(suffix_ids)), device=device, dtype=torch.long
        )
        cache = _expand_cache_batch(base_cache, 1)
        kwargs = {
            "input_ids": ids,
            "position_ids": positions,
            "attention_mask": attention_mask,
            "past_key_values": cache,
            "use_cache": False,
            "return_dict": True,
        }
        model.zero_grad(set_to_none=True)
        try:
            output = model(logits_to_keep=1, **kwargs)
        except TypeError:
            output = model(**kwargs)
        logits = output.logits[:, -1].float()
        selected = logits[0, label_ids[correct_label]] - logits[0, label_ids[foil_label]]
        if not bool(torch.isfinite(selected).detach().cpu()):
            raise FloatingPointError("non-finite differentiable task margin")
        selected.backward()
        expected = set(module_keys)
        if set(captures) != expected:
            raise RuntimeError(f"missing activation captures: {sorted(expected - set(captures))}")
        gradients = {}
        for key, value in captures.items():
            if value.grad is None:
                raise RuntimeError(f"missing downstream gradient for {key}")
            gradient = value.grad[0, -1].detach().to(device="cpu", dtype=torch.float16)
            if not bool(torch.isfinite(gradient.float()).all()):
                raise FloatingPointError(f"non-finite downstream gradient for {key}")
            gradients[key] = gradient.contiguous()
        selected_logits = (
            float(logits[0, label_ids["A"]].detach().cpu()),
            float(logits[0, label_ids["B"]].detach().cpu()),
        )
        return gradients, selected_logits
    finally:
        for handle in handles:
            handle.remove()
        model.zero_grad(set_to_none=True)


def _write_shard(
    output_dir: Path,
    name: str,
    *,
    metadata: list[dict],
    gradients: dict[str, list[torch.Tensor]],
    prefix: dict,
) -> dict:
    final = output_dir / name
    incoming = output_dir / f".{name}.incoming"
    if final.exists() or incoming.exists():
        raise FileExistsError(f"refusing existing gradient shard: {final}")
    payload = {
        "schema_version": 1,
        "metadata": metadata,
        "prefix": prefix,
        "task_margin_gradients": {
            key: torch.stack(values, dim=0) for key, values in gradients.items()
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


def _validate_authorization(
    authorization_path: Path,
    *,
    mode: str,
    editor_contract_path: Path,
    crossover_contract_path: Path,
    site_manifest_path: Path,
    manifest_path: Path,
    design_audit_path: Path,
    model_manifest_path: Path,
    expected_rows: int,
    expected_key_sha256: str,
) -> dict:
    authorization = json.loads(authorization_path.read_text())
    required = {
        "stage": f"governance_gradient_capture_{mode}",
        "execution_allowed": True,
        "editor_contract_sha256": sha256_file(editor_contract_path),
        "crossover_contract_sha256": sha256_file(crossover_contract_path),
        "operator_site_manifest_sha256": sha256_file(site_manifest_path),
        "benchmark_manifest_sha256": sha256_file(manifest_path),
        "design_audit_sha256": sha256_file(design_audit_path),
        "model_manifest_sha256": sha256_file(model_manifest_path),
        "expected_rows": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for key, expected in required.items():
        if authorization.get(key) != expected:
            raise ValueError(f"gradient-capture authorization mismatch: {key}")
    code_root = str(authorization.get("code_root", ""))
    if not code_root.startswith("/workspace/context-mismatch-qwen3-8b/code-v"):
        raise ValueError("gradient-capture code root is invalid")
    if int(code_root.rsplit("code-v", 1)[1]) < 29:
        raise ValueError("DSGE-V3 requires code-v29 or newer")
    bundle = Path(code_root) / "bundle.sha256"
    if not bundle.is_file():
        raise ValueError("gradient-capture immutable bundle manifest is missing")
    if authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(bundle):
        raise ValueError("gradient-capture bundle SHA mismatch")
    return authorization


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
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing gradient output: {args.output_dir}")

    editor = json.loads(args.editor_contract.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    sites_manifest = json.loads(args.operator_site_manifest.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    design_audit = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if editor.get("status") != "design_locked_after_v2_terminal_failure_audit_before_any_v3_forward":
        raise ValueError("DSGE-V3 contract is not locked")
    if int(editor.get("code_version_minimum", 0)) != 29:
        raise ValueError("unexpected DSGE-V3 code minimum")
    if editor.get("final_test_open") or editor.get("final_test_open_count") != 0:
        raise ValueError("DSGE-V3 contract unexpectedly opens the final test")
    if editor.get("production_rollout_approved"):
        raise ValueError("DSGE-V3 contract unexpectedly approves production")
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if design_audit.get("stage") != "replication" or not design_audit.get("audit", {}).get(
        "success"
    ):
        raise ValueError("replication design audit is invalid")
    if design_audit.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("replication design contract SHA mismatch")
    if design_audit.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("replication design manifest SHA mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model revision mismatch")

    selected_sites = sorted(
        (int(site["layer"]), str(site["component"]))
        for site in sites_manifest["selected_sites_ordered"]
    )
    candidate_sites = sorted(
        (int(site["layer"]), str(site["component"]))
        for site in editor["candidate_sites"]
    )
    if selected_sites != candidate_sites:
        raise ValueError("DSGE-V3 candidate sites differ from the locked operator manifest")

    manifest = load_jsonl(args.manifest)
    jobs = _selected_jobs(manifest, crossover, args.mode)
    if not audit_design(jobs)["success"]:
        raise ValueError("in-memory gradient design audit failed")
    expected_rows = 192 if args.mode == "smoke" else 6144
    expected_key_sha256 = key_hash(jobs)
    _validate_authorization(
        args.execution_authorization,
        mode=args.mode,
        editor_contract_path=args.editor_contract,
        crossover_contract_path=args.crossover_contract,
        site_manifest_path=args.operator_site_manifest,
        manifest_path=args.manifest,
        design_audit_path=args.design_audit,
        model_manifest_path=args.model_manifest,
        expected_rows=expected_rows,
        expected_key_sha256=expected_key_sha256,
    )

    args.output_dir.mkdir(parents=True)
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze all base-model parameters")
    label_ids = _label_ids(tokenizer)
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

    shard_records = []
    observed_keys = []
    nonzero_counts = {f"{layer}:{component}": 0 for layer, component in candidate_sites}
    for prefix_key in sorted(grouped):
        role, history, style, depth, realization = prefix_key
        messages = prefix_messages(role, history, style, depth, realization)
        prefix_ids = _exact_prefix(tokenizer, messages)
        base_cache = _gradient_safe_prefill(model, device, prefix_ids)
        metadata = []
        gradients = {f"{layer}:{component}": [] for layer, component in candidate_sites}
        for job in sorted(grouped[prefix_key], key=job_key):
            current_key = job_key(job)
            prompt = job["prompt"]
            rendered_prefix, suffix_ids = _split_final_user(
                tokenizer,
                _chat_ids(tokenizer, messages + [{"role": "user", "content": prompt["prompt"]}]),
            )
            if rendered_prefix != prefix_ids:
                raise RuntimeError(f"prefix mismatch for {current_key}")
            current_gradients, logits = _suffix_margin_gradient_capture(
                model,
                device,
                base_cache,
                len(prefix_ids),
                suffix_ids,
                label_ids,
                candidate_sites,
                prompt["task_correct_label"],
                prompt["task_foil_label"],
            )
            values = {"A": logits[0], "B": logits[1]}
            task_margin = values[prompt["task_correct_label"]] - values[prompt["task_foil_label"]]
            if not math.isfinite(task_margin):
                raise FloatingPointError(f"non-finite gradient-capture margin for {current_key}")
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
                "logit_a": values["A"],
                "logit_b": values["B"],
                "task_aligned_margin": task_margin,
                "prefix_tokens": len(prefix_ids),
                "suffix_tokens": len(suffix_ids),
                "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
            }
            metadata.append(row)
            observed_keys.append(current_key)
            for site, gradient in current_gradients.items():
                key = f"{site[0]}:{site[1]}"
                gradients[key].append(gradient)
                if float(torch.linalg.vector_norm(gradient.float())) > 0.0:
                    nonzero_counts[key] += 1
        prefix = {
            "declared_role": role,
            "history_condition": history,
            "history_style": style,
            "history_depth": depth,
            "history_realization": realization,
        }
        name = f"margin_gradients__{role}__{history}__{style}__r{realization}.pt"
        shard_records.append(
            _write_shard(
                args.output_dir,
                name,
                metadata=metadata,
                gradients=gradients,
                prefix=prefix,
            )
        )
        del base_cache
        gc.collect()
        if device.type == "npu" and hasattr(torch, "npu"):
            torch.npu.empty_cache()

    if len(observed_keys) != expected_rows or len(set(observed_keys)) != expected_rows:
        raise RuntimeError("gradient capture row identity is incomplete or duplicated")
    observed_sha = hashlib.sha256(
        ("\n".join(sorted(observed_keys)) + "\n").encode("utf-8")
    ).hexdigest()
    if observed_sha != expected_key_sha256:
        raise RuntimeError("gradient capture key SHA mismatch")
    nonzero_fraction = {
        key: count / expected_rows for key, count in sorted(nonzero_counts.items())
    }
    minimum_nonzero = float(
        editor["downstream_gradient_capture"]["nonzero_gradient_fraction_minimum_per_site"]
    )
    if any(value < minimum_nonzero for value in nonzero_fraction.values()):
        raise RuntimeError("one or more sites failed the nonzero-gradient fraction gate")
    manifest_value = {
        "schema_version": 1,
        "stage": f"governance_gradient_capture_{args.mode}",
        "partition": "subspace_fit",
        "rows": expected_rows,
        "unique_job_keys": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "observed_key_sha256": observed_sha,
        "shard_count": len(shard_records),
        "sites": editor["candidate_sites"],
        "nonzero_gradient_fraction_by_site": nonzero_fraction,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "shards": shard_records,
        "complete": True,
        "base_model_parameter_gradients": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "margin_gradient_manifest.json",
        json.dumps(manifest_value, indent=2, sort_keys=True) + "\n",
    )
    atomic_write_text(
        args.output_dir / "environment.json",
        json.dumps(
            {
                "device": str(device),
                "torch_version": torch.__version__,
                "model_class": type(model).__name__,
                "label_token_ids": label_ids,
                "base_model_parameter_gradients": False,
                "final_test_open": False,
                "final_test_open_count": 0,
                "production_rollout_approved": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )
    print(_canonical(manifest_value))


if __name__ == "__main__":
    main()
