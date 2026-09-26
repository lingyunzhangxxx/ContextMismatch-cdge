#!/usr/bin/env python3
"""Capture split-isolated protected states for Qwen3-8B operator fitting."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import torch

from .common import atomic_write_text, load_jsonl, sha256_file
from .controls import (
    factual_memory_messages,
    factual_memory_prompt,
    load_controls,
    supported_authority_prompt,
)
from .histories import pairwise_prompt, prefix_messages, system_message
from .run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _prefill,
    _split_final_user,
    _token_hash,
)
from .run_mechanism_discovery import _layers, _load_model, _normalize_output
from .run_subspace_capture import _margin, _suffix_component_capture


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def _prefill_with_last_boundary_capture(
    model,
    device: torch.device,
    prefix_ids: list[int],
    layers: list[int],
):
    captures = {}
    handles = []

    def make_hook(layer):
        def hook(_module_value, _inputs, output):
            value = _normalize_output(output)
            captures[str(layer)] = (
                value[0, -1].detach().to(device="cpu", dtype=torch.float16).contiguous()
            )

        return hook

    for layer in layers:
        handles.append(_layers(model)[layer].register_forward_hook(make_hook(layer)))
    try:
        cache = _prefill(model, device, prefix_ids)
        if set(captures) != {str(layer) for layer in layers}:
            raise RuntimeError("protected boundary capture is incomplete")
        return cache, captures
    finally:
        for handle in handles:
            handle.remove()


def _write_shard(
    *,
    output_dir: Path,
    shard_name: str,
    metadata: list[dict],
    boundary_metadata: dict,
    boundary_states: dict[str, torch.Tensor],
    input_values: dict[str, list[torch.Tensor]],
    output_values: dict[str, list[torch.Tensor]],
) -> dict:
    shard_path = output_dir / shard_name
    incoming = output_dir / f".{shard_name}.incoming"
    if shard_path.exists() or incoming.exists():
        raise FileExistsError(f"refusing existing protected shard: {shard_path}")
    shard = {
        "schema_version": 1,
        "metadata": metadata,
        "boundary_metadata": boundary_metadata,
        "boundary_states": boundary_states,
        "component_inputs": {
            key: torch.stack(values, dim=0) for key, values in input_values.items()
        },
        "component_outputs": {
            key: torch.stack(values, dim=0) for key, values in output_values.items()
        },
    }
    torch.save(shard, incoming)
    os.replace(incoming, shard_path)
    return {
        "file": shard_name,
        "sha256": _sha256(shard_path),
        "rows": len(metadata),
        **boundary_metadata,
    }


def _capture_group(
    *,
    tokenizer,
    model,
    device: torch.device,
    label_ids: dict[str, int],
    messages: list[dict],
    cases: list[dict],
    layers: list[int],
    module_keys: list[tuple[int, str]],
    boundary_metadata: dict,
    output_dir: Path,
    shard_name: str,
) -> dict:
    prefix_ids = _exact_prefix(tokenizer, messages)
    base_cache, boundary_states = _prefill_with_last_boundary_capture(
        model, device, prefix_ids, layers
    )
    metadata = []
    input_values = {f"{layer}:{component}": [] for layer, component in module_keys}
    output_values = {f"{layer}:{component}": [] for layer, component in module_keys}
    for case in cases:
        rendered_prefix, suffix_ids = _split_final_user(
            tokenizer,
            _chat_ids(tokenizer, messages + [{"role": "user", "content": case["prompt"]}]),
        )
        if rendered_prefix != prefix_ids:
            raise RuntimeError("protected capture prefix mismatch")
        cache = copy.deepcopy(base_cache)
        inputs, outputs, logits, extended_cache = _suffix_component_capture(
            model,
            device,
            cache,
            len(prefix_ids),
            suffix_ids,
            label_ids,
            module_keys,
        )
        row = {
            **case["metadata"],
            "correct_label": case["correct_label"],
            "foil_label": case["foil_label"],
            "logit_a": logits[0],
            "logit_b": logits[1],
            "correct_logit_margin": _margin(logits, case["correct_label"]),
            "prefix_tokens": len(prefix_ids),
            "suffix_tokens": len(suffix_ids),
            "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
        }
        if not all(
            math.isfinite(float(row[key]))
            for key in ("logit_a", "logit_b", "correct_logit_margin")
        ):
            raise FloatingPointError("non-finite protected baseline")
        metadata.append(row)
        for layer, component in module_keys:
            key = f"{layer}:{component}"
            input_values[key].append(inputs[(layer, component)])
            output_values[key].append(outputs[(layer, component)])
        del cache, extended_cache
    del base_cache
    return _write_shard(
        output_dir=output_dir,
        shard_name=shard_name,
        metadata=metadata,
        boundary_metadata=boundary_metadata,
        boundary_states=boundary_states,
        input_values=input_values,
        output_values=output_values,
    )


def _benchmark_cases(items: list[dict], condition: str, role: str, style: str) -> list[dict]:
    cases = []
    for item in sorted(items, key=lambda row: row["item_id"]):
        for label_swap in (0, 1):
            prompt, correct_label, foil_label = pairwise_prompt(item, label_swap)
            cases.append(
                {
                    "prompt": prompt,
                    "correct_label": correct_label,
                    "foil_label": foil_label,
                    "metadata": {
                        "control_family": condition,
                        "benchmark": item["benchmark"],
                        "item_id": item["item_id"],
                        "partition": item["partition"],
                        "declared_role": role,
                        "history_style": style,
                        "history_realization": item["history_realization"],
                        "label_swap": label_swap,
                        "verification_required": 1.0,
                    },
                }
            )
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--behavior-analysis", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=["subspace_fit", "component_discovery"],
        default="subspace_fit",
    )
    parser.add_argument("--execution-authorization", type=Path)
    parser.add_argument("--router-contract", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()

    mechanism = json.loads(args.mechanism_contract.read_text())
    operator = json.loads(args.operator_contract.read_text())
    execution = json.loads(args.execution_contract.read_text())
    site_manifest = json.loads(args.operator_site_manifest.read_text())
    behavior_analysis = json.loads(args.behavior_analysis.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    controls = load_controls(args.controls)
    if (args.execution_authorization is None) != (args.router_contract is None):
        raise ValueError(
            "protected router-audit capture requires both authorization and router contract"
        )
    authorization = None
    if args.execution_authorization is not None:
        authorization = json.loads(args.execution_authorization.read_text())
        router_contract = json.loads(args.router_contract.read_text())
        required_authorization = {
            "schema_version": 1,
            "stage": "governance_consensus_router_audit_capture",
            "execution_allowed": True,
            "router_contract_sha256": sha256_file(args.router_contract),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "v5_audit_controls_sha256": sha256_file(args.controls),
            "protected_partition": args.partition,
            "expected_protected_rows": 4008,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        for field, expected in required_authorization.items():
            if authorization.get(field) != expected:
                raise ValueError(f"protected capture authorization mismatch: {field}")
        if router_contract.get("status") != "frozen_before_any_v5_capture_or_fit":
            raise ValueError("V5 router contract is not frozen")
        if controls.get("frozen_before_any_v5_capture_forward") is not True:
            raise ValueError("V5 protected controls were not frozen before capture")
        if controls.get("identity_disjoint_from_mitigation_controls_v1") is not True:
            raise ValueError("V5 protected controls are not identity-disjoint")
        code_root = authorization.get("code_root", "")
        if not code_root.startswith("/workspace/context-mismatch-qwen3-8b/code-v"):
            raise ValueError("protected capture authorization code root is invalid")
        bundle = Path(code_root) / "bundle.sha256"
        if bundle.exists() and authorization.get(
            "immutable_code_bundle_manifest_sha256"
        ) != sha256_file(bundle):
            raise ValueError("protected capture authorization bundle SHA mismatch")
    if execution["mechanism_contract_sha256"] != sha256_file(args.mechanism_contract):
        raise ValueError("execution/mechanism contract mismatch")
    if execution["operator_contract_sha256"] != sha256_file(args.operator_contract):
        raise ValueError("execution/operator contract mismatch")
    if operator["benchmark_manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("operator/benchmark manifest mismatch")
    if execution["mitigation_controls_sha256"] != sha256_file(args.controls):
        raise ValueError("execution/mitigation-controls mismatch")
    if manifest_report["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("manifest report mismatch")
    if not behavior_analysis.get("audit", {}).get("success"):
        raise ValueError("full behavior analysis did not pass")
    if not site_manifest.get("locked") or site_manifest["selection_partition"] != "component_discovery":
        raise ValueError("operator site manifest is not locked")
    if site_manifest["operator_site_execution_contract_sha256"] != sha256_file(args.execution_contract):
        raise ValueError("operator site manifest execution binding mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest["revision"] != operator["base_model_revision"]:
        raise ValueError("model revision mismatch")
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing protected capture directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)

    manifest = load_jsonl(args.manifest)
    items = [row for row in manifest if row["partition"] == args.partition]
    if len(items) != 192:
        raise ValueError(
            f"protected capture partition {args.partition} has {len(items)} items, expected 192"
        )
    by_realization = defaultdict(list)
    for item in items:
        by_realization[int(item["history_realization"])].append(item)
    layers = [int(site["layer"]) for site in site_manifest["selected_sites_ordered"]]
    site_components = [(int(site["layer"]), site["component"]) for site in site_manifest["selected_sites_ordered"]]
    # Protected fitting uses the selected component at each layer, not both exploratory components.
    module_keys = site_components
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)
    records = []

    for role in mechanism["factorial"]["declared_roles"]:
        fresh_cases = _benchmark_cases(items, "fresh", role, "none")
        records.append(
            _capture_group(
                tokenizer=tokenizer,
                model=model,
                device=device,
                label_ids=label_ids,
                messages=prefix_messages(role, "fresh", "natural", 32, 0),
                cases=fresh_cases,
                layers=layers,
                module_keys=module_keys,
                boundary_metadata={
                    "control_family": "fresh",
                    "declared_role": role,
                    "history_style": "none",
                    "history_realization": 0,
                    "verification_required": 1.0,
                },
                output_dir=args.output_dir,
                shard_name=f"protected__fresh__{role}.pt",
            )
        )
        for style in mechanism["factorial"]["history_styles"]:
            for realization, realization_items in sorted(by_realization.items()):
                for condition in ("verification", "obedience_reset"):
                    records.append(
                        _capture_group(
                            tokenizer=tokenizer,
                            model=model,
                            device=device,
                            label_ids=label_ids,
                            messages=prefix_messages(role, condition, style, 32, realization),
                            cases=_benchmark_cases(realization_items, condition, role, style),
                            layers=layers,
                            module_keys=module_keys,
                            boundary_metadata={
                                "control_family": condition,
                                "declared_role": role,
                                "history_style": style,
                                "history_realization": realization,
                                "verification_required": 1.0,
                            },
                            output_dir=args.output_dir,
                            shard_name=(
                                f"protected__{condition}__{role}__{style}__r{realization}.pt"
                            ),
                        )
                    )
                authority_cases = []
                for control in controls["supported_user_authority"]:
                    for label_swap in (0, 1):
                        prompt, correct_label, foil_label = supported_authority_prompt(control, label_swap)
                        authority_cases.append(
                            {
                                "prompt": prompt,
                                "correct_label": correct_label,
                                "foil_label": foil_label,
                                "metadata": {
                                    "control_family": "supported_user_authority",
                                    "control_id": control["id"],
                                    "declared_role": role,
                                    "history_style": style,
                                    "history_realization": realization,
                                    "label_swap": label_swap,
                                    "verification_required": 0.0,
                                },
                            }
                        )
                records.append(
                    _capture_group(
                        tokenizer=tokenizer,
                        model=model,
                        device=device,
                        label_ids=label_ids,
                        messages=prefix_messages(role, "obedience", style, 32, realization),
                        cases=authority_cases,
                        layers=layers,
                        module_keys=module_keys,
                        boundary_metadata={
                            "control_family": "supported_user_authority",
                            "declared_role": role,
                            "history_style": style,
                            "history_realization": realization,
                            "verification_required": 0.0,
                        },
                        output_dir=args.output_dir,
                        shard_name=(
                            f"protected__supported_authority__{role}__{style}__r{realization}.pt"
                        ),
                    )
                )
        for control in controls["factual_boundary_memory"]:
            memory_cases = []
            for label_swap in (0, 1):
                prompt, correct_label, foil_label = factual_memory_prompt(control, label_swap)
                memory_cases.append(
                    {
                        "prompt": prompt,
                        "correct_label": correct_label,
                        "foil_label": foil_label,
                        "metadata": {
                            "control_family": "factual_boundary_memory",
                            "control_id": control["id"],
                            "declared_role": role,
                            "history_style": "none",
                            "history_realization": 0,
                            "label_swap": label_swap,
                            "verification_required": 0.0,
                        },
                    }
                )
            records.append(
                _capture_group(
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    label_ids=label_ids,
                    messages=[{"role": "system", "content": system_message(role)}]
                    + factual_memory_messages(control),
                    cases=memory_cases,
                    layers=layers,
                    module_keys=module_keys,
                    boundary_metadata={
                        "control_family": "factual_boundary_memory",
                        "control_id": control["id"],
                        "declared_role": role,
                        "history_style": "none",
                        "history_realization": 0,
                        "verification_required": 0.0,
                    },
                    output_dir=args.output_dir,
                    shard_name=f"protected__factual_memory__{role}__{control['id']}.pt",
                )
            )

    family_counts = defaultdict(int)
    for record in records:
        family_counts[record["control_family"]] += int(record["rows"])
    expected_counts = {
        "fresh": 768,
        "verification": 1536,
        "obedience_reset": 1536,
        "supported_user_authority": 144,
        "factual_boundary_memory": 24,
    }
    if dict(sorted(family_counts.items())) != expected_counts:
        raise RuntimeError(
            f"protected capture count mismatch: {dict(family_counts)} != {expected_counts}"
        )
    capture_manifest = {
        "schema_version": 1,
        "stage": (
            "governance_consensus_router_audit_protected_capture"
            if authorization is not None
            else "protected_capture"
        ),
        "partition": args.partition,
        "rows": sum(record["rows"] for record in records),
        "family_rows": expected_counts,
        "layers": layers,
        "sites": site_manifest["selected_sites_ordered"],
        "hidden_size": mechanism["base_model"]["hidden_size"],
        "dtype": "float16_capture_from_bfloat16_forward",
        "mechanism_contract_sha256": sha256_file(args.mechanism_contract),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "operator_site_execution_contract_sha256": sha256_file(args.execution_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "controls_sha256": sha256_file(args.controls),
        "behavior_analysis_sha256": sha256_file(args.behavior_analysis),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "authorization_sha256": (
            sha256_file(args.execution_authorization)
            if args.execution_authorization is not None
            else None
        ),
        "router_contract_sha256": (
            sha256_file(args.router_contract) if args.router_contract is not None else None
        ),
        "shards": records,
        "application_gated_and_forced_on_stress_required": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "capture_manifest.json",
        json.dumps(capture_manifest, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(capture_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
