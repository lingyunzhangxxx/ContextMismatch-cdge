#!/usr/bin/env python3
"""Evaluate application-gated and forced-on collateral controls for one candidate."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v1.controls import (
    factual_memory_messages,
    factual_memory_prompt,
    load_controls,
    supported_authority_prompt,
)
from scripts.benchmark_v1.histories import prefix_messages, system_message
from scripts.benchmark_v1.run_behavior import (
    _chat_ids,
    _device,
    _exact_prefix,
    _label_ids,
    _split_final_user,
    _suffix_logits_batch,
    _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import _load_model, _module
from scripts.benchmark_v1.run_protected_capture import _prefill_with_last_boundary_capture
from scripts.benchmark_v2.crossover import task_prompt
from scripts.benchmark_v2.run_operator_candidate import (
    _build_operator,
    _gate_values,
    _load_torch,
)


def _operator_items(manifest: list[dict], half: str) -> list[dict]:
    selected = []
    for benchmark in sorted({row["benchmark"] for row in manifest}):
        rows = sorted(
            (
                row
                for row in manifest
                if row["benchmark"] == benchmark and row["partition"] == "operator_dev"
            ),
            key=lambda row: row["item_id"],
        )
        if len(rows) != 32:
            raise ValueError(f"operator_dev:{benchmark} has {len(rows)} items")
        selected.extend(rows[:16] if half == "screening" else rows[16:])
    if len(selected) != 96:
        raise ValueError("operator control benchmark half must contain 96 items")
    return selected


def _case_key(case: dict) -> str:
    metadata = case["metadata"]
    return "__".join(
        str(value)
        for value in (
            metadata["control_family"],
            metadata.get("benchmark", "none"),
            metadata["case_id"],
            metadata["declared_role"],
            metadata["history_style"],
            metadata["history_realization"],
            metadata["label_swap"],
        )
    )


def _benchmark_cases(items: list[dict], role: str, style: str, family: str) -> list[dict]:
    result = []
    for item in sorted(items, key=lambda row: row["item_id"]):
        for label_swap in (0, 1):
            spec = task_prompt(item, label_swap, "independent_verification")
            result.append(
                {
                    "prompt": spec["prompt"],
                    "correct_label": spec["task_correct_label"],
                    "foil_label": spec["task_foil_label"],
                    "target_obedience": 0.0,
                    "applicable": 1.0,
                    "metadata": {
                        "control_family": family,
                        "benchmark": item["benchmark"],
                        "case_id": item["item_id"],
                        "declared_role": role,
                        "history_style": style,
                        "history_realization": int(item["history_realization"]),
                        "label_swap": label_swap,
                    },
                }
            )
    return result


def _groups(items: list[dict], controls: dict, roles: list[str], styles: list[str]) -> list[dict]:
    groups = []
    by_realization = defaultdict(list)
    for item in items:
        by_realization[int(item["history_realization"])].append(item)
    for role in roles:
        groups.append(
            {
                "group_id": f"fresh__{role}",
                "messages": prefix_messages(role, "fresh", "natural", 32, 0),
                "cases": _benchmark_cases(items, role, "none", "fresh_verification"),
            }
        )
        for style in styles:
            for realization, current_items in sorted(by_realization.items()):
                groups.append(
                    {
                        "group_id": f"verification__{role}__{style}__r{realization}",
                        "messages": prefix_messages(role, "verification", style, 32, realization),
                        "cases": _benchmark_cases(
                            current_items, role, style, "matched_verification"
                        ),
                    }
                )
                groups.append(
                    {
                        "group_id": f"reset__{role}__{style}__r{realization}",
                        "messages": prefix_messages(
                            role, "obedience_reset", style, 32, realization
                        ),
                        "cases": _benchmark_cases(
                            current_items, role, style, "explicit_governance_reset"
                        ),
                    }
                )
                authority_cases = []
                for control in controls["supported_user_authority"]:
                    for label_swap in (0, 1):
                        prompt, correct_label, foil_label = supported_authority_prompt(
                            control, label_swap
                        )
                        authority_cases.append(
                            {
                                "prompt": prompt,
                                "correct_label": correct_label,
                                "foil_label": foil_label,
                                "target_obedience": 1.0,
                                "applicable": 1.0,
                                "metadata": {
                                    "control_family": "supported_user_authority",
                                    "case_id": control["id"],
                                    "declared_role": role,
                                    "history_style": style,
                                    "history_realization": realization,
                                    "label_swap": label_swap,
                                },
                            }
                        )
                groups.append(
                    {
                        "group_id": f"authority__{role}__{style}__r{realization}",
                        "messages": prefix_messages(role, "obedience", style, 32, realization),
                        "cases": authority_cases,
                    }
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
                        "target_obedience": 0.0,
                        "applicable": 0.0,
                        "metadata": {
                            "control_family": "factual_boundary_memory",
                            "case_id": control["id"],
                            "declared_role": role,
                            "history_style": "none",
                            "history_realization": 0,
                            "label_swap": label_swap,
                        },
                    }
                )
            groups.append(
                {
                    "group_id": f"memory__{role}__{control['id']}",
                    "messages": [{"role": "system", "content": system_message(role)}]
                    + factual_memory_messages(control),
                    "cases": memory_cases,
                }
            )
    keys = [_case_key(case) for group in groups for case in group["cases"]]
    if len(keys) != 2088 or len(set(keys)) != 2088:
        raise ValueError(
            f"operator controls enumerate {len(keys)} rows and {len(set(keys))} unique keys; expected 2088"
        )
    return groups


def _key_hash(groups: list[dict]) -> str:
    keys = sorted(_case_key(case) for group in groups for case in group["cases"])
    return hashlib.sha256((("\n".join(keys)) + "\n").encode("utf-8")).hexdigest()


def _margins(logits: tuple[float, float], correct: str, foil: str) -> dict:
    values = {"A": float(logits[0]), "B": float(logits[1])}
    margin = values[correct] - values[foil]
    return {
        "logit_a": values["A"],
        "logit_b": values["B"],
        "correct_margin": margin,
        "correct_probability_binary": 1.0
        / (1.0 + math.exp(max(-700.0, min(700.0, -margin)))),
    }


def _binary_kl(p: float, q: float) -> float:
    epsilon = 1e-7
    p = min(max(float(p), epsilon), 1.0 - epsilon)
    q = min(max(float(q), epsilon), 1.0 - epsilon)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-tensor", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dev-half", choices=["screening", "selection"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidate = _load_torch(args.candidate_tensor)
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    identity = json.loads(args.identity_report.read_text())
    contract = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    controls = load_controls(args.controls)
    model_manifest = json.loads(args.model_manifest.read_text())
    if candidate_manifest["tensor_sha256"] != sha256_file(args.candidate_tensor):
        raise ValueError("candidate tensor hash mismatch")
    if identity.get("success") is not True or identity.get("max_error") != 0.0:
        raise ValueError("operator zero-gate identity report did not pass")
    if manifest_report["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != contract["base_model"]["revision"]:
        raise ValueError("model revision mismatch")
    items = _operator_items(load_jsonl(args.manifest), args.dev_half)
    groups = _groups(
        items,
        controls,
        list(contract["factorial"]["declared_roles"]),
        list(contract["factorial"]["history_styles"]),
    )
    expected_key_sha256 = _key_hash(groups)
    required = {
        "stage": f"operator_controls_{args.dev_half}",
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "candidate_tensor_sha256": sha256_file(args.candidate_tensor),
        "identity_report_sha256": sha256_file(args.identity_report),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "controls_sha256": sha256_file(args.controls),
        "expected_key_sha256": expected_key_sha256,
        "expected_rows": 2088,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"control authorization field mismatch: {field}")
    if not authorization.get("execution_allowed"):
        raise ValueError("control execution is not authorized")

    completed = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        completed = {row["case_key"] for row in load_jsonl(args.output)}
    if args.environment_output.exists() and not args.resume:
        raise FileExistsError(f"refusing existing environment: {args.environment_output}")
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)
    manager, gate_models = _build_operator(candidate, device)
    specs = list(manager.operators)
    layers = sorted({spec.layer for spec in specs})
    environment = {
        "schema_version": 1,
        "stage": f"operator_controls_{args.dev_half}",
        "candidate_id": candidate["candidate_id"],
        "candidate_tensor_sha256": sha256_file(args.candidate_tensor),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "identity_report_sha256": sha256_file(args.identity_report),
        "controls_sha256": sha256_file(args.controls),
        "expected_key_sha256": expected_key_sha256,
        "planned_rows": 2088,
        "device": str(device),
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    if not args.environment_output.exists():
        atomic_write_text(
            args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as output_handle:
        for group in groups:
            messages = group["messages"]
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(
                model, device, prefix_ids, layers
            )
            boundaries = {
                spec: captured[str(spec.layer)].to(device=device, dtype=torch.float32)
                for spec in specs
            }
            for case in sorted(group["cases"], key=_case_key):
                current_key = _case_key(case)
                if current_key in completed:
                    continue
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(
                        tokenizer, messages + [{"role": "user", "content": case["prompt"]}]
                    ),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"control prefix mismatch: {current_key}")
                baseline_logits, baseline_cache = _suffix_logits_batch(
                    model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                )
                del baseline_cache
                baseline = _margins(
                    baseline_logits[0], case["correct_label"], case["foil_label"]
                )
                application_gates, gate_diagnostics = _gate_values(
                    candidate,
                    gate_models,
                    boundaries,
                    float(case["target_obedience"]),
                    applicable=float(case["applicable"]),
                )
                manager.install(
                    model,
                    gate=application_gates,
                    module_resolver=_module,
                    boundary_state=boundaries,
                    collect_diagnostics=True,
                )
                try:
                    gated_logits, gated_cache = _suffix_logits_batch(
                        model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                    )
                finally:
                    manager.remove()
                del gated_cache
                gated = _margins(gated_logits[0], case["correct_label"], case["foil_label"])
                gated_stats = {
                    f"{spec.layer}:{spec.component}": value
                    for spec, value in manager.last_stats.items()
                }
                forced_gates = {spec: 1.0 for spec in specs}
                manager.install(
                    model,
                    gate=forced_gates,
                    module_resolver=_module,
                    boundary_state=boundaries,
                    collect_diagnostics=True,
                )
                try:
                    forced_logits, forced_cache = _suffix_logits_batch(
                        model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                    )
                finally:
                    manager.remove()
                del forced_cache
                forced = _margins(forced_logits[0], case["correct_label"], case["foil_label"])
                forced_stats = {
                    f"{spec.layer}:{spec.component}": value
                    for spec, value in manager.last_stats.items()
                }
                record = {
                    "case_key": current_key,
                    "candidate_id": candidate["candidate_id"],
                    "dev_half": args.dev_half,
                    **case["metadata"],
                    "correct_label": case["correct_label"],
                    "foil_label": case["foil_label"],
                    "target_obedience": case["target_obedience"],
                    "operator_applicable": case["applicable"],
                    "prefix_tokens": len(prefix_ids),
                    "suffix_tokens": len(suffix_ids),
                    "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                    "baseline": baseline,
                    "application_gated": gated,
                    "forced_on_stress": forced,
                    "application_gated_margin_change": gated["correct_margin"]
                    - baseline["correct_margin"],
                    "forced_on_margin_change": forced["correct_margin"]
                    - baseline["correct_margin"],
                    "application_gated_binary_kl": _binary_kl(
                        baseline["correct_probability_binary"],
                        gated["correct_probability_binary"],
                    ),
                    "forced_on_binary_kl": _binary_kl(
                        baseline["correct_probability_binary"],
                        forced["correct_probability_binary"],
                    ),
                    "gate_diagnostics": gate_diagnostics,
                    "application_gated_intervention_diagnostics": gated_stats,
                    "forced_on_intervention_diagnostics": forced_stats,
                }
                finite = [
                    *baseline.values(),
                    *gated.values(),
                    *forced.values(),
                    record["application_gated_margin_change"],
                    record["forced_on_margin_change"],
                    record["application_gated_binary_kl"],
                    record["forced_on_binary_kl"],
                ]
                if not all(math.isfinite(float(value)) for value in finite):
                    raise FloatingPointError(f"non-finite control output: {current_key}")
                output_handle.write(canonical_json(record) + "\n")
                completed.add(current_key)
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()
    print(
        json.dumps(
            {
                "candidate_id": candidate["candidate_id"],
                "dev_half": args.dev_half,
                "completed_unique_rows": len(completed),
                "expected_rows": 2088,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if len(completed) != 2088:
        raise SystemExit("operator controls ended without every expected row")


if __name__ == "__main__":
    main()
