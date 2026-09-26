#!/usr/bin/env python3
"""Evaluate gated and forced-on controls for one immutable AMSGE checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v1.controls import load_controls
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
from scripts.benchmark_v2.run_operator_controls import (
    _binary_kl,
    _margins,
)
from scripts.benchmark_v3.adaptive_governance import editor_from_checkpoint
from scripts.benchmark_v3.governance_control_design import (
    governance_control_case_key,
    governance_control_groups,
    governance_control_key_hash,
    operator_selection_items,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--identity-report", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    for path in (args.output, args.environment_output):
        if path.exists():
            raise FileExistsError(f"refusing existing governance-control output: {path}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fit_report = json.loads(args.fit_report.read_text())
    editor_contract = json.loads(args.editor_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    identity = json.loads(args.identity_report.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    controls = load_controls(args.controls)
    model_manifest = json.loads(args.model_manifest.read_text())
    diagnostic = fit_report.get("all_fit_gates_pass") is False
    if diagnostic:
        if args.diagnostic_contract is None:
            raise ValueError("failed-fit controls require a diagnostic contract")
        diagnostic_contract = json.loads(args.diagnostic_contract.read_text())
        if diagnostic_contract.get("post_failure_characterization") is not True:
            raise ValueError("invalid post-failure diagnostic contract")
    elif fit_report.get("all_fit_gates_pass") is not True:
        raise ValueError("governance controls require a terminal fit report")
    if fit_report.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("fit report checkpoint SHA mismatch")
    if checkpoint.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("checkpoint editor-contract binding mismatch")
    if identity.get("success") is not True or identity.get("max_error") != 0.0:
        raise ValueError("behavior identity report did not pass exact-zero gate")
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model revision mismatch")

    items = operator_selection_items(load_jsonl(args.manifest))
    groups = governance_control_groups(
        items,
        controls,
        list(crossover["factorial"]["declared_roles"]),
        list(crossover["factorial"]["history_styles"]),
    )
    expected_key_sha256 = governance_control_key_hash(groups)
    expected_stage = (
        "governance_failed_fit_controls" if diagnostic else "governance_controls_selection"
    )
    required = {
        "stage": expected_stage,
        "execution_allowed": True,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(args.fit_report),
        "identity_report_sha256": sha256_file(args.identity_report),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "controls_sha256": sha256_file(args.controls),
        "expected_rows": 2856,
        "expected_key_sha256": expected_key_sha256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if diagnostic:
        required.update(
            {
                "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
                "post_failure_characterization": True,
                "fit_gates_passed": False,
                "candidate_eligible": False,
                "candidate_may_be_locked": False,
            }
        )
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"governance-control authorization mismatch: {field}")

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    adaptive = editor_from_checkpoint(checkpoint).to(device).eval()
    manager = adaptive.hook_manager().to(device)
    specs = list(manager.operators)
    layers = sorted({spec.layer for spec in specs})
    label_ids = _label_ids(tokenizer)
    environment = {
        "schema_version": 1,
        "stage": expected_stage,
        "candidate_id": f"{editor_contract['method_short_name']}-{sha256_file(args.checkpoint)[:16]}",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(args.fit_report),
        "identity_report_sha256": sha256_file(args.identity_report),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "controls_sha256": sha256_file(args.controls),
        "expected_key_sha256": expected_key_sha256,
        "planned_rows": 2856,
        "device": str(device),
        "base_model_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "editor_trainable_parameters": adaptive.trainable_parameter_count,
        "post_failure_characterization": diagnostic,
        "fit_gates_passed": bool(fit_report.get("all_fit_gates_pass")),
        "candidate_eligible": False if diagnostic else None,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    completed = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as output_handle:
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
            for case in sorted(group["cases"], key=governance_control_case_key):
                current_key = governance_control_case_key(case)
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
                application_gates = {
                    spec: float(case["applicable"]) for spec in specs
                }
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
                gated = _margins(
                    gated_logits[0], case["correct_label"], case["foil_label"]
                )
                gated_stats = {
                    f"{spec.layer}:{spec.component}": value
                    for spec, value in manager.last_stats.items()
                }
                manager.install(
                    model,
                    gate={spec: 1.0 for spec in specs},
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
                forced = _margins(
                    forced_logits[0], case["correct_label"], case["foil_label"]
                )
                forced_stats = {
                    f"{spec.layer}:{spec.component}": value
                    for spec, value in manager.last_stats.items()
                }
                record = {
                    "case_key": current_key,
                    "candidate_id": environment["candidate_id"],
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
    if len(completed) != 2856:
        raise RuntimeError("governance controls ended without every expected row")
    print(
        json.dumps(
            {
                "candidate_id": environment["candidate_id"],
                "completed_unique_rows": len(completed),
                "expected_rows": 2856,
                "expected_key_sha256": expected_key_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
