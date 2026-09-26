#!/usr/bin/env python3
"""Run the frozen ADSGE-V4 checkpoint on all protected-control families."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v1.controls import load_controls
from scripts.benchmark_v1.operators import MultiLayerContextOperator
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
from scripts.benchmark_v2.run_operator_controls import _binary_kl, _margins
from scripts.benchmark_v3.governance_control_design import (
    governance_control_case_key,
    governance_control_groups,
    governance_control_key_hash,
    operator_selection_items,
)
from scripts.benchmark_v4.run_directional_controls import (
    ForcedDirectionalSiteEditor,
    _selected_logit_error,
)
from scripts.benchmark_v5.abstaining_directional_governance import editor_from_checkpoint


EXPECTED_ROWS = 2856
EXPECTED_KEY_SHA256 = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"
CODE_ROOT = "/workspace/context-mismatch-qwen3-8b/code-v58"
FAMILIES = {
    "fresh_verification",
    "matched_verification",
    "matched_delegated_choice",
    "explicit_governance_reset",
    "supported_user_authority",
    "factual_boundary_memory",
}


def _diagnostics(values: dict[str, torch.Tensor]) -> dict[str, float]:
    allowed = {
        "history_probability",
        "task_probability",
        "forced_history_probability",
        "forced_task_probability",
        "positive_evidence",
        "negative_evidence",
        "positive_route",
        "negative_route",
        "structural_gate_active",
        "application_logit",
        "application_probability",
        "application_gate_active",
        "v4_gate_active",
    }
    return {
        name: float(value.reshape(-1)[0].detach().cpu())
        for name, value in values.items()
        if name in allowed
    }


def _require(value: dict, expected: dict, label: str) -> None:
    for field, wanted in expected.items():
        if value.get(field) != wanted:
            raise ValueError(f"{label} mismatch: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--v4-behavior-analysis", type=Path, required=True)
    parser.add_argument("--v3-controls-analysis", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--behavior-contract", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
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
            raise FileExistsError(f"refusing existing V4 protected-control output: {path}")

    checkpoint_sha = sha256_file(args.checkpoint)
    fit_report_sha = sha256_file(args.fit_report)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fit = json.loads(args.fit_report.read_text())
    behavior = json.loads(args.v4_behavior_analysis.read_text())
    v3_controls = json.loads(args.v3_controls_analysis.read_text())
    editor_contract = json.loads(args.editor_contract.read_text())
    behavior_contract = json.loads(args.behavior_contract.read_text())
    diagnostic_contract = json.loads(args.diagnostic_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())

    _require(
        fit,
        {
            "stage": "governance_abstaining_router_fit",
            "method": "ADSGE-V4",
            "fit_complete": True,
            "fit_eligible": False,
            "operator_dev_accessed": False,
            "protected_behavior_outputs_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
            "checkpoint_sha256": checkpoint_sha,
        },
        "V4 fit report",
    )
    _require(
        behavior,
        {
            "method": "ADSGE-V4",
            "post_failure_characterization": True,
            "fit_gates_passed": False,
            "fit_eligible": False,
            "candidate_eligible": False,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
            "fit_report_sha256": fit_report_sha,
        },
        "V4 behavior analysis",
    )
    if behavior.get("audit", {}).get("success") is not True:
        raise ValueError("V4 behavior analysis is not terminal audited")
    _require(
        v3_controls,
        {
            "method": "DSGE-V3",
            "evaluation_stage": "governance_directional_controls",
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "V3 controls analysis",
    )
    if v3_controls.get("audit", {}).get("success") is not True:
        raise ValueError("V3 controls analysis is not terminal audited")
    if editor_contract.get("method_short_name") != "ADSGE-V4":
        raise ValueError("unexpected V4 editor contract")
    if checkpoint.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("V4 checkpoint/editor-contract binding mismatch")
    _require(
        diagnostic_contract,
        {
            "post_failure_characterization": True,
            "fit_gates_passed": False,
            "candidate_eligible": False,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "V4 protected-controls contract",
    )
    bound = diagnostic_contract.get("bound_artifacts", {})
    bound_expected = {
        "v4_editor_contract_sha256": sha256_file(args.editor_contract),
        "v4_behavior_diagnostic_contract_sha256": sha256_file(args.behavior_contract),
        "v4_checkpoint_sha256": checkpoint_sha,
        "v4_fit_report_sha256": fit_report_sha,
        "v4_operator_dev_analysis_sha256": sha256_file(args.v4_behavior_analysis),
        "v3_protected_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
        "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "mitigation_controls_sha256": sha256_file(args.controls),
    }
    _require(bound, bound_expected, "V4 protected-controls bound artifact")
    evaluation = diagnostic_contract.get("evaluation", {})
    if evaluation.get("expected_rows") != EXPECTED_ROWS:
        raise ValueError("V4 protected-controls row contract mismatch")
    if evaluation.get("expected_key_sha256") != EXPECTED_KEY_SHA256:
        raise ValueError("V4 protected-controls key contract mismatch")
    if set(evaluation.get("control_families", [])) != FAMILIES:
        raise ValueError("V4 protected-controls family contract mismatch")
    if evaluation.get("forced_directions") != ["positive", "negative"]:
        raise ValueError("V4 protected-controls forced-direction contract mismatch")
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model revision mismatch")

    items = operator_selection_items(load_jsonl(args.manifest))
    groups = governance_control_groups(
        items,
        load_controls(args.controls),
        list(crossover["factorial"]["declared_roles"]),
        list(crossover["factorial"]["history_styles"]),
    )
    expected_key_sha = governance_control_key_hash(groups)
    if len([case for group in groups for case in group["cases"]]) != EXPECTED_ROWS:
        raise ValueError("materialized V4 control row count mismatch")
    if expected_key_sha != EXPECTED_KEY_SHA256:
        raise ValueError("materialized V4 control key mismatch")

    required_authorization = {
        "stage": "governance_failed_fit_protected_controls",
        "code_root": CODE_ROOT,
        "execution_allowed": True,
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_report_sha,
        "v4_behavior_analysis_sha256": sha256_file(args.v4_behavior_analysis),
        "v3_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "behavior_contract_sha256": sha256_file(args.behavior_contract),
        "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "controls_sha256": sha256_file(args.controls),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "forced_directions": ["positive", "negative"],
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    _require(authorization, required_authorization, "V4 protected-controls authorization")
    bundle = Path(CODE_ROOT) / "bundle.sha256"
    if authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(bundle):
        raise ValueError("code-v58 bundle binding mismatch")

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze base model")
    v4 = editor_from_checkpoint(checkpoint).to(device).eval()
    application_manager = v4.hook_manager().to(device)
    specs = list(application_manager.operators)
    if len(specs) != 1 or specs[0].layer != 27 or specs[0].component != "mlp":
        raise RuntimeError("V4 protected controls require exactly site 27:mlp")
    spec = specs[0]
    forced_managers: dict[str, MultiLayerContextOperator] = {}
    for direction in ("positive", "negative"):
        wrapper = ForcedDirectionalSiteEditor(v4.frozen_v3.site_editor, direction).to(device).eval()
        forced_managers[direction] = MultiLayerContextOperator({spec: wrapper}).to(device)
    label_ids = _label_ids(tokenizer)
    environment = {
        "schema_version": 1,
        "stage": "governance_failed_fit_protected_controls",
        "method": "ADSGE-V4",
        "candidate_id": "ADSGE-V4-27:mlp",
        "site": "27:mlp",
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_report_sha,
        "v4_behavior_analysis_sha256": sha256_file(args.v4_behavior_analysis),
        "v3_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "expected_rows": EXPECTED_ROWS,
        "planned_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "forced_directions": ["positive", "negative"],
        "device": str(device),
        "torch_version": torch.__version__,
        "base_model_trainable_parameters": 0,
        "editor_trainable_parameters": v4.trainable_parameter_count,
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "prior_candidate_eligible": False,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n")

    completed: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as output_handle:
        for group in groups:
            messages = group["messages"]
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(model, device, prefix_ids, [spec.layer])
            boundary = captured[str(spec.layer)].to(device=device, dtype=torch.float32)
            for case in sorted(group["cases"], key=governance_control_case_key):
                current_key = governance_control_case_key(case)
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(tokenizer, messages + [{"role": "user", "content": case["prompt"]}]),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"V4 protected-control prefix mismatch: {current_key}")
                baseline_logits, baseline_cache = _suffix_logits_batch(
                    model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                )
                del baseline_cache
                baseline = _margins(baseline_logits[0], case["correct_label"], case["foil_label"])

                application_manager.install(
                    model,
                    gate={spec: float(case["applicable"])},
                    module_resolver=_module,
                    boundary_state={spec: boundary},
                    collect_diagnostics=True,
                )
                try:
                    gated_logits, gated_cache = _suffix_logits_batch(
                        model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                    )
                finally:
                    application_manager.remove()
                del gated_cache
                gated = _margins(gated_logits[0], case["correct_label"], case["foil_label"])
                gated_controller = _diagnostics(v4.site_editor.last_diagnostics)
                gated_intervention = application_manager.last_stats.get(
                    spec,
                    {"mean_relative_intervention_norm": 0.0, "max_relative_intervention_norm": 0.0},
                )

                forced_results = {}
                forced_stats = {}
                forced_controllers = {}
                for direction in ("positive", "negative"):
                    manager = forced_managers[direction]
                    manager.install(
                        model,
                        gate={spec: 1.0},
                        module_resolver=_module,
                        boundary_state={spec: boundary},
                        collect_diagnostics=True,
                    )
                    try:
                        logits, extended = _suffix_logits_batch(
                            model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                        )
                    finally:
                        manager.remove()
                    del extended
                    forced_results[direction] = _margins(
                        logits[0], case["correct_label"], case["foil_label"]
                    )
                    forced_stats[direction] = manager.last_stats[spec]
                    forced_editor = manager.operators[spec]
                    forced_controllers[direction] = _diagnostics(forced_editor.last_diagnostics)

                record = {
                    "case_key": current_key,
                    "method": "ADSGE-V4",
                    "candidate_id": "ADSGE-V4-27:mlp",
                    "site": "27:mlp",
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
                    "application_gated_selected_logit_error": _selected_logit_error(gated, baseline),
                    "application_gated_margin_change": gated["correct_margin"] - baseline["correct_margin"],
                    "application_gated_binary_kl": _binary_kl(
                        baseline["correct_probability_binary"], gated["correct_probability_binary"]
                    ),
                    "application_gated_controller_diagnostics": gated_controller,
                    "application_gated_intervention_diagnostics": gated_intervention,
                }
                for direction in ("positive", "negative"):
                    forced = forced_results[direction]
                    record[f"forced_on_{direction}"] = forced
                    record[f"forced_on_{direction}_margin_change"] = (
                        forced["correct_margin"] - baseline["correct_margin"]
                    )
                    record[f"forced_on_{direction}_binary_kl"] = _binary_kl(
                        baseline["correct_probability_binary"], forced["correct_probability_binary"]
                    )
                    record[f"forced_on_{direction}_controller_diagnostics"] = forced_controllers[direction]
                    record[f"forced_on_{direction}_intervention_diagnostics"] = forced_stats[direction]
                finite = [
                    *baseline.values(),
                    *gated.values(),
                    record["application_gated_selected_logit_error"],
                    record["application_gated_margin_change"],
                    record["application_gated_binary_kl"],
                ]
                for direction in ("positive", "negative"):
                    finite.extend(forced_results[direction].values())
                    finite.extend(
                        (
                            record[f"forced_on_{direction}_margin_change"],
                            record[f"forced_on_{direction}_binary_kl"],
                        )
                    )
                if not all(math.isfinite(float(value)) for value in finite):
                    raise FloatingPointError(f"non-finite V4 protected-control output: {current_key}")
                output_handle.write(canonical_json(record) + "\n")
                completed.add(current_key)
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    if len(completed) != EXPECTED_ROWS:
        raise RuntimeError("V4 protected controls ended without every expected row")
    print(json.dumps({
        "completed_unique_rows": len(completed),
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "forced_directions": ["positive", "negative"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
