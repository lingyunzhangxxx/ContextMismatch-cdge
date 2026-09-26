#!/usr/bin/env python3
"""Run six-family protected controls for one immutable DSGE-V3 checkpoint."""

from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import torch
from torch import nn

from scripts.benchmark_v1.common import (
    atomic_write_text,
    canonical_json,
    load_jsonl,
    sha256_file,
)
from scripts.benchmark_v1.controls import load_controls
from scripts.benchmark_v1.operators import LayerOperatorSpec, MultiLayerContextOperator
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
from scripts.benchmark_v4.directional_governance import (
    DirectionalGovernanceSiteEditor,
    editor_from_checkpoint,
)
from scripts.benchmark_v4.run_directional_governance import _candidate_report


class ForcedDirectionalSiteEditor(nn.Module):
    """Protected-control-only wrapper that forces exactly one V3 expert on."""

    def __init__(self, editor: DirectionalGovernanceSiteEditor, direction: str):
        super().__init__()
        if direction not in {"positive", "negative"}:
            raise ValueError("forced direction must be positive or negative")
        self.editor = editor
        self.direction = direction
        self.last_diagnostics: dict[str, torch.Tensor] = {}

    def forward(
        self,
        value: torch.Tensor,
        gate: torch.Tensor | float,
        boundary_state: torch.Tensor | None = None,
        context_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if boundary_state is None or context_state is None:
            raise ValueError("forced directional control requires boundary and context")
        gate_tensor = torch.as_tensor(gate, device=value.device, dtype=torch.float32)
        if not bool(torch.any(gate_tensor != 0).detach().cpu()):
            self.last_diagnostics = {}
            return value
        positive, negative, diagnostics = self.editor.forced_direction_coordinates(
            direction=self.direction,
            value=value,
            boundary_state=boundary_state,
            context_state=context_state,
        )
        correction = (
            positive @ self.editor.positive_output_basis.transpose(0, 1)
            + negative @ self.editor.negative_output_basis.transpose(0, 1)
        )
        if gate_tensor.ndim == 0:
            gate_tensor = gate_tensor.expand(correction.shape[0])
        elif gate_tensor.numel() == 1:
            gate_tensor = gate_tensor.reshape(()).expand(correction.shape[0])
        elif gate_tensor.shape != (correction.shape[0],):
            raise ValueError("forced external gate must be scalar or one value per row")
        applied = gate_tensor.unsqueeze(-1) * correction
        edited = value.float().clone()
        if edited.ndim == 2:
            edited = edited + applied
        elif edited.ndim == 3:
            edited[:, -1] = edited[:, -1] + applied
        else:
            raise ValueError("forced edit value must have rank two or three")
        diagnostics["applied_correction"] = applied
        self.last_diagnostics = diagnostics
        return edited.to(value.dtype)


def _controller_diagnostics(values: dict[str, torch.Tensor]) -> dict[str, float]:
    allowed = {
        "history_probability",
        "task_probability",
        "forced_history_probability",
        "forced_task_probability",
        "positive_evidence",
        "negative_evidence",
        "positive_route",
        "negative_route",
        "trust_scale",
    }
    return {
        name: float(value.reshape(-1)[0].detach().cpu())
        for name, value in values.items()
        if name in allowed
    }


def _selected_logit_error(left: dict, right: dict) -> float:
    return max(
        abs(float(left["logit_a"]) - float(right["logit_a"])),
        abs(float(left["logit_b"]) - float(right["logit_b"])),
    )


def _validate_prior(report: dict, *, checkpoint_sha: str, fit_report_sha: str) -> None:
    required = {
        "method": "DSGE-V3",
        "evaluation_stage": "governance_directional_operator_dev",
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_report_sha,
        "fit_eligible": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if report.get(field) != expected:
            raise ValueError(f"operator-dev prior mismatch: {field}")
    if report.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev prior is incomplete")
    if report.get("selection_gate", {}).get("candidate_eligible") is not True:
        raise ValueError("operator-dev candidate did not pass its frozen gate")
    if report.get("matched_selected_logit_max_error") != 0.0:
        raise ValueError("operator-dev matched identity is not exact")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fit-audit-analysis", type=Path, required=True)
    parser.add_argument("--operator-dev-analysis", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
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
            raise FileExistsError(f"refusing existing directional-control output: {path}")

    checkpoint_sha = sha256_file(args.checkpoint)
    fit_report_sha = sha256_file(args.fit_report)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fit_report = json.loads(args.fit_report.read_text())
    fit_audit = json.loads(args.fit_audit_analysis.read_text())
    operator_dev = json.loads(args.operator_dev_analysis.read_text())
    contract = json.loads(args.editor_contract.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    authorization = json.loads(args.execution_authorization.read_text())

    for field, expected in {
        "stage": "governance_directional_fit",
        "method": "DSGE-V3",
        "fit_complete": True,
        "candidate_count": 3,
        "all_three_candidate_checkpoints_materialized": True,
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if fit_report.get(field) != expected:
            raise ValueError(f"directional fit report mismatch: {field}")
    candidate = _candidate_report(fit_report, checkpoint_sha)
    if candidate.get("fit_eligible") is not True:
        raise ValueError("fit-ineligible directional candidate cannot run controls")
    if checkpoint.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("checkpoint editor-contract binding mismatch")
    for field, expected in {
        "method": "DSGE-V3",
        "evaluation_stage": "governance_directional_fit_audit",
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_report_sha,
        "fit_eligible": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if fit_audit.get(field) != expected:
            raise ValueError(f"fit-audit prior mismatch: {field}")
    if fit_audit.get("audit", {}).get("success") is not True or fit_audit.get(
        "selection_gate", {}
    ).get("candidate_eligible") is not True:
        raise ValueError("fit-audit prior did not pass")
    _validate_prior(operator_dev, checkpoint_sha=checkpoint_sha, fit_report_sha=fit_report_sha)
    if contract.get("method_short_name") != "DSGE-V3":
        raise ValueError("unexpected directional editor contract")
    if contract.get("final_test_open") or contract.get("final_test_open_count") != 0:
        raise ValueError("directional contract unexpectedly opens final test")
    if contract.get("production_rollout_approved"):
        raise ValueError("directional contract unexpectedly approves production")
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
    expected_rows = 2856
    expected_key_sha = governance_control_key_hash(groups)
    required_authorization = {
        "stage": "governance_directional_controls",
        "execution_allowed": True,
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_report_sha,
        "fit_audit_analysis_sha256": sha256_file(args.fit_audit_analysis),
        "operator_dev_analysis_sha256": sha256_file(args.operator_dev_analysis),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "controls_sha256": sha256_file(args.controls),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows": expected_rows,
        "expected_key_sha256": expected_key_sha,
        "forced_directions": ["positive", "negative"],
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_authorization.items():
        if authorization.get(field) != expected:
            raise ValueError(f"directional-control authorization mismatch: {field}")
    code_root = str(authorization.get("code_root", ""))
    if code_root != "/workspace/context-mismatch-qwen3-8b/code-v30":
        raise ValueError("directional controls require exact code-v30")
    bundle = Path(code_root) / "bundle.sha256"
    if authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(bundle):
        raise ValueError("directional-control immutable bundle SHA mismatch")

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze base model")
    directional = editor_from_checkpoint(checkpoint).to(device).eval()
    application_manager = directional.hook_manager().to(device)
    specs = list(application_manager.operators)
    if len(specs) != 1:
        raise RuntimeError("directional controls require exactly one editor site")
    spec = specs[0]
    forced_managers: dict[str, MultiLayerContextOperator] = {}
    for direction in ("positive", "negative"):
        wrapper = ForcedDirectionalSiteEditor(directional.site_editor, direction).to(device).eval()
        forced_managers[direction] = MultiLayerContextOperator({spec: wrapper}).to(device)
    label_ids = _label_ids(tokenizer)
    environment = {
        "schema_version": 1,
        "stage": "governance_directional_controls",
        "method": "DSGE-V3",
        "candidate_id": candidate["candidate_id"],
        "site": candidate["site"],
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": fit_report_sha,
        "fit_audit_analysis_sha256": sha256_file(args.fit_audit_analysis),
        "operator_dev_analysis_sha256": sha256_file(args.operator_dev_analysis),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "controls_sha256": sha256_file(args.controls),
        "expected_rows": expected_rows,
        "planned_rows": expected_rows,
        "expected_key_sha256": expected_key_sha,
        "forced_directions": ["positive", "negative"],
        "device": str(device),
        "torch_version": torch.__version__,
        "base_model_trainable_parameters": 0,
        "editor_trainable_parameters": directional.trainable_parameter_count,
        "prior_candidate_eligible": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n"
    )

    completed: set[str] = set()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as output_handle:
        for group in groups:
            messages = group["messages"]
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(
                model, device, prefix_ids, [spec.layer]
            )
            boundary = captured[str(spec.layer)].to(device=device, dtype=torch.float32)
            for case in sorted(group["cases"], key=governance_control_case_key):
                current_key = governance_control_case_key(case)
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(
                        tokenizer, messages + [{"role": "user", "content": case["prompt"]}]
                    ),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"directional control prefix mismatch: {current_key}")
                baseline_logits, baseline_cache = _suffix_logits_batch(
                    model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                )
                del baseline_cache
                baseline = _margins(
                    baseline_logits[0], case["correct_label"], case["foil_label"]
                )

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
                gated = _margins(
                    gated_logits[0], case["correct_label"], case["foil_label"]
                )
                gated_controller = _controller_diagnostics(
                    directional.site_editor.last_diagnostics
                )
                gated_intervention = application_manager.last_stats.get(
                    spec,
                    {
                        "mean_relative_intervention_norm": 0.0,
                        "max_relative_intervention_norm": 0.0,
                    },
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
                            model,
                            device,
                            base_cache,
                            len(prefix_ids),
                            [suffix_ids],
                            label_ids,
                            0,
                        )
                    finally:
                        manager.remove()
                    del extended
                    forced_results[direction] = _margins(
                        logits[0], case["correct_label"], case["foil_label"]
                    )
                    forced_stats[direction] = manager.last_stats[spec]
                    forced_editor = manager.operators[spec]
                    assert isinstance(forced_editor, ForcedDirectionalSiteEditor)
                    forced_controllers[direction] = _controller_diagnostics(
                        forced_editor.last_diagnostics
                    )

                record = {
                    "case_key": current_key,
                    "method": "DSGE-V3",
                    "candidate_id": candidate["candidate_id"],
                    "site": candidate["site"],
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
                    "application_gated_selected_logit_error": _selected_logit_error(
                        gated, baseline
                    ),
                    "application_gated_margin_change": gated["correct_margin"]
                    - baseline["correct_margin"],
                    "application_gated_binary_kl": _binary_kl(
                        baseline["correct_probability_binary"],
                        gated["correct_probability_binary"],
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
                        baseline["correct_probability_binary"],
                        forced["correct_probability_binary"],
                    )
                    record[f"forced_on_{direction}_controller_diagnostics"] = (
                        forced_controllers[direction]
                    )
                    record[f"forced_on_{direction}_intervention_diagnostics"] = (
                        forced_stats[direction]
                    )
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
                    raise FloatingPointError(
                        f"non-finite directional control output: {current_key}"
                    )
                output_handle.write(canonical_json(record) + "\n")
                completed.add(current_key)
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    if len(completed) != expected_rows:
        raise RuntimeError("directional controls ended without every expected row")
    print(
        json.dumps(
            {
                "candidate_id": candidate["candidate_id"],
                "completed_unique_rows": len(completed),
                "expected_rows": expected_rows,
                "expected_key_sha256": expected_key_sha,
                "forced_directions": ["positive", "negative"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
