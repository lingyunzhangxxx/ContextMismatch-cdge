#!/usr/bin/env python3
"""Run one immutable AMSGE checkpoint on a frozen behavioral split."""

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
from scripts.benchmark_v1.histories import prefix_messages
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
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v2.run_operator_candidate import (
    _binary_kl,
    _binary_probability,
    _half_jobs,
    _key_hash,
    _margins,
)
from scripts.benchmark_v3.adaptive_governance import editor_from_checkpoint
from scripts.benchmark_v3.run_governance_capture import _selected_jobs


def _jobs(manifest: list[dict], contract: dict, evaluation_split: str) -> tuple[list[dict], str]:
    if evaluation_split in {"fit_smoke", "failed_fit_smoke"}:
        return _selected_jobs(manifest, contract, "smoke"), "replication"
    if evaluation_split in {"selection", "failed_fit_selection"}:
        return _half_jobs(enumerate_jobs(manifest, contract, "operator_dev"), "selection"), "operator_dev"
    if evaluation_split == "final_test":
        rows = enumerate_jobs(manifest, contract, "final_test")
        if len(rows) != 6144:
            raise ValueError("final-test row count mismatch")
        return rows, "final_test"
    raise ValueError(f"unknown evaluation split: {evaluation_split}")


def _validate_authorization(
    path: Path,
    *,
    evaluation_split: str,
    checkpoint: Path,
    fit_report: Path,
    editor_contract: Path,
    crossover_contract: Path,
    manifest: Path,
    design_audit: Path,
    expected_rows: int,
    expected_key_sha256: str,
    editor_lock: Path | None,
    diagnostic_contract: Path | None,
) -> dict:
    authorization = json.loads(path.read_text())
    diagnostic = evaluation_split.startswith("failed_fit_")
    stage = (
        f"governance_{evaluation_split}"
        if diagnostic
        else f"governance_behavior_{evaluation_split}"
    )
    required = {
        "stage": stage,
        "execution_allowed": True,
        "checkpoint_sha256": sha256_file(checkpoint),
        "fit_report_sha256": sha256_file(fit_report),
        "editor_contract_sha256": sha256_file(editor_contract),
        "crossover_contract_sha256": sha256_file(crossover_contract),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "design_audit_sha256": sha256_file(design_audit),
        "expected_rows": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "production_rollout_approved": False,
    }
    if diagnostic:
        if diagnostic_contract is None:
            raise ValueError("failed-fit behavior requires a diagnostic contract")
        required.update(
            {
                "diagnostic_contract_sha256": sha256_file(diagnostic_contract),
                "post_failure_characterization": True,
                "fit_gates_passed": False,
                "candidate_eligible": False,
                "candidate_may_be_locked": False,
            }
        )
    if evaluation_split == "final_test":
        if editor_lock is None:
            raise ValueError("final_test requires an editor lock")
        required.update(
            {
                "editor_lock_sha256": sha256_file(editor_lock),
                "final_test_open": True,
                "final_test_open_count": 1,
            }
        )
    else:
        required.update({"final_test_open": False, "final_test_open_count": 0})
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"behavior authorization mismatch: {field}")
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--evaluation-split",
        choices=[
            "fit_smoke",
            "selection",
            "final_test",
            "failed_fit_smoke",
            "failed_fit_selection",
        ],
        required=True,
    )
    parser.add_argument("--editor-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    for path in (args.output, args.environment_output, args.identity_output):
        if path.exists():
            raise FileExistsError(f"refusing existing behavior output: {path}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fit_report = json.loads(args.fit_report.read_text())
    editor_contract = json.loads(args.editor_contract.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    design_audit = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    diagnostic = args.evaluation_split.startswith("failed_fit_")
    diagnostic_contract = None
    if diagnostic:
        if args.diagnostic_contract is None:
            raise ValueError("failed-fit behavior requires a diagnostic contract")
        diagnostic_contract = json.loads(args.diagnostic_contract.read_text())
        if diagnostic_contract.get("post_failure_characterization") is not True:
            raise ValueError("diagnostic contract is not a post-failure characterization")
        if diagnostic_contract.get("fit_gates_passed") is not False:
            raise ValueError("diagnostic contract does not preserve fit failure")
        if fit_report.get("all_fit_gates_pass") is not False:
            raise ValueError("failed-fit stage requires a failed fit report")
    elif not fit_report.get("all_fit_gates_pass"):
        raise ValueError("editor fit gates did not pass")
    if fit_report.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("fit report checkpoint SHA mismatch")
    if checkpoint.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("checkpoint editor-contract binding mismatch")
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("model revision mismatch")

    manifest = load_jsonl(args.manifest)
    jobs, design_stage = _jobs(manifest, crossover, args.evaluation_split)
    if design_audit.get("stage") != design_stage or not design_audit.get("audit", {}).get("success"):
        raise ValueError(f"{design_stage} design audit did not pass")
    if design_audit.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("design audit contract SHA mismatch")
    expected_rows = len(jobs)
    expected_key_sha256 = _key_hash(jobs)
    authorization = _validate_authorization(
        args.execution_authorization,
        evaluation_split=args.evaluation_split,
        checkpoint=args.checkpoint,
        fit_report=args.fit_report,
        editor_contract=args.editor_contract,
        crossover_contract=args.crossover_contract,
        manifest=args.manifest,
        design_audit=args.design_audit,
        expected_rows=expected_rows,
        expected_key_sha256=expected_key_sha256,
        editor_lock=args.editor_lock,
        diagnostic_contract=args.diagnostic_contract,
    )

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
        "stage": (
            f"governance_{args.evaluation_split}"
            if diagnostic
            else f"governance_behavior_{args.evaluation_split}"
        ),
        "candidate_id": f"{editor_contract['method_short_name']}-{sha256_file(args.checkpoint)[:16]}",
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(args.fit_report),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "expected_rows": expected_rows,
        "planned_rows": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "device": str(device),
        "torch_version": torch.__version__,
        "base_model_trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "editor_trainable_parameters": adaptive.trainable_parameter_count,
        "post_failure_characterization": diagnostic,
        "fit_gates_passed": bool(fit_report.get("all_fit_gates_pass")),
        "candidate_eligible": False if diagnostic else None,
        "candidate_may_be_locked": False if diagnostic else None,
        "diagnostic_contract_sha256": (
            sha256_file(args.diagnostic_contract) if diagnostic else None
        ),
        "final_test_open": args.evaluation_split == "final_test",
        "final_test_open_count": 1 if args.evaluation_split == "final_test" else 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n"
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
    completed = set()
    identity_checks = []
    identity_done = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as handle:
        for prefix_key in sorted(grouped):
            role, history, style, depth, realization = prefix_key
            messages = prefix_messages(role, history, style, depth, realization)
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(
                model, device, prefix_ids, layers
            )
            boundaries = {
                spec: captured[str(spec.layer)].to(device=device, dtype=torch.float32)
                for spec in specs
            }
            for job in sorted(grouped[prefix_key], key=job_key):
                current_key = job_key(job)
                prompt = job["prompt"]
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(tokenizer, messages + [{"role": "user", "content": prompt["prompt"]}]),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"prefix mismatch for {current_key}")
                baseline_logits, baseline_extended = _suffix_logits_batch(
                    model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                )
                del baseline_extended
                baseline = _margins(baseline_logits[0], prompt)
                if not identity_done:
                    for one_spec in specs:
                        single = MultiLayerContextOperator(
                            {one_spec: manager.operators[one_spec]}
                        ).to(device)
                        single.install(
                            model,
                            gate={one_spec: 0.0},
                            module_resolver=_module,
                            boundary_state={one_spec: boundaries[one_spec]},
                        )
                        try:
                            zero_logits, zero_extended = _suffix_logits_batch(
                                model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                            )
                        finally:
                            single.remove()
                        del zero_extended
                        max_error = max(
                            abs(float(zero_logits[0][0]) - float(baseline_logits[0][0])),
                            abs(float(zero_logits[0][1]) - float(baseline_logits[0][1])),
                        )
                        identity_checks.append(
                            {
                                "site": f"{one_spec.layer}:{one_spec.component}",
                                "job_key": current_key,
                                "selected_logit_max_error": max_error,
                                "success": max_error == 0.0,
                            }
                        )
                    identity_done = True
                    identity_report = {
                        "schema_version": 1,
                        "method": editor_contract["method_short_name"],
                        "checks": identity_checks,
                        "max_error": max(
                            row["selected_logit_max_error"] for row in identity_checks
                        ),
                        "success": all(row["success"] for row in identity_checks),
                        "final_test_open": args.evaluation_split == "final_test",
                        "production_rollout_approved": False,
                    }
                    atomic_write_text(
                        args.identity_output,
                        json.dumps(identity_report, indent=2, sort_keys=True) + "\n",
                    )
                    if not identity_report["success"]:
                        raise RuntimeError("zero-gate identity audit failed")
                gates = {spec: 1.0 for spec in specs}
                manager.install(
                    model,
                    gate=gates,
                    module_resolver=_module,
                    boundary_state=boundaries,
                    collect_diagnostics=True,
                )
                try:
                    edited_logits, edited_extended = _suffix_logits_batch(
                        model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                    )
                finally:
                    manager.remove()
                del edited_extended
                edited = _margins(edited_logits[0], prompt)
                controller = {}
                for spec in specs:
                    site_editor = manager.operators[spec]
                    diagnostics = site_editor.last_diagnostics
                    controller[f"{spec.layer}:{spec.component}"] = {
                        name: float(value.reshape(-1)[0].detach().cpu())
                        for name, value in diagnostics.items()
                        if name
                        in {
                            "history_probability",
                            "task_probability",
                            "signed_governance_factor",
                            "trust_scale",
                        }
                    }
                record = {
                    "job_key": current_key,
                    "method": editor_contract["method_short_name"],
                    "evaluation_split": args.evaluation_split,
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
                    "factual_correct_label": prompt["factual_correct_label"],
                    "user_selected_label": prompt["user_selected_label"],
                    "candidate_payload_sha256": prompt["candidate_payload_sha256"],
                    "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                    "prefix_tokens": len(prefix_ids),
                    "suffix_tokens": len(suffix_ids),
                    "baseline": baseline,
                    "edited": edited,
                    "delta_task_aligned_margin": edited["task_aligned_margin"]
                    - baseline["task_aligned_margin"],
                    "delta_factual_margin": edited["factual_margin"] - baseline["factual_margin"],
                    "delta_user_choice_margin": edited["user_choice_margin"]
                    - baseline["user_choice_margin"],
                    "task_aligned_binary_kl": _binary_kl(
                        _binary_probability(baseline["task_aligned_margin"]),
                        _binary_probability(edited["task_aligned_margin"]),
                    ),
                    "controller_diagnostics": controller,
                    "intervention_diagnostics": {
                        f"{spec.layer}:{spec.component}": values
                        for spec, values in manager.last_stats.items()
                    },
                }
                finite = [
                    *baseline.values(),
                    *edited.values(),
                    record["delta_task_aligned_margin"],
                    record["delta_factual_margin"],
                    record["delta_user_choice_margin"],
                    record["task_aligned_binary_kl"],
                ]
                if not all(math.isfinite(float(value)) for value in finite):
                    raise FloatingPointError(f"non-finite editor output for {current_key}")
                handle.write(canonical_json(record) + "\n")
                completed.add(current_key)
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    observed_sha = hashlib.sha256(
        ("\n".join(sorted(completed)) + "\n").encode("utf-8")
    ).hexdigest()
    if len(completed) != expected_rows or observed_sha != expected_key_sha256:
        raise RuntimeError("behavior output row/key audit failed")
    print(
        json.dumps(
            {
                "completed_unique_rows": len(completed),
                "expected_key_sha256": expected_key_sha256,
                "authorization_id": authorization.get("authorization_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
