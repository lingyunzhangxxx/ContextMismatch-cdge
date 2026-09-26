#!/usr/bin/env python3
"""Run one DSGE-V3 single-site checkpoint on a frozen developmental split."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from scripts.benchmark_v1.common import (
    atomic_write_text,
    canonical_json,
    load_jsonl,
    sha256_file,
)
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
from scripts.benchmark_v3.fit_governance_editor import _fold
from scripts.benchmark_v4.directional_governance import editor_from_checkpoint


def _jobs(manifest: list[dict], contract: dict, split: str) -> tuple[list[dict], str]:
    if split == "fit_audit":
        full = enumerate_jobs(manifest, contract, "replication")
        rows = [job for job in full if _fold(job["item"]["item_id"]) == 7]
        if not rows:
            raise ValueError("directional fit-audit fold is empty")
        return rows, "replication"
    if split == "operator_dev":
        return (
            _half_jobs(enumerate_jobs(manifest, contract, "operator_dev"), "selection"),
            "operator_dev",
        )
    raise ValueError(f"unsupported directional evaluation split: {split}")


def _candidate_report(fit_report: dict, checkpoint_sha256: str) -> dict:
    matches = [
        row
        for row in fit_report.get("candidate_reports", [])
        if row.get("checkpoint_sha256") == checkpoint_sha256
    ]
    if len(matches) != 1:
        raise ValueError("fit report does not bind exactly one candidate checkpoint")
    return matches[0]


def _validate_authorization(
    path: Path,
    *,
    split: str,
    checkpoint: Path,
    fit_report: Path,
    editor_contract: Path,
    crossover_contract: Path,
    manifest: Path,
    design_audit: Path,
    model_manifest: Path,
    expected_rows: int,
    expected_key_sha256: str,
    prior_analysis: Path | None,
) -> dict:
    authorization = json.loads(path.read_text())
    required = {
        "stage": f"governance_directional_{split}",
        "execution_allowed": True,
        "checkpoint_sha256": sha256_file(checkpoint),
        "fit_report_sha256": sha256_file(fit_report),
        "editor_contract_sha256": sha256_file(editor_contract),
        "crossover_contract_sha256": sha256_file(crossover_contract),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "design_audit_sha256": sha256_file(design_audit),
        "model_manifest_sha256": sha256_file(model_manifest),
        "expected_rows": expected_rows,
        "expected_key_sha256": expected_key_sha256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if split == "operator_dev":
        if prior_analysis is None:
            raise ValueError("operator-dev evaluation requires the fit-audit analysis")
        required["fit_audit_analysis_sha256"] = sha256_file(prior_analysis)
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"directional behavior authorization mismatch: {field}")
    code_root = str(authorization.get("code_root", ""))
    allowed_roots = (
        "/workspace/context-mismatch-qwen3-8b/code-v",
        "/workspace/context-mismatch-qwen3-5-9b/code-v",
    )
    if not code_root.startswith(allowed_roots):
        raise ValueError("directional behavior code root is invalid")
    if int(code_root.rsplit("code-v", 1)[1]) < 29:
        raise ValueError("directional behavior requires code-v29 or newer")
    bundle = Path(code_root) / "bundle.sha256"
    if not bundle.is_file():
        raise ValueError("directional behavior immutable bundle manifest is missing")
    if authorization.get("immutable_code_bundle_manifest_sha256") != sha256_file(bundle):
        raise ValueError("directional behavior bundle SHA mismatch")
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--evaluation-split", choices=["fit_audit", "operator_dev"], required=True)
    parser.add_argument("--prior-analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    for path in (args.output, args.environment_output, args.identity_output):
        if path.exists():
            raise FileExistsError(f"refusing existing directional behavior output: {path}")

    checkpoint_sha = sha256_file(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    fit_report = json.loads(args.fit_report.read_text())
    contract = json.loads(args.editor_contract.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    design_audit = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
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
    if len(fit_report.get("candidate_reports", [])) != 3:
        raise ValueError("directional fit report candidate set is incomplete")
    if fit_report.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("fit report editor-contract binding mismatch")
    candidate = _candidate_report(fit_report, checkpoint_sha)
    if checkpoint.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("checkpoint editor-contract binding mismatch")
    if args.evaluation_split == "operator_dev":
        if candidate.get("fit_eligible") is not True:
            raise ValueError("fit-ineligible candidate cannot advance to operator_dev")
        if args.prior_analysis is None:
            raise ValueError("operator_dev requires a fit-audit analysis")
        prior = json.loads(args.prior_analysis.read_text())
        for field, expected in {
            "method": "DSGE-V3",
            "evaluation_stage": "governance_directional_fit_audit",
            "checkpoint_sha256": checkpoint_sha,
            "fit_report_sha256": sha256_file(args.fit_report),
            "editor_contract_sha256": sha256_file(args.editor_contract),
            "fit_eligible": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }.items():
            if prior.get(field) != expected:
                raise ValueError(f"fit-audit analysis mismatch: {field}")
        if prior.get("audit", {}).get("success") is not True:
            raise ValueError("fit-audit analysis is incomplete")
        if prior.get("selection_gate", {}).get("candidate_eligible") is not True:
            raise ValueError("candidate did not pass the frozen fit-audit behavior gate")
    elif args.prior_analysis is not None:
        raise ValueError("fit_audit must not receive prior behavior evidence")
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

    manifest = load_jsonl(args.manifest)
    jobs, design_stage = _jobs(manifest, crossover, args.evaluation_split)
    if design_audit.get("stage") != design_stage or not design_audit.get("audit", {}).get(
        "success"
    ):
        raise ValueError(f"{design_stage} design audit did not pass")
    if design_audit.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("design audit contract SHA mismatch")
    expected_rows = len(jobs)
    expected_key_sha = _key_hash(jobs)
    authorization = _validate_authorization(
        args.execution_authorization,
        split=args.evaluation_split,
        checkpoint=args.checkpoint,
        fit_report=args.fit_report,
        editor_contract=args.editor_contract,
        crossover_contract=args.crossover_contract,
        manifest=args.manifest,
        design_audit=args.design_audit,
        model_manifest=args.model_manifest,
        expected_rows=expected_rows,
        expected_key_sha256=expected_key_sha,
        prior_analysis=args.prior_analysis,
    )

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("failed to freeze base model")
    directional = editor_from_checkpoint(checkpoint).to(device).eval()
    manager = directional.hook_manager().to(device)
    specs = list(manager.operators)
    if len(specs) != 1:
        raise RuntimeError("directional behavior requires exactly one editor site")
    spec = specs[0]
    layers = [spec.layer]
    label_ids = _label_ids(tokenizer)
    environment = {
        "schema_version": 1,
        "stage": f"governance_directional_{args.evaluation_split}",
        "candidate_id": candidate["candidate_id"],
        "site": candidate["site"],
        "checkpoint_sha256": checkpoint_sha,
        "fit_report_sha256": sha256_file(args.fit_report),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "expected_rows": expected_rows,
        "planned_rows": expected_rows,
        "expected_key_sha256": expected_key_sha,
        "device": str(device),
        "torch_version": torch.__version__,
        "base_model_trainable_parameters": 0,
        "editor_trainable_parameters": directional.trainable_parameter_count,
        "fit_eligible": bool(candidate["fit_eligible"]),
        "all_candidates_require_fit_audit": args.evaluation_split == "fit_audit",
        "final_test_open": False,
        "final_test_open_count": 0,
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
    completed: set[str] = set()
    identity_checks = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", buffering=1) as handle:
        for prefix_key in sorted(grouped):
            role, history, style, depth, realization = prefix_key
            messages = prefix_messages(role, history, style, depth, realization)
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(
                model, device, prefix_ids, layers
            )
            boundary = captured[str(spec.layer)].to(device=device, dtype=torch.float32)
            first_in_prefix = True
            for job in sorted(grouped[prefix_key], key=job_key):
                current_key = job_key(job)
                prompt = job["prompt"]
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(
                        tokenizer,
                        messages + [{"role": "user", "content": prompt["prompt"]}],
                    ),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"prefix mismatch for {current_key}")
                baseline_logits, baseline_extended = _suffix_logits_batch(
                    model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                )
                del baseline_extended
                baseline = _margins(baseline_logits[0], prompt)
                if first_in_prefix:
                    zero = MultiLayerContextOperator({spec: manager.operators[spec]}).to(device)
                    zero.install(
                        model,
                        gate={spec: 0.0},
                        module_resolver=_module,
                        boundary_state={spec: boundary},
                    )
                    try:
                        zero_logits, zero_extended = _suffix_logits_batch(
                            model,
                            device,
                            base_cache,
                            len(prefix_ids),
                            [suffix_ids],
                            label_ids,
                            0,
                        )
                    finally:
                        zero.remove()
                    del zero_extended
                    error = max(
                        abs(float(zero_logits[0][0]) - float(baseline_logits[0][0])),
                        abs(float(zero_logits[0][1]) - float(baseline_logits[0][1])),
                    )
                    identity_checks.append(
                        {
                            "prefix": {
                                "declared_role": role,
                                "history_condition": history,
                                "history_style": style,
                                "history_realization": realization,
                            },
                            "job_key": current_key,
                            "selected_logit_max_error": error,
                            "success": error == 0.0,
                        }
                    )
                    if error != 0.0:
                        raise RuntimeError("directional external-zero identity failed")
                    first_in_prefix = False

                manager.install(
                    model,
                    gate={spec: 1.0},
                    module_resolver=_module,
                    boundary_state={spec: boundary},
                    collect_diagnostics=True,
                )
                try:
                    edited_logits, edited_extended = _suffix_logits_batch(
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
                del edited_extended
                edited = _margins(edited_logits[0], prompt)
                diagnostics = manager.operators[spec].last_diagnostics
                controller = {
                    name: float(value.reshape(-1)[0].detach().cpu())
                    for name, value in diagnostics.items()
                    if name
                    in {
                        "history_probability",
                        "task_probability",
                        "positive_evidence",
                        "negative_evidence",
                        "positive_route",
                        "negative_route",
                        "trust_scale",
                    }
                }
                matched = (
                    (history == "obedience" and float(prompt["target_obedience"]) == 1.0)
                    or (history == "verification" and float(prompt["target_obedience"]) == 0.0)
                )
                record = {
                    "job_key": current_key,
                    "method": "DSGE-V3",
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
                    "matched_context": matched,
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
                    "selected_logit_max_error": max(
                        abs(edited["logit_a"] - baseline["logit_a"]),
                        abs(edited["logit_b"] - baseline["logit_b"]),
                    ),
                    "delta_task_aligned_margin": edited["task_aligned_margin"]
                    - baseline["task_aligned_margin"],
                    "delta_factual_margin": edited["factual_margin"]
                    - baseline["factual_margin"],
                    "delta_user_choice_margin": edited["user_choice_margin"]
                    - baseline["user_choice_margin"],
                    "task_aligned_binary_kl": _binary_kl(
                        _binary_probability(baseline["task_aligned_margin"]),
                        _binary_probability(edited["task_aligned_margin"]),
                    ),
                    "controller_diagnostics": {candidate["site"]: controller},
                    "intervention_diagnostics": {
                        candidate["site"]: manager.last_stats[spec]
                    },
                }
                finite = [
                    *baseline.values(),
                    *edited.values(),
                    record["selected_logit_max_error"],
                    record["delta_task_aligned_margin"],
                    record["delta_factual_margin"],
                    record["delta_user_choice_margin"],
                    record["task_aligned_binary_kl"],
                ]
                if not all(math.isfinite(float(value)) for value in finite):
                    raise FloatingPointError(f"non-finite directional output: {current_key}")
                handle.write(canonical_json(record) + "\n")
                completed.add(current_key)
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    observed_sha = hashlib.sha256(
        ("\n".join(sorted(completed)) + "\n").encode("utf-8")
    ).hexdigest()
    if len(completed) != expected_rows or observed_sha != expected_key_sha:
        raise RuntimeError("directional behavior row/key audit failed")
    identity_report = {
        "schema_version": 1,
        "method": "DSGE-V3",
        "checkpoint_sha256": checkpoint_sha,
        "checks": identity_checks,
        "max_error": max(row["selected_logit_max_error"] for row in identity_checks),
        "success": all(row["success"] for row in identity_checks),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.identity_output,
        json.dumps(identity_report, indent=2, sort_keys=True) + "\n",
    )
    print(
        json.dumps(
            {
                "completed_unique_rows": len(completed),
                "expected_key_sha256": expected_key_sha,
                "authorization_id": authorization.get("authorization_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
