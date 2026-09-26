#!/usr/bin/env python3
"""Run one immutable operator candidate on one frozen operator-dev half."""

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
from scripts.benchmark_v1.operators import (
    ContextConditionedLowRankTransport,
    FixedDirectionTranslation,
    LayerOperatorSpec,
    MismatchGate,
    MultiLayerContextOperator,
    OneSidedProtectedProjection,
    SignedGovernanceAlignmentGate,
)
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


def _load_torch(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _half_jobs(jobs: list[dict], half: str) -> list[dict]:
    selected_ids = set()
    by_benchmark = defaultdict(set)
    for job in jobs:
        by_benchmark[job["item"]["benchmark"]].add(job["item"]["item_id"])
    for benchmark, item_ids in sorted(by_benchmark.items()):
        ordered = sorted(item_ids)
        if len(ordered) != 32:
            raise ValueError(f"operator_dev:{benchmark} has {len(ordered)} items instead of 32")
        chosen = ordered[:16] if half == "screening" else ordered[16:]
        selected_ids.update((benchmark, item_id) for item_id in chosen)
    result = [
        job
        for job in jobs
        if (job["item"]["benchmark"], job["item"]["item_id"]) in selected_ids
    ]
    if len(result) != 3072:
        raise ValueError(f"operator-dev {half} has {len(result)} rows instead of 3072")
    return result


def _evaluation_jobs(all_jobs: list[dict], evaluation_split: str) -> list[dict]:
    if evaluation_split in {"screening", "selection"}:
        return _half_jobs(all_jobs, evaluation_split)
    if evaluation_split == "final_test":
        if len(all_jobs) != 6144:
            raise ValueError(f"final_test has {len(all_jobs)} rows instead of 6144")
        return all_jobs
    raise ValueError(f"unknown evaluation split: {evaluation_split}")


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        (("\n".join(sorted(job_key(job) for job in jobs))) + "\n").encode("utf-8")
    ).hexdigest()


def _validate_authorization(
    path: Path,
    *,
    evaluation_split: str,
    candidate_manifest: Path,
    candidate_tensor: Path,
    contract: Path,
    manifest: Path,
    design_audit: Path,
    expected_key_sha256: str,
    operator_lock: Path | None,
) -> dict:
    value = json.loads(path.read_text())
    expected_rows = 6144 if evaluation_split == "final_test" else 3072
    required = {
        "stage": (
            "operator_final_test"
            if evaluation_split == "final_test"
            else f"operator_dev_{evaluation_split}"
        ),
        "candidate_manifest_sha256": sha256_file(candidate_manifest),
        "candidate_tensor_sha256": sha256_file(candidate_tensor),
        "crossover_contract_sha256": sha256_file(contract),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "design_audit_sha256": sha256_file(design_audit),
        "expected_key_sha256": expected_key_sha256,
        "expected_rows": expected_rows,
        "production_rollout_approved": False,
    }
    if evaluation_split == "final_test":
        if operator_lock is None:
            raise ValueError("final_test requires the immutable operator lock")
        required.update(
            {
                "operator_lock_sha256": sha256_file(operator_lock),
                "final_test_open": True,
                "final_test_open_count": 1,
            }
        )
    else:
        required["final_test_open"] = False
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"operator authorization field mismatch: {field}")
    if not value.get("execution_allowed"):
        raise ValueError("operator execution is not authorized")
    return value


def _build_operator(candidate: dict, device: torch.device):
    operators = {}
    gates = {}
    for key in candidate["site_order"]:
        site = candidate["sites"][key]
        spec = LayerOperatorSpec(
            layer=int(site["layer"]),
            component=str(site["component"]),
            rank=int(site["history_basis"].shape[1]),
        )
        if site["operator_kind"] == "fixed_translation":
            operators[spec] = FixedDirectionTranslation(
                direction=site["output_basis"][:, 0],
                strength=float(site["projection_strength"]),
                max_relative_correction=float(site["max_relative_correction"]),
            )
        elif site["operator_kind"] == "projection":
            rank = int(site["output_basis"].shape[1])
            operators[spec] = OneSidedProtectedProjection(
                harmful_basis=site["output_basis"],
                threshold=site["projection_threshold"],
                strength=torch.full((rank,), float(site["projection_strength"])),
                center=site["component_center"],
                protected_basis=None,
                one_sided=bool(site["one_sided"]),
                max_relative_correction=float(site["max_relative_correction"]),
            )
        elif site["operator_kind"] == "context_transport":
            operators[spec] = ContextConditionedLowRankTransport(
                history_trigger_basis=site["history_basis"],
                output_basis=site["output_basis"],
                transport=site["transport"],
                threshold=site["history_threshold"],
                context_basis=site["context_basis"],
                history_center=site["history_center"],
                context_center=site["context_center"],
                context_scale=site["context_scale"],
                history_reference=site.get("history_reference"),
                max_relative_correction=float(site["site_trust_radius"]),
                one_sided=True,
                edit_last_token_only=bool(site["edit_last_token_only"]),
            )
        else:
            raise ValueError(f"unknown materialized operator kind: {site['operator_kind']}")
        if candidate["config"]["family"] == "v5_signed":
            gates[spec] = SignedGovernanceAlignmentGate(
                site["gate_weight"],
                history_bias=float(site["gate_bias"]),
                deadzone=float(site["deadzone"]),
            )
        else:
            gates[spec] = MismatchGate(
                site["gate_weight"], history_bias=float(site["gate_bias"])
            )
    manager = MultiLayerContextOperator(operators).to(device)
    for gate in gates.values():
        gate.to(device)
    return manager, gates


def _gate_values(
    candidate: dict,
    gates: dict,
    boundaries: dict[LayerOperatorSpec, torch.Tensor],
    target_obedience: float,
    applicable: float = 1.0,
) -> tuple[dict[LayerOperatorSpec, torch.Tensor], dict[str, dict[str, float]]]:
    values = {}
    diagnostics = {}
    for spec, gate_model in gates.items():
        boundary = boundaries[spec]
        if candidate["config"]["family"] == "v5_signed":
            probability = gate_model.history_probability(boundary)
            gate = gate_model(
                boundary,
                target_obedience=target_obedience,
                applicable=applicable,
            )
        else:
            probability = torch.sigmoid(
                boundary.float() @ gate_model.history_weight.float() + gate_model.history_bias
            )
            gate = (
                gate_model(
                    boundary,
                    verification_required=torch.tensor(
                        1.0 - target_obedience, device=boundary.device
                    ),
                )
                * float(applicable)
            )
        values[spec] = gate
        diagnostics[f"{spec.layer}:{spec.component}"] = {
            "history_obedience_probability": float(probability.detach().cpu()),
            "signed_gate": float(gate.detach().cpu()),
            "target_obedience": float(target_obedience),
            "applicable": float(applicable),
        }
    return values, diagnostics


def _margins(logits: tuple[float, float], spec: dict) -> dict:
    values = {"A": float(logits[0]), "B": float(logits[1])}
    return {
        "logit_a": values["A"],
        "logit_b": values["B"],
        "task_aligned_margin": values[spec["task_correct_label"]]
        - values[spec["task_foil_label"]],
        "factual_margin": values[spec["factual_correct_label"]]
        - values[spec["factual_foil_label"]],
        "user_choice_margin": values[spec["user_selected_label"]]
        - values[spec["factual_correct_label"]],
    }


def _binary_probability(margin: float) -> float:
    return 1.0 / (1.0 + math.exp(max(-700.0, min(700.0, -float(margin)))))


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
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--evaluation-split",
        choices=["screening", "selection", "final_test"],
        required=True,
    )
    parser.add_argument("--operator-lock", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    candidate = _load_torch(args.candidate_tensor)
    candidate_manifest = json.loads(args.candidate_manifest.read_text())
    contract = json.loads(args.crossover_contract.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    design_audit = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if candidate_manifest["tensor_sha256"] != sha256_file(args.candidate_tensor):
        raise ValueError("candidate tensor hash mismatch")
    if candidate_manifest["candidate_id"] != candidate["candidate_id"]:
        raise ValueError("candidate identity mismatch")
    if manifest_report["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    expected_design_stage = (
        "final_test" if args.evaluation_split == "final_test" else "operator_dev"
    )
    if design_audit.get("stage") != expected_design_stage or not design_audit.get("audit", {}).get("success"):
        raise ValueError(f"{expected_design_stage} design audit did not pass")
    if design_audit["contract_sha256"] != sha256_file(args.crossover_contract):
        raise ValueError("operator-dev design audit contract mismatch")
    if design_audit["manifest_sha256"] != sha256_file(args.manifest):
        raise ValueError("operator-dev design audit manifest mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != contract["base_model"]["revision"]:
        raise ValueError("model revision mismatch")
    if not (model_manifest.get("non_quantized") or model_manifest.get("quantization") == "none"):
        raise ValueError("quantized checkpoints are forbidden")
    contract_stage = "final_test" if args.evaluation_split == "final_test" else "operator_dev"
    all_jobs = enumerate_jobs(load_jsonl(args.manifest), contract, contract_stage)
    jobs = _evaluation_jobs(all_jobs, args.evaluation_split)
    expected_key_sha256 = _key_hash(jobs)
    if args.evaluation_split == "final_test":
        if args.operator_lock is None:
            raise ValueError("final_test requires --operator-lock")
        operator_lock = json.loads(args.operator_lock.read_text())
        if operator_lock.get("locked") is not True:
            raise ValueError("operator lock is not immutable")
        if operator_lock.get("chosen_candidate_id") != candidate["candidate_id"]:
            raise ValueError("final-test candidate does not match operator lock")
        if operator_lock.get("candidate_tensor_sha256") != sha256_file(args.candidate_tensor):
            raise ValueError("operator lock tensor binding mismatch")
        if operator_lock.get("candidate_manifest_sha256") != sha256_file(args.candidate_manifest):
            raise ValueError("operator lock manifest binding mismatch")
    authorization = _validate_authorization(
        args.execution_authorization,
        evaluation_split=args.evaluation_split,
        candidate_manifest=args.candidate_manifest,
        candidate_tensor=args.candidate_tensor,
        contract=args.crossover_contract,
        manifest=args.manifest,
        design_audit=args.design_audit,
        expected_key_sha256=expected_key_sha256,
        operator_lock=args.operator_lock,
    )

    completed = set()
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f"refusing to overwrite {args.output}")
        completed = {row["job_key"] for row in load_jsonl(args.output)}
    for path in (args.environment_output, args.identity_output):
        if path.exists() and not args.resume:
            raise FileExistsError(f"refusing existing output: {path}")
    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    label_ids = _label_ids(tokenizer)
    manager, gate_models = _build_operator(candidate, device)
    specs = list(manager.operators)
    layers = sorted({spec.layer for spec in specs})
    environment = {
        "schema_version": 1,
        "stage": (
            "operator_final_test"
            if args.evaluation_split == "final_test"
            else f"operator_dev_{args.evaluation_split}"
        ),
        "candidate_id": candidate["candidate_id"],
        "candidate_tensor_sha256": sha256_file(args.candidate_tensor),
        "candidate_manifest_sha256": sha256_file(args.candidate_manifest),
        "authorization_sha256": sha256_file(args.execution_authorization),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "expected_key_sha256": expected_key_sha256,
        "planned_rows": len(jobs),
        "device": str(device),
        "torch_version": torch.__version__,
        "model_class": type(model).__name__,
        "final_test_open": args.evaluation_split == "final_test",
        "production_rollout_approved": False,
    }
    if args.evaluation_split == "final_test":
        environment["operator_lock_sha256"] = sha256_file(args.operator_lock)
    if not args.environment_output.exists():
        atomic_write_text(
            args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n"
        )

    grouped = defaultdict(list)
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
    identity_checks = [] if not args.identity_output.exists() else json.loads(args.identity_output.read_text())["checks"]
    identity_done = bool(identity_checks)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a", buffering=1) as output_handle:
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
                if current_key in completed:
                    continue
                spec = job["prompt"]
                rendered_prefix, suffix_ids = _split_final_user(
                    tokenizer,
                    _chat_ids(
                        tokenizer,
                        messages + [{"role": "user", "content": spec["prompt"]}],
                    ),
                )
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"prefix mismatch for {current_key}")
                baseline_logits, baseline_extended = _suffix_logits_batch(
                    model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                )
                del baseline_extended
                baseline = _margins(baseline_logits[0], spec)
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
                                model,
                                device,
                                base_cache,
                                len(prefix_ids),
                                [suffix_ids],
                                label_ids,
                                0,
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
                        "candidate_id": candidate["candidate_id"],
                        "checks": identity_checks,
                        "max_error": max(row["selected_logit_max_error"] for row in identity_checks),
                        "success": all(row["success"] for row in identity_checks),
                        "final_test_open": args.evaluation_split == "final_test",
                        "production_rollout_approved": False,
                    }
                    atomic_write_text(
                        args.identity_output,
                        json.dumps(identity_report, indent=2, sort_keys=True) + "\n",
                    )
                    if not identity_report["success"]:
                        raise RuntimeError("zero-gate identity check failed")
                gates, gate_diagnostics = _gate_values(
                    candidate, gate_models, boundaries, float(spec["target_obedience"])
                )
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
                edited = _margins(edited_logits[0], spec)
                record = {
                    "job_key": current_key,
                    "candidate_id": candidate["candidate_id"],
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
                    "target_obedience": spec["target_obedience"],
                    "label_swap": job["label_swap"],
                    "task_correct_label": spec["task_correct_label"],
                    "factual_correct_label": spec["factual_correct_label"],
                    "user_selected_label": spec["user_selected_label"],
                    "candidate_payload_sha256": spec["candidate_payload_sha256"],
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
                    "gate_diagnostics": gate_diagnostics,
                    "intervention_diagnostics": {
                        f"{site.layer}:{site.component}": values
                        for site, values in manager.last_stats.items()
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
                    raise FloatingPointError(f"non-finite operator output for {current_key}")
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
                "evaluation_split": args.evaluation_split,
                "completed_unique_rows": len(completed),
                "expected_rows": len(jobs),
                "authorization_id": authorization.get("authorization_id"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if len(completed) != len(jobs):
        raise SystemExit("operator candidate run ended without every expected row")


if __name__ == "__main__":
    main()
