#!/usr/bin/env python3
"""Run one immutable two-method shard on the frozen DGE operator-dev identity."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import statistics
import time
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
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v2.run_operator_candidate import (
    _binary_kl,
    _binary_probability,
    _build_operator,
    _gate_values,
    _half_jobs,
    _load_torch,
    _margins,
)
from scripts.benchmark_v4.directional_governance import (
    editor_from_checkpoint as v3_editor_from_checkpoint,
)
from scripts.benchmark_v6.consensus_directional_governance import (
    editor_from_checkpoint as v5_editor_from_checkpoint,
)
from scripts.benchmark_v7.same_identity_variants import (
    DGEAblationSiteEditor,
    V3_ABLATION_METHODS,
)


EXPECTED_ROWS = 3072
EXPECTED_KEY_SHA256 = "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e"
METHOD_SHARDS = {
    "operators": ("fixed_negative_vector", "symmetric_rank_one"),
    "experts": ("shared_single_expert", "positive_only_expert"),
    "routing": ("negative_only_expert", "dge_without_structural_routing"),
    "v5": ("full_dge", "consensus_v5"),
}
CANDIDATE_METHODS = {"fixed_negative_vector", "symmetric_rank_one"}
V5_COMPARISON_METHODS = {"full_dge", "consensus_v5"}


def _sync(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _memory_allocated(device: torch.device) -> int:
    if device.type == "npu" and hasattr(torch, "npu"):
        return int(torch.npu.memory_allocated(device))
    if device.type == "cuda":
        return int(torch.cuda.memory_allocated(device))
    return 0


def _reset_peak(device: torch.device) -> None:
    if device.type == "npu" and hasattr(torch, "npu"):
        torch.npu.reset_peak_memory_stats(device)
    elif device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _peak_memory(device: torch.device) -> int:
    if device.type == "npu" and hasattr(torch, "npu"):
        return int(torch.npu.max_memory_allocated(device))
    if device.type == "cuda":
        return int(torch.cuda.max_memory_allocated(device))
    return 0


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
    ).hexdigest()


def _source(path_record: dict, label: str) -> Path:
    path = Path(str(path_record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"missing {label}: {path}")
    if sha256_file(path) != path_record.get("sha256"):
        raise ValueError(f"{label} SHA mismatch")
    return path


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(quantile * len(ordered))) - 1))
    return float(ordered[index])


def _validate_authorization(
    path: Path,
    *,
    code_root: Path,
    supplement: Path,
    crossover: Path,
    manifest: Path,
    design_audit: Path,
    model_manifest: Path,
) -> tuple[dict, tuple[str, str]]:
    value = json.loads(path.read_text())
    shard = str(value.get("method_shard", ""))
    if shard not in METHOD_SHARDS:
        raise ValueError("unknown same-identity method shard")
    methods = tuple(value.get("methods", []))
    if methods != METHOD_SHARDS[shard]:
        raise ValueError("same-identity method list is not frozen")
    stage = (
        "governance_v5_same_identity_behavior"
        if shard == "v5"
        else "dge_same_identity_supplement"
    )
    required = {
        "stage": stage,
        "code_root": str(code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(code_root / "bundle.sha256"),
        "execution_allowed": True,
        "supplement_contract_sha256": sha256_file(supplement),
        "crossover_contract_sha256": sha256_file(crossover),
        "benchmark_manifest_sha256": sha256_file(manifest),
        "design_audit_sha256": sha256_file(design_audit),
        "model_manifest_sha256": sha256_file(model_manifest),
        "expected_rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"same-identity authorization mismatch: {field}")
    return value, methods


def _candidate_manager(record: dict, device: torch.device):
    tensor_path = _source(record["tensor"], "operator tensor")
    manifest_path = _source(record["manifest"], "operator manifest")
    candidate = _load_torch(tensor_path)
    candidate_manifest = json.loads(manifest_path.read_text())
    if candidate_manifest.get("candidate_id") != candidate.get("candidate_id"):
        raise ValueError("operator candidate identity mismatch")
    if candidate_manifest.get("tensor_sha256") != sha256_file(tensor_path):
        raise ValueError("operator tensor/manifest binding mismatch")
    manager, gates = _build_operator(candidate, device)
    return manager, gates, candidate_manifest


def _variant_manager(
    method: str,
    *,
    v3_checkpoint_path: Path | None,
    v5_checkpoint_path: Path | None,
    device: torch.device,
):
    if method == "full_dge":
        if v3_checkpoint_path is None:
            raise ValueError("full DGE requires the V3 checkpoint")
        frozen = v3_editor_from_checkpoint(_load_torch(v3_checkpoint_path))
        return frozen.hook_manager().to(device), {}, {
            "candidate_id": method,
            "tensor_sha256": sha256_file(v3_checkpoint_path),
        }
    if method == "consensus_v5":
        if v5_checkpoint_path is None:
            raise ValueError("consensus V5 requires the V5 checkpoint")
        frozen = v5_editor_from_checkpoint(_load_torch(v5_checkpoint_path))
        return frozen.hook_manager().to(device), {}, {
            "candidate_id": method,
            "tensor_sha256": sha256_file(v5_checkpoint_path),
        }
    if method not in V3_ABLATION_METHODS:
        raise ValueError(f"unsupported same-identity method: {method}")
    if v3_checkpoint_path is None:
        raise ValueError("DGE ablations require the V3 checkpoint")
    checkpoint = _load_torch(v3_checkpoint_path)
    frozen = v3_editor_from_checkpoint(checkpoint)
    wrapper = DGEAblationSiteEditor(frozen, method).to(device).eval()
    site = frozen.site
    spec = LayerOperatorSpec(site.layer, site.component, max(site.positive_rank, site.negative_rank))
    return MultiLayerContextOperator({spec: wrapper}).to(device), {}, {
        "candidate_id": method,
        "tensor_sha256": sha256_file(v3_checkpoint_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--supplement-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing existing output directory: {args.output_dir}")

    authorization, methods = _validate_authorization(
        args.execution_authorization,
        code_root=args.code_root,
        supplement=args.supplement_contract,
        crossover=args.crossover_contract,
        manifest=args.manifest,
        design_audit=args.design_audit,
        model_manifest=args.model_manifest,
    )
    manifest_report = json.loads(args.manifest_report.read_text())
    design = json.loads(args.design_audit.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev design audit is invalid")
    if design.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("operator-dev design/crossover mismatch")
    if not (model_manifest.get("verified") or model_manifest.get("success")):
        raise ValueError("model manifest is unverified")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("base model revision mismatch")
    jobs = _half_jobs(
        enumerate_jobs(load_jsonl(args.manifest), crossover, "operator_dev"), "selection"
    )
    if len(jobs) != EXPECTED_ROWS or _key_hash(jobs) != EXPECTED_KEY_SHA256:
        raise ValueError("frozen same-identity job set changed")

    v3_checkpoint_path = None
    v5_checkpoint_path = None
    if any(method in V3_ABLATION_METHODS | {"full_dge"} for method in methods):
        v3_checkpoint_path = _source(
            authorization["sources"]["v3_checkpoint"], "V3 checkpoint"
        )
        _source(authorization["sources"]["v3_fit_report"], "V3 fit report")
    if "consensus_v5" in methods:
        v5_checkpoint_path = _source(
            authorization["sources"]["v5_checkpoint"], "V5 checkpoint"
        )
        _source(authorization["sources"]["v5_fit_report"], "V5 fit report")

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model was not frozen")
    label_ids = _label_ids(tokenizer)
    managers = {}
    gate_models = {}
    method_metadata = {}
    for method in methods:
        if method in CANDIDATE_METHODS:
            manager, gates, metadata = _candidate_manager(
                authorization["sources"]["operators"][method], device
            )
        else:
            manager, gates, metadata = _variant_manager(
                method,
                v3_checkpoint_path=v3_checkpoint_path,
                v5_checkpoint_path=v5_checkpoint_path,
                device=device,
            )
        managers[method] = manager.to(device)
        gate_models[method] = gates
        method_metadata[method] = metadata

    args.output_dir.mkdir(parents=True)
    rows_dir = args.output_dir / "rows"
    environment_dir = args.output_dir / "environment"
    identity_dir = args.output_dir / "identity"
    analysis_dir = args.output_dir / "analysis"
    for directory in (rows_dir, environment_dir, identity_dir, analysis_dir):
        directory.mkdir()
    for method in methods:
        environment = {
            "schema_version": 1,
            "stage": authorization["stage"],
            "candidate_id": method,
            "method": method,
            "method_shard": authorization["method_shard"],
            "authorization_sha256": sha256_file(args.execution_authorization),
            "expected_key_sha256": EXPECTED_KEY_SHA256,
            "planned_rows": EXPECTED_ROWS,
            "device": str(device),
            "torch_version": torch.__version__,
            "source_metadata": method_metadata[method],
            "base_model_trainable_parameters": 0,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        atomic_write_text(
            environment_dir / f"{method}.json",
            json.dumps(environment, indent=2, sort_keys=True) + "\n",
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
    outputs = {method: (rows_dir / f"{method}.jsonl").open("x", buffering=1) for method in methods}
    completed = {method: set() for method in methods}
    identity_checks = {method: [] for method in methods}
    elapsed = {method: [] for method in methods}
    peak_overhead = {method: [] for method in methods}
    all_layers = sorted(
        {
            spec.layer
            for method in methods
            for spec in managers[method].operators
        }
    )
    started = time.perf_counter()
    try:
        for prefix_key in sorted(grouped):
            role, history, style, depth, realization = prefix_key
            messages = prefix_messages(role, history, style, depth, realization)
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(
                model, device, prefix_ids, all_layers
            )
            boundary_by_layer = {
                int(layer): value.to(device=device, dtype=torch.float32)
                for layer, value in captured.items()
            }
            first_in_prefix = True
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
                matched = (
                    (history == "obedience" and float(prompt["target_obedience"]) == 1.0)
                    or (history == "verification" and float(prompt["target_obedience"]) == 0.0)
                )
                for method in methods:
                    manager = managers[method]
                    boundaries = {
                        spec: boundary_by_layer[spec.layer] for spec in manager.operators
                    }
                    if first_in_prefix:
                        zero_gates = {spec: 0.0 for spec in manager.operators}
                        manager.install(
                            model,
                            gate=zero_gates,
                            module_resolver=_module,
                            boundary_state=boundaries,
                        )
                        try:
                            zero_logits, zero_extended = _suffix_logits_batch(
                                model, device, base_cache, len(prefix_ids), [suffix_ids], label_ids, 0
                            )
                        finally:
                            manager.remove()
                        del zero_extended
                        error = max(
                            abs(float(zero_logits[0][0]) - float(baseline_logits[0][0])),
                            abs(float(zero_logits[0][1]) - float(baseline_logits[0][1])),
                        )
                        identity_checks[method].append(
                            {"job_key": current_key, "selected_logit_max_error": error, "success": error == 0.0}
                        )
                        if error != 0.0:
                            raise RuntimeError(f"zero-gate identity failed for {method}")
                    if method in CANDIDATE_METHODS:
                        gates, gate_diagnostics = _gate_values(
                            {"config": {"family": method_metadata[method].get("config", {}).get("family", "")}},
                            gate_models[method],
                            boundaries,
                            float(prompt["target_obedience"]),
                        )
                    else:
                        gates = {spec: 1.0 for spec in manager.operators}
                        gate_diagnostics = {}
                    before = _memory_allocated(device)
                    _reset_peak(device)
                    _sync(device)
                    method_started = time.perf_counter()
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
                    _sync(device)
                    duration = time.perf_counter() - method_started
                    elapsed[method].append(duration)
                    peak_overhead[method].append(max(0, _peak_memory(device) - before))
                    del edited_extended
                    edited = _margins(edited_logits[0], prompt)
                    controller = {}
                    for spec in manager.operators:
                        # Legacy fixed/rank-one baselines intentionally expose no
                        # controller diagnostics.  Diagnostic collection must not
                        # turn that optional interface into a scientific forward
                        # prerequisite.
                        diagnostics = getattr(
                            manager.operators[spec], "last_diagnostics", {}
                        )
                        controller[f"{spec.layer}:{spec.component}"] = {
                            name: float(value.reshape(-1)[0].detach().cpu())
                            for name, value in diagnostics.items()
                            if isinstance(value, torch.Tensor)
                            and value.numel() > 0
                            and name in {
                                "history_probability",
                                "task_probability",
                                "positive_route",
                                "negative_route",
                                "trust_scale",
                                "effective_route_active",
                                "all_negative_application_logit",
                                "all_negative_application_active",
                                "matched_state_application_logit",
                                "matched_state_application_active",
                                "consensus_application_active",
                                "v5_gate_active",
                            }
                        }
                    record = {
                        "job_key": current_key,
                        "candidate_id": method,
                        "method": method,
                        "evaluation_split": "same_identity_supplement",
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
                        "delta_task_aligned_margin": edited["task_aligned_margin"] - baseline["task_aligned_margin"],
                        "delta_factual_margin": edited["factual_margin"] - baseline["factual_margin"],
                        "delta_user_choice_margin": edited["user_choice_margin"] - baseline["user_choice_margin"],
                        "task_aligned_binary_kl": _binary_kl(
                            _binary_probability(baseline["task_aligned_margin"]),
                            _binary_probability(edited["task_aligned_margin"]),
                        ),
                        "forward_wall_seconds": duration,
                        "peak_incremental_memory_bytes": peak_overhead[method][-1],
                        "gate_diagnostics": gate_diagnostics,
                        "controller_diagnostics": controller,
                        "intervention_diagnostics": {
                            f"{spec.layer}:{spec.component}": values
                            for spec, values in manager.last_stats.items()
                        },
                    }
                    finite = [
                        *baseline.values(), *edited.values(),
                        record["delta_task_aligned_margin"], record["delta_factual_margin"],
                        record["delta_user_choice_margin"], record["task_aligned_binary_kl"],
                        duration,
                    ]
                    if not all(math.isfinite(float(value)) for value in finite):
                        raise FloatingPointError(f"non-finite output for {method}:{current_key}")
                    outputs[method].write(canonical_json(record) + "\n")
                    completed[method].add(current_key)
                first_in_prefix = False
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    finally:
        for handle in outputs.values():
            handle.close()

    wall_seconds = time.perf_counter() - started
    method_systems = {}
    for method in methods:
        observed_sha = hashlib.sha256(
            ("\n".join(sorted(completed[method])) + "\n").encode("utf-8")
        ).hexdigest()
        if len(completed[method]) != EXPECTED_ROWS or observed_sha != EXPECTED_KEY_SHA256:
            raise RuntimeError(f"row/key audit failed for {method}")
        checks = identity_checks[method]
        identity = {
            "schema_version": 1,
            "method": method,
            "checks": checks,
            "max_error": max(row["selected_logit_max_error"] for row in checks),
            "success": all(row["success"] for row in checks),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        atomic_write_text(
            identity_dir / f"{method}.json",
            json.dumps(identity, indent=2, sort_keys=True) + "\n",
        )
        values = elapsed[method]
        method_systems[method] = {
            "rows": len(values),
            "mean_forward_wall_seconds": statistics.fmean(values),
            "p50_forward_wall_seconds": statistics.median(values),
            "p95_forward_wall_seconds": _percentile(values, 0.95),
            "throughput_rows_per_second_exclusive": len(values) / sum(values),
            "peak_incremental_memory_bytes": max(peak_overhead[method]),
        }
    report = {
        "schema_version": 1,
        "stage": authorization["stage"],
        "method_shard": authorization["method_shard"],
        "methods": list(methods),
        "rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "wall_seconds_including_shared_baselines": wall_seconds,
        "methods_systems": method_systems,
        "all_rows_finite": True,
        "operator_dev_accessed_for_fit": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output_dir / "systems_metrics.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
