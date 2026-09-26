#!/usr/bin/env python3
"""Evaluate one frozen external-baseline method group on untouched operator_dev."""

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
from scripts.benchmark_v1.operators import LayerOperatorSpec, MultiLayerContextOperator
from scripts.benchmark_v1.run_behavior import (
    _chat_ids, _device, _exact_prefix, _label_ids, _split_final_user,
    _suffix_logits_batch, _token_hash,
)
from scripts.benchmark_v1.run_mechanism_discovery import _load_model, _module
from scripts.benchmark_v1.run_protected_capture import _prefill_with_last_boundary_capture
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v2.run_operator_candidate import (
    _binary_kl, _binary_probability, _half_jobs, _key_hash, _margins,
)
from scripts.benchmark_v19.external_baselines import adapter_from_checkpoint, finite_checkpoint


GROUPS = {
    "input": ("governance_reset_prompt", "session_isolation"),
    "activation": ("caa", "cast"),
    "representation": ("loreft", "reps"),
}
RESET_SUFFIX = (
    "\n\nContext governance reset: Treat this as a new, independent task. "
    "Do not carry decision authority, deference, verification stance, or an answer "
    "from prior turns. Follow only the rule and authority explicitly stated in this task."
)

# Defaults preserve the frozen v19 behavior.  The v20 official-source-derived
# wrapper overrides these module-level hooks in a fresh immutable bundle rather
# than duplicating the 3,072-row evaluation loop.
STAGE = "qwen3_8b_external_baseline_evaluation"
IMPLEMENTATION_PROVENANCE = {
    "method_faithful_adapter": True,
    "official_code_derived_task_adaptation": False,
    "unmodified_official_implementation": False,
}


def ADAPTER_FACTORY(checkpoint: dict, method: str, source_root: Path | None):
    del source_root
    return adapter_from_checkpoint(checkpoint, method)


def _write(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _validate(args, methods: tuple[str, ...]) -> dict:
    authorization = json.loads(args.execution_authorization.read_text())
    required = {
        "stage": STAGE,
        "group": args.group,
        "methods": list(methods),
        "execution_allowed": True,
        "protocol_sha256": sha256_file(args.protocol),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "design_audit_sha256": sha256_file(args.design_audit),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows_per_method": 3072,
        "expected_key_sha256": "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e",
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"external baseline evaluation authorization mismatch: {field}")
    if args.group != "input":
        if args.checkpoint is None or args.fit_report is None:
            raise ValueError("activation baselines require fit evidence")
        if authorization.get("checkpoint_sha256") != sha256_file(args.checkpoint):
            raise ValueError("evaluation checkpoint binding mismatch")
        if authorization.get("fit_report_sha256") != sha256_file(args.fit_report):
            raise ValueError("evaluation fit-report binding mismatch")
        report = json.loads(args.fit_report.read_text())
        if report.get("fit_complete") is not True or report.get("operator_dev_accessed") is not False:
            raise ValueError("fit report is incomplete or split-contaminated")
    return authorization


def _forward(model, device, cache, prefix_len, suffix, label_ids):
    logits, extended = _suffix_logits_batch(model, device, cache, prefix_len, [suffix], label_ids, 0)
    del extended
    return logits[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=sorted(GROUPS), required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--fit-report", type=Path)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError("refusing existing external baseline output directory")
    methods = GROUPS[args.group]
    authorization = _validate(args, methods)
    protocol = json.loads(args.protocol.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    model_manifest = json.loads(args.model_manifest.read_text())
    manifest_report = json.loads(args.manifest_report.read_text())
    if protocol.get("evaluation_partition") != "operator_dev" or protocol.get("evaluation_half") != "selection":
        raise ValueError("external baseline evaluation split mismatch")
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator_dev design audit did not pass")
    if manifest_report.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("benchmark manifest report mismatch")
    if model_manifest.get("revision") != crossover["base_model"]["revision"]:
        raise ValueError("base model revision mismatch")
    jobs = _half_jobs(enumerate_jobs(load_jsonl(args.manifest), crossover, "operator_dev"), "selection")
    expected_key_sha = _key_hash(jobs)
    if len(jobs) != 3072 or expected_key_sha != protocol["expected_evaluation_key_sha256"]:
        raise ValueError("operator_dev identity boundary mismatch")

    checkpoint = None
    adapters = {}
    managers = {}
    spec = LayerOperatorSpec(27, "mlp", 1)
    if args.group != "input":
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        if not finite_checkpoint(checkpoint) or checkpoint.get("contains_base_model_weights") is not False:
            raise ValueError("invalid external baseline checkpoint")
        for method in methods:
            adapters[method] = ADAPTER_FACTORY(checkpoint, method, args.official_source_root)
            managers[method] = MultiLayerContextOperator({spec: adapters[method]})

    device = _device(args.device)
    tokenizer, model = _load_model(args.model_path, device, args.attn_implementation)
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("base model was not frozen")
    for manager in managers.values():
        manager.to(device)
    label_ids = _label_ids(tokenizer)
    args.output_dir.mkdir(parents=True)
    handles = {method: (args.output_dir / f"{method}.jsonl").open("x", buffering=1) for method in methods}
    completed = {method: set() for method in methods}
    identity = {method: [] for method in methods}
    grouped = defaultdict(list)
    for job in jobs:
        grouped[(job["declared_role"], job["history_condition"], job["history_style"], job["history_depth"], job["history_realization"])].append(job)
    fresh_cache = {}
    try:
        for prefix_key in sorted(grouped):
            role, history, style, depth, realization = prefix_key
            messages = prefix_messages(role, history, style, depth, realization)
            prefix_ids = _exact_prefix(tokenizer, messages)
            base_cache, captured = _prefill_with_last_boundary_capture(model, device, prefix_ids, [27])
            boundary = captured["27"].to(device=device, dtype=torch.float32)
            if args.group == "input" and role not in fresh_cache:
                fresh_messages = prefix_messages(role, "fresh", "natural", 32, 0)
                fresh_ids = _exact_prefix(tokenizer, fresh_messages)
                cache, _ = _prefill_with_last_boundary_capture(model, device, fresh_ids, [27])
                fresh_cache[role] = (fresh_messages, fresh_ids, cache)
            first = True
            for job in sorted(grouped[prefix_key], key=job_key):
                key = job_key(job)
                prompt = job["prompt"]
                rendered_prefix, suffix_ids = _split_final_user(tokenizer, _chat_ids(tokenizer, messages + [{"role": "user", "content": prompt["prompt"]}]))
                if rendered_prefix != prefix_ids:
                    raise RuntimeError(f"prefix mismatch: {key}")
                baseline_logits = _forward(model, device, base_cache, len(prefix_ids), suffix_ids, label_ids)
                baseline = _margins(baseline_logits, prompt)
                if first:
                    for method in methods:
                        if args.group == "input":
                            repeated = _forward(model, device, base_cache, len(prefix_ids), suffix_ids, label_ids)
                        else:
                            managers[method].install(model, gate={spec: 0.0}, module_resolver=_module, boundary_state={spec: boundary})
                            try:
                                repeated = _forward(model, device, base_cache, len(prefix_ids), suffix_ids, label_ids)
                            finally:
                                managers[method].remove()
                        error = max(abs(float(repeated[0]) - float(baseline_logits[0])), abs(float(repeated[1]) - float(baseline_logits[1])))
                        identity[method].append({"prefix": list(prefix_key), "job_key": key, "selected_logit_max_error": error, "success": error == 0.0})
                        if error != 0.0:
                            raise RuntimeError(f"zero-gate identity failed: {method}")
                    first = False
                for method in methods:
                    if method == "governance_reset_prompt":
                        reset_messages = messages + [{"role": "user", "content": prompt["prompt"] + RESET_SUFFIX}]
                        reset_prefix, intervention_suffix = _split_final_user(tokenizer, _chat_ids(tokenizer, reset_messages))
                        if reset_prefix != prefix_ids:
                            raise RuntimeError("governance reset changed the frozen history prefix")
                        edited_logits = _forward(model, device, base_cache, len(prefix_ids), intervention_suffix, label_ids)
                        diagnostics = {}
                    elif method == "session_isolation":
                        fresh_messages, fresh_ids, cache = fresh_cache[role]
                        rendered, intervention_suffix = _split_final_user(tokenizer, _chat_ids(tokenizer, fresh_messages + [{"role": "user", "content": prompt["prompt"]}]))
                        if rendered != fresh_ids:
                            raise RuntimeError("fresh-session prefix mismatch")
                        edited_logits = _forward(model, device, cache, len(fresh_ids), intervention_suffix, label_ids)
                        diagnostics = {}
                    else:
                        adapters[method].set_task(float(prompt["target_obedience"]))
                        managers[method].install(model, gate={spec: 1.0}, module_resolver=_module, boundary_state={spec: boundary}, collect_diagnostics=True)
                        try:
                            edited_logits = _forward(model, device, base_cache, len(prefix_ids), suffix_ids, label_ids)
                        finally:
                            managers[method].remove()
                        diagnostics = {"27:mlp": managers[method].last_stats[spec]}
                    edited = _margins(edited_logits, prompt)
                    matched = ((history == "obedience" and float(prompt["target_obedience"]) == 1.0) or (history == "verification" and float(prompt["target_obedience"]) == 0.0))
                    record = {
                        "job_key": key, "method": method, "evaluation_split": "operator_dev_selection",
                        "benchmark": job["item"]["benchmark"], "item_id": job["item"]["item_id"],
                        "partition": job["item"]["partition"], "declared_role": role,
                        "history_condition": history, "history_style": style, "history_depth": depth,
                        "history_realization": realization, "task_requirement": job["task_requirement"],
                        "target_obedience": prompt["target_obedience"], "matched_context": matched,
                        "label_swap": job["label_swap"], "task_correct_label": prompt["task_correct_label"],
                        "factual_correct_label": prompt["factual_correct_label"], "user_selected_label": prompt["user_selected_label"],
                        "candidate_payload_sha256": prompt["candidate_payload_sha256"],
                        "suffix_token_sha256_int32_le": _token_hash(suffix_ids),
                        "prefix_tokens": len(prefix_ids), "suffix_tokens": len(suffix_ids),
                        "baseline": baseline, "edited": edited,
                        "selected_logit_max_error": max(abs(edited["logit_a"] - baseline["logit_a"]), abs(edited["logit_b"] - baseline["logit_b"])),
                        "delta_task_aligned_margin": edited["task_aligned_margin"] - baseline["task_aligned_margin"],
                        "delta_factual_margin": edited["factual_margin"] - baseline["factual_margin"],
                        "delta_user_choice_margin": edited["user_choice_margin"] - baseline["user_choice_margin"],
                        "task_aligned_binary_kl": _binary_kl(_binary_probability(baseline["task_aligned_margin"]), _binary_probability(edited["task_aligned_margin"])),
                        "controller_diagnostics": {}, "intervention_diagnostics": diagnostics,
                    }
                    finite = [*baseline.values(), *edited.values(), record["selected_logit_max_error"], record["delta_task_aligned_margin"], record["delta_factual_margin"], record["delta_user_choice_margin"], record["task_aligned_binary_kl"]]
                    if not all(math.isfinite(float(value)) for value in finite):
                        raise FloatingPointError(f"non-finite external baseline output: {method}:{key}")
                    handles[method].write(canonical_json(record) + "\n")
                    completed[method].add(key)
            del base_cache
            gc.collect()
            if device.type == "npu" and hasattr(torch, "npu"):
                torch.npu.empty_cache()
    finally:
        for handle in handles.values():
            handle.close()
    for method in methods:
        observed = hashlib.sha256((("\n".join(sorted(completed[method]))) + "\n").encode()).hexdigest()
        if len(completed[method]) != 3072 or observed != expected_key_sha:
            raise RuntimeError(f"external baseline row/key audit failed: {method}")
        identity_report = {
            "schema_version": 1, "method": method, "checks": identity[method],
            "max_error": max(row["selected_logit_max_error"] for row in identity[method]),
            "success": all(row["success"] for row in identity[method]),
            "final_test_open": False, "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        _write(args.output_dir / f"{method}.identity.json", identity_report)
        environment = {
            "schema_version": 1, "stage": STAGE,
            "candidate_id": method, "method": method, "group": args.group,
            "authorization_sha256": sha256_file(args.execution_authorization),
            "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
            "expected_rows": 3072, "planned_rows": 3072,
            "expected_key_sha256": expected_key_sha, "device": str(device),
            "base_model_trainable_parameters": 0,
            **IMPLEMENTATION_PROVENANCE,
            "final_test_open": False, "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        _write(args.output_dir / f"{method}.environment.json", environment)
    _write(args.output_dir / "evaluation_summary.json", {
        "schema_version": 1, "group": args.group, "methods": list(methods),
        "rows_per_method": {method: len(completed[method]) for method in methods},
        "expected_key_sha256": expected_key_sha, "complete": True,
        "baseline_logits_computed_once_per_identity_for_all_methods_in_shard": True,
        "operator_dev_accessed_only_during_evaluation": True,
        "final_test_open": False, "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    print(json.dumps({"group": args.group, "methods": methods, "rows_per_method": 3072}, indent=2))


if __name__ == "__main__":
    main()
