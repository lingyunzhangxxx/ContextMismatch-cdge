#!/usr/bin/env python3
"""Run Qwen3.5 C-DGE on frozen operator-dev or one-time final identities."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v3 import run_governance_editor as kernel
from scripts.benchmark_v13.qwen35_eval_contract import (
    FINAL_KEY, FINAL_ROWS, METHOD, OPERATOR_KEY, OPERATOR_ROWS,
    require_authorization, require_fit, require_pareto_lock,
)


def _write(path: Path, value: dict) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--evaluation-split", choices=("operator_dev", "final_test"), required=True)
    for name in (
        "checkpoint", "directional_checkpoint", "fit_report", "fit_receipt", "editor_contract",
        "composite_contract", "composite_report", "evaluation_contract", "execution_authorization",
        "crossover_contract", "manifest", "manifest_report", "design_audit", "model_contract",
        "model_path", "output", "environment_output", "identity_output", "compatibility_dir",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--editor-lock", type=Path)
    p.add_argument("--device", default="npu:0")
    args = p.parse_args()
    outputs = (args.output, args.environment_output, args.identity_output, args.compatibility_dir)
    if any(path is None or path.exists() for path in outputs):
        raise FileExistsError("Qwen3.5 behavior outputs must all be fresh paths")
    fit, composite, _ = require_fit(
        fit_report=args.fit_report, fit_receipt=args.fit_receipt,
        directional_checkpoint=args.directional_checkpoint,
        applicability_checkpoint=args.checkpoint,
        composite_report=args.composite_report,
        evaluation_contract=args.evaluation_contract,
    )
    final = args.evaluation_split == "final_test"
    expected_rows, expected_key = (FINAL_ROWS, FINAL_KEY) if final else (OPERATOR_ROWS, OPERATOR_KEY)
    authorization = require_authorization(
        args.execution_authorization,
        {
            "stage": "qwen35_cdge_final" if final else "qwen35_cdge_behavior",
            "expected_rows": expected_rows,
            "expected_key_sha256": expected_key,
            "final_test_open": final,
            "final_test_open_count": 1 if final else 0,
        },
    )
    bound = authorization["bound_artifacts"]
    for field, path in {
        "fit_report": args.fit_report, "fit_receipt": args.fit_receipt,
        "applicability_checkpoint": args.checkpoint,
        "directional_checkpoint": args.directional_checkpoint,
        "applicability_contract": args.editor_contract,
        "composite_contract": args.composite_contract,
        "composite_report": args.composite_report,
        "evaluation_contract": args.evaluation_contract,
        "crossover_contract": args.crossover_contract, "manifest": args.manifest,
        "manifest_report": args.manifest_report, "model_contract": args.model_contract,
        "final_design_audit" if final else "operator_dev_design_audit": args.design_audit,
    }.items():
        if bound.get(field, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"authorization artifact mismatch: {field}")
    if final:
        if args.editor_lock is None or bound.get("pareto_lock", {}).get("sha256") != sha256_file(args.editor_lock):
            raise ValueError("one-time final authorization/lock mismatch")
        require_pareto_lock(args.editor_lock)
    elif args.editor_lock is not None:
        raise ValueError("operator-dev behavior must not receive a final lock")
    args.compatibility_dir.mkdir(parents=True)
    fit_envelope = args.compatibility_dir / "runtime_fit_envelope.json"
    auth_envelope = args.compatibility_dir / "runtime_authorization_envelope.json"
    model_envelope = args.compatibility_dir / "runtime_model_manifest_envelope.json"
    _write(fit_envelope, {
        "schema_version": 1, "method": "ADSGE-V4", "fit_complete": True,
        "all_fit_gates_pass": True, "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_qwen35_fit_report_sha256": sha256_file(args.fit_report),
        "source_all_fit_gates_pass": fit["all_fit_gates_pass"],
        "source_composite_candidate_eligible": composite["candidate_eligible"],
        "final_test_open": False, "final_test_open_count": 0,
        "production_rollout_approved": False,
    })
    legacy_split = "final_test" if final else "selection"
    legacy_stage = "governance_behavior_final_test" if final else "governance_behavior_selection"
    runtime_auth = {
        "stage": legacy_stage, "execution_allowed": True,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(fit_envelope),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "expected_rows": expected_rows, "expected_key_sha256": expected_key,
        "final_test_open": final, "final_test_open_count": 1 if final else 0,
        "production_rollout_approved": False,
    }
    if final: runtime_auth["editor_lock_sha256"] = sha256_file(args.editor_lock)
    _write(auth_envelope, runtime_auth)
    model = json.loads(args.model_contract.read_text())
    _write(model_envelope, {
        "schema_version": 1, "verified": True,
        "revision": model["lineage"]["weight_and_config_file_revision"],
        "source_model_contract_sha256": sha256_file(args.model_contract),
    })
    runtime_output = args.output.with_name("." + args.output.name + ".runtime")
    runtime_environment = args.environment_output.with_name("." + args.environment_output.name + ".runtime")
    runtime_identity = args.identity_output.with_name("." + args.identity_output.name + ".runtime")
    original = sys.argv
    try:
        sys.argv = [
            "run_governance_editor", "--checkpoint", str(args.checkpoint),
            "--fit-report", str(fit_envelope), "--editor-contract", str(args.editor_contract),
            "--execution-authorization", str(auth_envelope), "--crossover-contract", str(args.crossover_contract),
            "--manifest", str(args.manifest), "--manifest-report", str(args.manifest_report),
            "--design-audit", str(args.design_audit), "--model-manifest", str(model_envelope),
            "--model-path", str(args.model_path), "--evaluation-split", legacy_split,
            "--output", str(runtime_output), "--environment-output", str(runtime_environment),
            "--identity-output", str(runtime_identity), "--device", args.device,
            "--attn-implementation", "eager",
        ]
        if final: sys.argv.extend(("--editor-lock", str(args.editor_lock)))
        kernel.main()
    finally:
        sys.argv = original
    environment = json.loads(runtime_environment.read_text())
    identity = json.loads(runtime_identity.read_text())
    environment.update({
        "stage": "qwen35_cdge_final_test" if final else "qwen35_cdge_operator_dev",
        "method": METHOD, "runtime_checkpoint_method": "ADSGE-V4",
        "candidate_id": "Qwen3.5-9B-C-DGE-V4.1-27:mlp",
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
        "fit_report_sha256": sha256_file(args.fit_report),
        "fit_receipt_sha256": sha256_file(args.fit_receipt),
        "composite_report_sha256": sha256_file(args.composite_report),
        "candidate_eligible": True, "candidate_may_be_locked": final,
        "final_test_open": final, "final_test_open_count": 1 if final else 0,
        "production_rollout_approved": False,
    })
    identity.update({
        "method": METHOD, "runtime_checkpoint_method": "ADSGE-V4",
        "stage": environment["stage"], "final_test_open": final,
        "final_test_open_count": 1 if final else 0,
        "production_rollout_approved": False,
    })
    promoted = []
    for row in load_jsonl(runtime_output):
        row["runtime_checkpoint_method"] = row.get("method")
        row["method"] = METHOD
        row["model"] = "Qwen3.5-9B"
        row["composite_eligible"] = True
        promoted.append(canonical_json(row))
    atomic_write_text(args.output, "\n".join(promoted) + "\n")
    _write(args.environment_output, environment)
    _write(args.identity_output, identity)
    _write(args.compatibility_dir / "adapter_manifest.json", {
        "schema_version": 1,
        "adapter": "qwen35_specific_validation_over_generic_behavior_kernel",
        "generic_numerical_kernel_sha256": sha256_file(Path(kernel.__file__)),
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
        "scientific_runtime_changed": False,
        "fresh_model_forwards_executed": True,
        "qwen3_8b_behavior_results_reused": False,
    })
    for path in (runtime_output, runtime_environment, runtime_identity): path.unlink()


if __name__ == "__main__":
    main()
