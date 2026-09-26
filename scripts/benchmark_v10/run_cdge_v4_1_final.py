#!/usr/bin/env python3
"""Run the one-time C-DGE-V4.1 final test after an immutable Pareto lock."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import sha256_file
from scripts.benchmark_v3 import run_governance_editor as legacy
from scripts.benchmark_v10.cdge_contract import (
    FINAL_EXPECTED_KEY_SHA256,
    FINAL_EXPECTED_ROWS,
    FINAL_STAGE,
    METHOD,
    RUNTIME_METHOD,
    _require,
    promote_final_outputs,
)


def _validate_inputs(args: argparse.Namespace) -> dict:
    fit = json.loads(args.fit_report.read_text())
    lock = json.loads(args.pareto_lock.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    evaluation = json.loads(args.evaluation_contract.read_text())
    _require(
        fit,
        {
            "method": RUNTIME_METHOD,
            "fit_complete": True,
            "fit_eligible": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "historical V4 fit report",
    )
    if fit.get("checkpoint_sha256") != sha256_file(args.checkpoint):
        raise ValueError("historical V4 checkpoint/report mismatch")
    _require(
        lock,
        {
            "manifest_id": "context-mismatch-qwen3-8b-cdge-v4-1-lock-v1",
            "stage": "governance_composite_pareto_lock",
            "method": METHOD,
            "locked": True,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "candidate_may_be_locked": True,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "C-DGE Pareto lock",
    )
    lock_sources = {
        "runtime_checkpoint": args.checkpoint,
        "fit_report": args.fit_report,
        "composite_contract": args.composite_contract,
        "evaluation_contract": args.evaluation_contract,
        "editor_contract": args.editor_contract,
    }
    for name, path in lock_sources.items():
        if lock.get(name, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"Pareto lock source mismatch: {name}")
    bound = evaluation.get("bound_artifacts", {})
    _require(
        bound,
        {
            "v4_contract_sha256": sha256_file(args.editor_contract),
            "v4_1_composite_contract_sha256": sha256_file(args.composite_contract),
            "v4_checkpoint_sha256": sha256_file(args.checkpoint),
            "v4_fit_report_sha256": sha256_file(args.fit_report),
            "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "final_test_design_audit_sha256": sha256_file(args.design_audit),
            "model_manifest_sha256": sha256_file(args.model_manifest),
        },
        "post-eligibility final binding",
    )
    _require(
        authorization,
        {
            "stage": FINAL_STAGE,
            "method": METHOD,
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "execution_allowed": True,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "fit_report_sha256": sha256_file(args.fit_report),
            "editor_contract_sha256": sha256_file(args.editor_contract),
            "composite_contract_sha256": sha256_file(args.composite_contract),
            "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
            "pareto_lock_sha256": sha256_file(args.pareto_lock),
            "editor_lock_sha256": sha256_file(args.pareto_lock),
            "crossover_contract_sha256": sha256_file(args.crossover_contract),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
            "design_audit_sha256": sha256_file(args.design_audit),
            "model_manifest_sha256": sha256_file(args.model_manifest),
            "expected_rows": FINAL_EXPECTED_ROWS,
            "expected_key_sha256": FINAL_EXPECTED_KEY_SHA256,
            "candidate_id": "C-DGE-V4.1-27:mlp",
            "candidate_eligible": True,
            "candidate_may_be_locked": True,
            "final_test_open": True,
            "final_test_open_count": 1,
            "production_rollout_approved": False,
        },
        "C-DGE final authorization",
    )
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--pareto-lock", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--environment-output", type=Path, required=True)
    parser.add_argument("--identity-output", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    authorization = _validate_inputs(args)
    formal_outputs = (args.output, args.environment_output, args.identity_output)
    if any(path.exists() for path in formal_outputs):
        raise FileExistsError("refusing existing formal C-DGE final output")
    runtime_output = args.output.with_name(f".{args.output.name}.runtime")
    runtime_environment = args.environment_output.with_name(
        f".{args.environment_output.name}.runtime"
    )
    runtime_identity = args.identity_output.with_name(
        f".{args.identity_output.name}.runtime"
    )
    if any(path.exists() for path in (runtime_output, runtime_environment, runtime_identity)):
        raise FileExistsError("refusing stale runtime C-DGE final output")

    original_gate = legacy._fit_gate_status
    original_validate = legacy._validate_authorization
    original_argv = sys.argv

    def composite_gate(fit_report: dict) -> bool:
        if fit_report.get("method") != RUNTIME_METHOD or fit_report.get("fit_eligible") is not False:
            raise ValueError("C-DGE final adapter requires the unchanged failed V4 fit report")
        return True

    def final_authorization(*_positional, **_keyword) -> dict:
        return authorization

    try:
        legacy._fit_gate_status = composite_gate
        legacy._validate_authorization = final_authorization
        sys.argv = [
            "run_governance_editor",
            "--checkpoint", str(args.checkpoint),
            "--fit-report", str(args.fit_report),
            "--editor-contract", str(args.editor_contract),
            "--execution-authorization", str(args.execution_authorization),
            "--crossover-contract", str(args.crossover_contract),
            "--manifest", str(args.manifest),
            "--manifest-report", str(args.manifest_report),
            "--design-audit", str(args.design_audit),
            "--model-manifest", str(args.model_manifest),
            "--model-path", str(args.model_path),
            "--evaluation-split", "final_test",
            "--editor-lock", str(args.pareto_lock),
            "--output", str(runtime_output),
            "--environment-output", str(runtime_environment),
            "--identity-output", str(runtime_identity),
            "--device", args.device,
            "--attn-implementation", args.attn_implementation,
        ]
        legacy.main()
    finally:
        legacy._fit_gate_status = original_gate
        legacy._validate_authorization = original_validate
        sys.argv = original_argv
    promote_final_outputs(
        runtime_output=runtime_output,
        runtime_environment=runtime_environment,
        runtime_identity=runtime_identity,
        output=args.output,
        environment_output=args.environment_output,
        identity_output=args.identity_output,
        pareto_lock=args.pareto_lock,
    )


if __name__ == "__main__":
    main()
