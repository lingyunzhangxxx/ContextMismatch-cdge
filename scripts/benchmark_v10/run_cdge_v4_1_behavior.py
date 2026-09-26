#!/usr/bin/env python3
"""Run fresh C-DGE-V4.1 behavior after the terminal composite audit.

The checkpoint remains the immutable ADSGE-V4 checkpoint.  This adapter changes
neither weights nor runtime semantics; it separates the historical failed fit
decision from the later, terminal composite-eligibility decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import sha256_file
from scripts.benchmark_v3 import run_governance_editor as legacy
from scripts.benchmark_v10.cdge_contract import (
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    METHOD,
    RUNTIME_METHOD,
    STAGE,
    _require,
    promote_behavior_outputs,
    validate_composite_prerequisite,
)


def _validate_inputs(args: argparse.Namespace) -> dict:
    fit = json.loads(args.fit_report.read_text())
    composite = json.loads(args.composite_report.read_text())
    receipt = json.loads(args.composite_receipt.read_text())
    contract = json.loads(args.evaluation_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    validate_composite_prerequisite(composite, receipt)
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
    bound = contract.get("bound_artifacts", {})
    bound_expected = {
        "v4_contract_sha256": sha256_file(args.editor_contract),
        "v4_1_composite_contract_sha256": sha256_file(args.composite_contract),
        "v4_checkpoint_sha256": sha256_file(args.checkpoint),
        "v4_fit_report_sha256": sha256_file(args.fit_report),
        "composite_eligibility_report_sha256": sha256_file(args.composite_report),
        "composite_eligibility_authorization_sha256": sha256_file(args.composite_authorization),
        "composite_eligibility_receipt_sha256": sha256_file(args.composite_receipt),
        "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "operator_dev_design_audit_sha256": sha256_file(args.design_audit),
        "v3_operator_dev_analysis_sha256": sha256_file(args.v3_analysis),
        "model_manifest_sha256": sha256_file(args.model_manifest),
    }
    _require(bound, bound_expected, "post-eligibility contract binding")
    _require(
        contract.get("operator_dev", {}),
        {
            "stage": STAGE,
            "expected_rows": EXPECTED_ROWS,
            "expected_key_sha256": EXPECTED_KEY_SHA256,
            "same_identity_as_v3_required": True,
            "all_rows_unique_and_finite_required": True,
            "external_zero_gate_exact_identity_required": True,
        },
        "operator-dev contract",
    )
    required_authorization = {
        "stage": STAGE,
        "execution_allowed": True,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(args.fit_report),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "composite_report_sha256": sha256_file(args.composite_report),
        "composite_authorization_sha256": sha256_file(args.composite_authorization),
        "composite_receipt_sha256": sha256_file(args.composite_receipt),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "v3_operator_dev_analysis_sha256": sha256_file(args.v3_analysis),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "original_v4_fit_eligible": False,
        "composite_eligible": True,
        "candidate_eligible": True,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    _require(authorization, required_authorization, "C-DGE behavior authorization")
    return authorization


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--composite-report", type=Path, required=True)
    parser.add_argument("--composite-authorization", type=Path, required=True)
    parser.add_argument("--composite-receipt", type=Path, required=True)
    parser.add_argument("--execution-authorization", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-report", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--v3-analysis", type=Path, required=True)
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
        raise FileExistsError("refusing existing formal C-DGE behavior output")
    runtime_output = args.output.with_name(f".{args.output.name}.runtime")
    runtime_environment = args.environment_output.with_name(
        f".{args.environment_output.name}.runtime"
    )
    runtime_identity = args.identity_output.with_name(
        f".{args.identity_output.name}.runtime"
    )
    if any(path.exists() for path in (runtime_output, runtime_environment, runtime_identity)):
        raise FileExistsError("refusing stale runtime C-DGE behavior output")

    original_gate = legacy._fit_gate_status
    original_validate = legacy._validate_authorization
    original_argv = sys.argv

    def composite_gate(fit_report: dict) -> bool:
        if fit_report.get("method") != RUNTIME_METHOD or fit_report.get("fit_eligible") is not False:
            raise ValueError("C-DGE adapter requires the unchanged failed V4 fit report")
        return True

    def composite_authorization(*_positional, **_keyword) -> dict:
        return authorization

    try:
        legacy._fit_gate_status = composite_gate
        legacy._validate_authorization = composite_authorization
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
            "--evaluation-split", "selection",
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
    promote_behavior_outputs(
        runtime_output=runtime_output,
        runtime_environment=runtime_environment,
        runtime_identity=runtime_identity,
        output=args.output,
        environment_output=args.environment_output,
        identity_output=args.identity_output,
        composite_contract=args.composite_contract,
        evaluation_contract=args.evaluation_contract,
        composite_report=args.composite_report,
        composite_authorization=args.composite_authorization,
        composite_receipt=args.composite_receipt,
    )


if __name__ == "__main__":
    main()
