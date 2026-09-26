#!/usr/bin/env python3
"""Run fresh formal C-DGE-V4.1 protected controls.

The numerical kernel is the frozen V4 protected-control runner.  A fully
archived compatibility envelope adapts its old diagnostic-only control plane;
the formal authorization and final outputs remain C-DGE post-eligibility data.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v10.cdge_contract import (
    METHOD,
    RUNTIME_METHOD,
    _require,
    formalize_control_row,
)


STAGE = "governance_composite_protected_controls"
EXPECTED_ROWS = 2856
EXPECTED_KEY_SHA256 = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"
FAMILIES = {
    "fresh_verification",
    "matched_verification",
    "matched_delegated_choice",
    "explicit_governance_reset",
    "supported_user_authority",
    "factual_boundary_memory",
}


def _validate(args: argparse.Namespace) -> dict:
    fit = json.loads(args.fit_report.read_text())
    behavior = json.loads(args.candidate_behavior_analysis.read_text())
    v3 = json.loads(args.v3_controls_analysis.read_text())
    contract = json.loads(args.evaluation_contract.read_text())
    authorization = json.loads(args.execution_authorization.read_text())
    _require(
        fit,
        {
            "method": RUNTIME_METHOD,
            "fit_complete": True,
            "fit_eligible": False,
            "checkpoint_sha256": sha256_file(args.checkpoint),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "historical V4 fit report",
    )
    _require(
        behavior,
        {
            "method": METHOD,
            "stage": "governance_composite_behavior_selection",
            "behavior_evaluation_complete": True,
            "original_v4_fit_eligible": False,
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        },
        "fresh C-DGE behavior analysis",
    )
    if behavior.get("audit", {}).get("success") is not True:
        raise ValueError("fresh behavior analysis is not audited")
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 controls comparison is not audited")
    controls = contract.get("protected_controls", {})
    _require(
        controls,
        {
            "stage": STAGE,
            "expected_rows": EXPECTED_ROWS,
            "expected_key_sha256": EXPECTED_KEY_SHA256,
            "forced_directions": ["positive", "negative"],
            "application_gated_identity_required": True,
            "forced_direction_reference_gates_required": True,
            "all_rows_unique_and_finite_required": True,
        },
        "protected-controls contract",
    )
    if set(controls.get("control_families", [])) != FAMILIES:
        raise ValueError("protected-controls family contract mismatch")
    required = {
        "stage": STAGE,
        "method": METHOD,
        "execution_allowed": True,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(args.fit_report),
        "candidate_behavior_analysis_sha256": sha256_file(args.candidate_behavior_analysis),
        "v3_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "composite_contract_sha256": sha256_file(args.composite_contract),
        "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "controls_sha256": sha256_file(args.controls),
        "model_manifest_sha256": sha256_file(args.model_manifest),
        "expected_rows": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "forced_directions": ["positive", "negative"],
        "original_v4_fit_eligible": False,
        "composite_eligible": True,
        "candidate_eligible": True,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    _require(authorization, required, "C-DGE controls authorization")
    return authorization


def _compatibility_files(args: argparse.Namespace, authorization: dict) -> tuple[Path, Path, Path]:
    args.compatibility_dir.mkdir(parents=True, exist_ok=False)
    behavior = json.loads(args.candidate_behavior_analysis.read_text())
    behavior.update(
        {
            "method": RUNTIME_METHOD,
            "post_failure_characterization": True,
            "fit_gates_passed": False,
            "fit_eligible": False,
            "candidate_eligible": False,
            "candidate_may_be_locked": False,
        }
    )
    compatibility_behavior = args.compatibility_dir / "legacy_behavior_envelope.json"
    atomic_write_text(compatibility_behavior, json.dumps(behavior, indent=2, sort_keys=True) + "\n")
    contract = {
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "bound_artifacts": {
            "v4_editor_contract_sha256": sha256_file(args.editor_contract),
            "v4_behavior_diagnostic_contract_sha256": sha256_file(args.evaluation_contract),
            "v4_checkpoint_sha256": sha256_file(args.checkpoint),
            "v4_fit_report_sha256": sha256_file(args.fit_report),
            "v4_operator_dev_analysis_sha256": sha256_file(compatibility_behavior),
            "v3_protected_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
            "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
            "benchmark_manifest_sha256": sha256_file(args.manifest),
            "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
            "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
            "mitigation_controls_sha256": sha256_file(args.controls),
        },
        "evaluation": {
            "expected_rows": EXPECTED_ROWS,
            "expected_key_sha256": EXPECTED_KEY_SHA256,
            "control_families": sorted(FAMILIES),
            "forced_directions": ["positive", "negative"],
        },
    }
    compatibility_contract = args.compatibility_dir / "legacy_diagnostic_envelope.json"
    atomic_write_text(compatibility_contract, json.dumps(contract, indent=2, sort_keys=True) + "\n")
    compatibility_authorization = {
        "stage": "governance_failed_fit_protected_controls",
        "code_root": authorization["code_root"],
        "immutable_code_bundle_manifest_sha256": authorization["immutable_code_bundle_manifest_sha256"],
        "execution_allowed": True,
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(args.fit_report),
        "v4_behavior_analysis_sha256": sha256_file(compatibility_behavior),
        "v3_controls_analysis_sha256": sha256_file(args.v3_controls_analysis),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "behavior_contract_sha256": sha256_file(args.evaluation_contract),
        "diagnostic_contract_sha256": sha256_file(compatibility_contract),
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
    compatibility_auth = args.compatibility_dir / "legacy_authorization_envelope.json"
    atomic_write_text(compatibility_auth, json.dumps(compatibility_authorization, indent=2, sort_keys=True) + "\n")
    return compatibility_behavior, compatibility_contract, compatibility_auth


def _promote(
    args: argparse.Namespace,
    files: tuple[Path, Path, Path],
    *,
    runtime_output: Path,
    runtime_environment: Path,
) -> None:
    environment = json.loads(runtime_environment.read_text())
    environment.update(
        {
            "stage": STAGE,
            "method": METHOD,
            "runtime_checkpoint_method": RUNTIME_METHOD,
            "candidate_behavior_analysis_sha256": sha256_file(args.candidate_behavior_analysis),
            "original_v4_fit_eligible": False,
            "post_failure_characterization": False,
            "composite_eligible": True,
            "candidate_eligible": True,
            "candidate_may_be_locked": False,
            "evaluation_contract_sha256": sha256_file(args.evaluation_contract),
            "authorization_sha256": sha256_file(args.execution_authorization),
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    atomic_write_text(args.environment_output, json.dumps(environment, indent=2, sort_keys=True) + "\n")
    promoted = []
    for row in load_jsonl(runtime_output):
        promoted.append(canonical_json(formalize_control_row(row)))
    atomic_write_text(args.output, "\n".join(promoted) + "\n")
    manifest = {
        "schema_version": 1,
        "adapter": "legacy_v4_protected_control_kernel_under_fresh_cdge_authorization",
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
        "formal_behavior_analysis_sha256": sha256_file(args.candidate_behavior_analysis),
        "compatibility_files": {path.name: sha256_file(path) for path in files},
        "scientific_runtime_changed": False,
        "old_diagnostic_results_reused": False,
        "fresh_model_forwards_executed": True,
    }
    atomic_write_text(
        args.compatibility_dir / "adapter_manifest.json",
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--candidate-behavior-analysis", type=Path, required=True)
    parser.add_argument("--v3-controls-analysis", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
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
    parser.add_argument("--compatibility-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--attn-implementation", default="eager", choices=["eager", "sdpa"])
    args = parser.parse_args()
    from scripts.benchmark_v8 import run_v4_protected_controls_diagnostic as legacy

    authorization = _validate(args)
    files = _compatibility_files(args, authorization)
    runtime_output = args.output.with_name(f".{args.output.name}.runtime")
    runtime_environment = args.environment_output.with_name(
        f".{args.environment_output.name}.runtime"
    )
    for path in (runtime_output, runtime_environment):
        if path.exists():
            raise FileExistsError(f"refusing existing controls runtime artifact: {path}")
    original_root = legacy.CODE_ROOT
    original_argv = sys.argv
    try:
        legacy.CODE_ROOT = str(args.code_root)
        sys.argv = [
            "run_v4_protected_controls_diagnostic",
            "--checkpoint", str(args.checkpoint),
            "--fit-report", str(args.fit_report),
            "--v4-behavior-analysis", str(files[0]),
            "--v3-controls-analysis", str(args.v3_controls_analysis),
            "--editor-contract", str(args.editor_contract),
            "--behavior-contract", str(args.evaluation_contract),
            "--diagnostic-contract", str(files[1]),
            "--execution-authorization", str(files[2]),
            "--crossover-contract", str(args.crossover_contract),
            "--operator-site-manifest", str(args.operator_site_manifest),
            "--manifest", str(args.manifest),
            "--manifest-report", str(args.manifest_report),
            "--controls", str(args.controls),
            "--model-manifest", str(args.model_manifest),
            "--model-path", str(args.model_path),
            "--output", str(runtime_output),
            "--environment-output", str(runtime_environment),
            "--device", args.device,
            "--attn-implementation", args.attn_implementation,
        ]
        legacy.main()
    finally:
        legacy.CODE_ROOT = original_root
        sys.argv = original_argv
    _promote(
        args,
        files,
        runtime_output=runtime_output,
        runtime_environment=runtime_environment,
    )


if __name__ == "__main__":
    main()
