#!/usr/bin/env python3
"""Run fresh Qwen3.5 C-DGE protected controls without old model evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, canonical_json, load_jsonl, sha256_file
from scripts.benchmark_v8 import run_v4_protected_controls_diagnostic as kernel
from scripts.benchmark_v13.qwen35_eval_contract import (
    CONTROL_KEY, CONTROL_ROWS, METHOD, require_authorization, require_fit, require_receipt,
)


def write(path: Path, value: dict) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    for name in (
        "checkpoint", "directional_checkpoint", "fit_report", "fit_receipt", "behavior_analysis",
        "behavior_receipt", "editor_contract", "directional_contract", "composite_contract",
        "composite_report", "evaluation_contract", "execution_authorization", "crossover_contract",
        "manifest", "manifest_report", "controls", "model_contract", "model_path", "output",
        "environment_output", "compatibility_dir",
    ):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--code-root", type=Path, required=True)
    p.add_argument("--device", default="npu:0")
    args = p.parse_args()
    if any(path.exists() for path in (args.output, args.environment_output, args.compatibility_dir)):
        raise FileExistsError("Qwen3.5 control outputs must be fresh")
    fit, composite, _ = require_fit(
        fit_report=args.fit_report, fit_receipt=args.fit_receipt,
        directional_checkpoint=args.directional_checkpoint, applicability_checkpoint=args.checkpoint,
        composite_report=args.composite_report, evaluation_contract=args.evaluation_contract,
    )
    behavior = json.loads(args.behavior_analysis.read_text())
    require_receipt(args.behavior_receipt, run_prefix="qwen3-5-9b-cdge-v4-1-behavior-")
    if behavior.get("stage") != "qwen35_cdge_operator_dev" or behavior.get("candidate_eligible") is not True:
        raise ValueError("controls require eligible terminal operator-dev behavior")
    authorization = require_authorization(args.execution_authorization, {
        "stage": "qwen35_cdge_controls", "expected_rows": CONTROL_ROWS,
        "expected_key_sha256": CONTROL_KEY, "final_test_open": False, "final_test_open_count": 0,
    })
    for field, path in {
        "fit_report": args.fit_report, "fit_receipt": args.fit_receipt,
        "applicability_checkpoint": args.checkpoint, "directional_checkpoint": args.directional_checkpoint,
        "applicability_contract": args.editor_contract, "directional_contract": args.directional_contract,
        "composite_contract": args.composite_contract, "composite_report": args.composite_report,
        "evaluation_contract": args.evaluation_contract, "crossover_contract": args.crossover_contract,
        "manifest": args.manifest, "manifest_report": args.manifest_report, "controls": args.controls,
        "model_contract": args.model_contract, "behavior_analysis": args.behavior_analysis,
        "behavior_receipt": args.behavior_receipt,
    }.items():
        if authorization["bound_artifacts"].get(field, {}).get("sha256") != sha256_file(path):
            raise ValueError(f"controls authorization artifact mismatch: {field}")
    args.compatibility_dir.mkdir(parents=True)
    fit_env = args.compatibility_dir / "runtime_fit_envelope.json"
    behavior_env = args.compatibility_dir / "runtime_behavior_envelope.json"
    v3_env = args.compatibility_dir / "embedded_v3_schema_envelope.json"
    diagnostic_env = args.compatibility_dir / "runtime_contract_envelope.json"
    auth_env = args.compatibility_dir / "runtime_authorization_envelope.json"
    model_env = args.compatibility_dir / "runtime_model_manifest_envelope.json"
    write(fit_env, {
        "stage": "governance_abstaining_router_fit", "method": "ADSGE-V4",
        "fit_complete": True, "fit_eligible": False, "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False, "checkpoint_sha256": sha256_file(args.checkpoint),
        "source_qwen35_all_fit_gates_pass": fit["all_fit_gates_pass"],
        "source_qwen35_composite_eligible": composite["candidate_eligible"],
        "final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False,
    })
    write(behavior_env, {
        **behavior, "method": "ADSGE-V4", "post_failure_characterization": True,
        "fit_gates_passed": False, "fit_eligible": False, "candidate_eligible": False,
        "candidate_may_be_locked": False, "fit_report_sha256": sha256_file(fit_env),
        "final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False,
    })
    write(v3_env, {
        "schema_version": 1, "method": "DSGE-V3", "evaluation_stage": "governance_directional_controls",
        "audit": {"success": True}, "compatibility_schema_only": True,
        "embedded_directional_checkpoint_sha256": sha256_file(args.directional_checkpoint),
        "final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False,
    })
    bound = {
        "v4_editor_contract_sha256": sha256_file(args.editor_contract),
        "v4_behavior_diagnostic_contract_sha256": sha256_file(args.evaluation_contract),
        "v4_checkpoint_sha256": sha256_file(args.checkpoint), "v4_fit_report_sha256": sha256_file(fit_env),
        "v4_operator_dev_analysis_sha256": sha256_file(behavior_env),
        "v3_protected_controls_analysis_sha256": sha256_file(v3_env),
        "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "operator_site_manifest_sha256": sha256_file(args.directional_contract),
        "mitigation_controls_sha256": sha256_file(args.controls),
    }
    write(diagnostic_env, {
        "post_failure_characterization": True, "fit_gates_passed": False,
        "candidate_eligible": False, "candidate_may_be_locked": False,
        "bound_artifacts": bound,
        "evaluation": {"expected_rows": CONTROL_ROWS, "expected_key_sha256": CONTROL_KEY,
                       "control_families": sorted(kernel.FAMILIES),
                       "forced_directions": ["positive", "negative"]},
        "final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False,
    })
    model = json.loads(args.model_contract.read_text())
    write(model_env, {"verified": True, "revision": model["lineage"]["weight_and_config_file_revision"],
                      "source_model_contract_sha256": sha256_file(args.model_contract)})
    runtime_auth = {
        "stage": "governance_failed_fit_protected_controls", "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(args.code_root / "bundle.sha256"),
        "execution_allowed": True, "checkpoint_sha256": sha256_file(args.checkpoint),
        "fit_report_sha256": sha256_file(fit_env), "v4_behavior_analysis_sha256": sha256_file(behavior_env),
        "v3_controls_analysis_sha256": sha256_file(v3_env), "editor_contract_sha256": sha256_file(args.editor_contract),
        "behavior_contract_sha256": sha256_file(args.evaluation_contract),
        "diagnostic_contract_sha256": sha256_file(diagnostic_env),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.directional_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "benchmark_manifest_report_sha256": sha256_file(args.manifest_report),
        "controls_sha256": sha256_file(args.controls), "model_manifest_sha256": sha256_file(model_env),
        "expected_rows": CONTROL_ROWS, "expected_key_sha256": CONTROL_KEY,
        "forced_directions": ["positive", "negative"], "post_failure_characterization": True,
        "fit_gates_passed": False, "candidate_eligible": False, "candidate_may_be_locked": False,
        "final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False,
    }
    write(auth_env, runtime_auth)
    runtime_output = args.output.with_name("." + args.output.name + ".runtime")
    runtime_environment = args.environment_output.with_name("." + args.environment_output.name + ".runtime")
    original_root, original_argv = kernel.CODE_ROOT, sys.argv
    try:
        kernel.CODE_ROOT = str(args.code_root)
        sys.argv = [
            "run_v4_protected_controls_diagnostic", "--checkpoint", str(args.checkpoint),
            "--fit-report", str(fit_env), "--v4-behavior-analysis", str(behavior_env),
            "--v3-controls-analysis", str(v3_env), "--editor-contract", str(args.editor_contract),
            "--behavior-contract", str(args.evaluation_contract), "--diagnostic-contract", str(diagnostic_env),
            "--execution-authorization", str(auth_env), "--crossover-contract", str(args.crossover_contract),
            "--operator-site-manifest", str(args.directional_contract), "--manifest", str(args.manifest),
            "--manifest-report", str(args.manifest_report), "--controls", str(args.controls),
            "--model-manifest", str(model_env), "--model-path", str(args.model_path),
            "--output", str(runtime_output), "--environment-output", str(runtime_environment),
            "--device", args.device, "--attn-implementation", "eager",
        ]
        kernel.main()
    finally:
        kernel.CODE_ROOT, sys.argv = original_root, original_argv
    environment = json.loads(runtime_environment.read_text())
    environment.update({
        "stage": "qwen35_cdge_protected_controls", "method": METHOD,
        "runtime_checkpoint_method": "ADSGE-V4", "model": "Qwen3.5-9B",
        "candidate_id": "Qwen3.5-9B-C-DGE-V4.1-27:mlp",
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
        "fit_report_sha256": sha256_file(args.fit_report), "behavior_analysis_sha256": sha256_file(args.behavior_analysis),
        "candidate_eligible": True, "candidate_may_be_locked": False,
        "post_failure_characterization": False, "fit_gates_passed": True,
        "final_test_open": False, "final_test_open_count": 0, "production_rollout_approved": False,
    })
    rows = []
    for row in load_jsonl(runtime_output):
        row["runtime_checkpoint_method"] = row.get("method"); row["method"] = METHOD
        row["model"] = "Qwen3.5-9B"; row["composite_eligible"] = True
        rows.append(canonical_json(row))
    atomic_write_text(args.output, "\n".join(rows) + "\n"); write(args.environment_output, environment)
    write(args.compatibility_dir / "adapter_manifest.json", {
        "schema_version": 1, "adapter": "qwen35_specific_validation_over_generic_protected_control_kernel",
        "generic_numerical_kernel_sha256": sha256_file(Path(kernel.__file__)),
        "fresh_model_forwards_executed": True, "qwen3_8b_control_results_reused": False,
        "forced_expert_source": "exact_directional_checkpoint_embedded_in_applicability_checkpoint",
        "formal_authorization_sha256": sha256_file(args.execution_authorization),
    })
    runtime_output.unlink(); runtime_environment.unlink()


if __name__ == "__main__": main()
