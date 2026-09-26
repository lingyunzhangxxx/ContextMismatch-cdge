#!/usr/bin/env python3
"""Build the frozen C-DGE-V4.1 Pareto report and immutable final-test lock."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v10.cdge_contract import (
    CONTROLS_EXPECTED_KEY_SHA256,
    CONTROLS_EXPECTED_ROWS,
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    METHOD,
    STAGE,
)


CONTROLS_STAGE = "governance_composite_protected_controls"
PERFORMANCE_STAGE = "governance_composite_performance"
PERFORMANCE_ROWS = 720
PERFORMANCE_CELLS = 24
OWNED_REMOTE_ROOTS = (
    "/workspace/context-mismatch-qwen3-8b/",
    "/workspace/context-mismatch-qwen3-8b/",
)


def _finite(value: object) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, (str, int)):
        return True
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return False


def _receipt_gate(receipt: dict, *, expected_run_prefix: str) -> list[str]:
    reasons: list[str] = []
    run_id = receipt.get("run_id")
    job_id = receipt.get("job_id")
    archive_sha = receipt.get("archive_sha256")
    if not isinstance(run_id, str) or not run_id.startswith(expected_run_prefix):
        reasons.append("receipt run identity mismatch")
    if not isinstance(job_id, int) or job_id <= 0:
        reasons.append("receipt job identity mismatch")
    if not (
        isinstance(archive_sha, str)
        and len(archive_sha) == 64
        and all(character in "0123456789abcdef" for character in archive_sha)
    ):
        reasons.append("receipt archive SHA is invalid")
    for field in (
        "cluster_shared_copy_verified",
        "host_data_copy_verified",
        "local_copy_verified",
    ):
        if receipt.get(field) is not True:
            reasons.append(f"receipt lacks verified copy: {field}")
    terminal = receipt.get("slurm_terminal_record", "")
    for fragment in (
        f"JobId={job_id}" if isinstance(job_id, int) else "JobId=",
        "JobState=COMPLETED",
        "ExitCode=0:0",
    ):
        if fragment not in terminal:
            reasons.append(f"receipt lacks terminal fragment: {fragment}")
    return reasons


def _behavior_gates(analysis: dict, contract: dict) -> tuple[dict, list[str]]:
    gate = contract["developmental_operator_dev_gate"]
    audit = analysis.get("audit", {})
    checks: dict[str, bool] = {
        "identity": analysis.get("method") == METHOD and analysis.get("stage") == STAGE,
        "candidate_eligible": analysis.get("candidate_eligible") is True,
        "audit_success": audit.get("success") is True,
        "rows_exact": audit.get("row_count") == EXPECTED_ROWS
        and audit.get("unique_job_keys") == EXPECTED_ROWS,
        "key_sha_exact": audit.get("expected_key_sha256") == EXPECTED_KEY_SHA256
        and audit.get("observed_key_sha256") == EXPECTED_KEY_SHA256,
        "zero_gate_exact": float(audit.get("zero_gate_max_error", math.inf)) == 0.0,
        "gap_lower_95": float(
            analysis.get("matched_minus_mismatched_reduction_bootstrap", {}).get(
                "ci95", [-math.inf]
            )[0]
        )
        >= float(gate["gap_reduction_lower_95_minimum"]),
        "normalized_gap_lower_95": float(
            analysis.get("normalized_gap_reduction_bootstrap", {}).get(
                "ci95", [-math.inf]
            )[0]
        )
        >= float(gate["normalized_gap_reduction_lower_95_minimum"]),
        "both_directions_positive": bool(analysis.get("directional_recovery"))
        and all(
            float(report["equal_weight_benchmark_mean"]) > 0.0
            for report in analysis.get("directional_recovery", {}).values()
        ),
        "both_label_swaps_positive": bool(analysis.get("gap_reduction_by_label_swap"))
        and all(
            float(report["equal_weight_benchmark_mean"]) > 0.0
            for report in analysis.get("gap_reduction_by_label_swap", {}).values()
        ),
        "all_benchmarks_nonnegative": len(analysis.get("gap_reduction_by_benchmark", {})) == 6
        and all(
            float(report["equal_weight_benchmark_mean"]) >= 0.0
            for report in analysis.get("gap_reduction_by_benchmark", {}).values()
        ),
        "matched_collateral": len(analysis.get("matched_cell_collateral", {})) == 2
        and all(
            float(report["margin_loss_upper"]["upper_one_sided_95"])
            <= float(gate["matched_cell_margin_loss_upper_95_maximum"])
            and float(report["binary_kl_upper"]["upper_one_sided_95"])
            <= float(gate["matched_cell_binary_kl_upper_95_maximum"])
            for report in analysis.get("matched_cell_collateral", {}).values()
        ),
        "safety_flags": analysis.get("candidate_may_be_locked") is False
        and analysis.get("final_test_open") is False
        and analysis.get("final_test_open_count") == 0
        and analysis.get("production_rollout_approved") is False,
        "finite": _finite(analysis),
    }
    return checks, [name for name, passed in checks.items() if not passed]


def _controls_gates(analysis: dict) -> tuple[dict, list[str]]:
    audit = analysis.get("audit", {})
    families = analysis.get("family_reports", {})
    family_checks = []
    for family in families.values():
        identity = family.get("application_gated_identity", {})
        forced = family.get("forced_on", {})
        family_checks.append(
            identity.get("exact") is True
            and float(identity.get("max_selected_logit_error", math.inf)) == 0.0
            and set(forced) == {"positive", "negative"}
            and all(
                value.get("reference_gate_pass") is True
                and value.get("same_frozen_expert_metric_as_v3") is True
                for value in forced.values()
            )
        )
    checks = {
        "identity": analysis.get("method") == METHOD
        and analysis.get("evaluation_stage") == CONTROLS_STAGE,
        "candidate_eligible": analysis.get("candidate_eligible") is True,
        "audit_success": audit.get("success") is True,
        "rows_exact": audit.get("row_count") == CONTROLS_EXPECTED_ROWS
        and audit.get("unique_case_keys") == CONTROLS_EXPECTED_ROWS,
        "key_sha_exact": audit.get("expected_key_sha256") == CONTROLS_EXPECTED_KEY_SHA256
        and audit.get("observed_key_sha256") == CONTROLS_EXPECTED_KEY_SHA256,
        "all_six_families": len(families) == 6,
        "application_gated_exact_identity": analysis.get(
            "all_six_application_gated_exact_identity"
        )
        is True,
        "forced_reference_gates": analysis.get("all_forced_direction_reference_gates_pass")
        is True,
        "family_level_gates": len(family_checks) == 6 and all(family_checks),
        "safety_flags": analysis.get("candidate_may_be_locked") is False
        and analysis.get("final_test_open") is False
        and analysis.get("final_test_open_count") == 0
        and analysis.get("production_rollout_approved") is False,
        "finite": _finite(analysis),
    }
    return checks, [name for name, passed in checks.items() if not passed]


def _performance_gates(analysis: dict) -> tuple[dict, list[str]]:
    audit = analysis.get("audit", {})
    checks = {
        "identity": analysis.get("method") == METHOD and analysis.get("stage") == PERFORMANCE_STAGE,
        "audit_success": audit.get("success") is True,
        "rows_exact": audit.get("measurement_rows") == PERFORMANCE_ROWS
        and audit.get("unique_measurement_keys") == PERFORMANCE_ROWS,
        "cells_exact": audit.get("cells") == PERFORMANCE_CELLS,
        "same_prompt_tokens": audit.get("same_prompt_and_tokens_within_cells") is True,
        "paired_complete": audit.get("paired_repeats_complete") is True
        and audit.get("all_method_repeat_groups_complete") is True,
        "methods_exact": set(analysis.get("method_reports", {}))
        == {"baseline", "DSGE-V3", METHOD},
        "paired_reports_exact": set(analysis.get("paired_reports", {}))
        == {"DSGE-V3", METHOD},
        "safety_flags": analysis.get("final_test_open") is False
        and analysis.get("final_test_open_count") == 0
        and analysis.get("production_rollout_approved") is False,
        "finite": _finite(analysis),
    }
    return checks, [name for name, passed in checks.items() if not passed]


def build_report_and_lock(
    *,
    behavior: dict,
    behavior_receipt: dict,
    controls: dict,
    controls_receipt: dict,
    performance: dict,
    performance_receipt: dict,
    v3_contract: dict,
    sources: dict,
) -> tuple[dict, dict | None]:
    behavior_checks, behavior_reasons = _behavior_gates(behavior, v3_contract)
    controls_checks, controls_reasons = _controls_gates(controls)
    performance_checks, performance_reasons = _performance_gates(performance)
    receipt_checks = {
        "behavior": not _receipt_gate(
            behavior_receipt, expected_run_prefix="qwen3-8b-cdge-v4-1-behavior-"
        ),
        "controls": not _receipt_gate(
            controls_receipt,
            expected_run_prefix="qwen3-8b-cdge-v4-1-protected-controls-",
        ),
        "performance": not _receipt_gate(
            performance_receipt,
            expected_run_prefix="qwen3-8b-cdge-v4-1-performance-",
        ),
    }
    receipt_reasons = [name for name, passed in receipt_checks.items() if not passed]
    rejection_reasons = [
        *(f"behavior:{name}" for name in behavior_reasons),
        *(f"controls:{name}" for name in controls_reasons),
        *(f"performance:{name}" for name in performance_reasons),
        *(f"receipt:{name}" for name in receipt_reasons),
    ]
    admissible = not rejection_reasons
    report = {
        "schema_version": 1,
        "stage": "governance_composite_pareto_lock",
        "method": METHOD,
        "candidate_id": "C-DGE-V4.1-27:mlp",
        "candidate_count": 1,
        "admissible_count": int(admissible),
        "pareto_front_ids": ["C-DGE-V4.1-27:mlp"] if admissible else [],
        "chosen_candidate_id": "C-DGE-V4.1-27:mlp" if admissible else None,
        "admissible": admissible,
        "rejection_reasons": rejection_reasons,
        "gate_checks": {
            "behavior": behavior_checks,
            "controls": controls_checks,
            "performance": performance_checks,
            "receipts": receipt_checks,
        },
        "locked_metrics": {
            "gap_reduction": behavior["matched_minus_mismatched_reduction"][
                "equal_weight_benchmark_mean"
            ],
            "gap_reduction_lower_95": behavior[
                "matched_minus_mismatched_reduction_bootstrap"
            ]["ci95"][0],
            "normalized_gap_reduction_lower_95": behavior[
                "normalized_gap_reduction_bootstrap"
            ]["ci95"][0],
            "latency_overhead_percent": performance["paired_reports"][METHOD][
                "latency_overhead_percent"
            ],
            "throughput_ratio_to_baseline": performance["paired_reports"][METHOD][
                "throughput_ratio_to_baseline"
            ],
            "peak_npu_memory_delta_bytes": performance["paired_reports"][METHOD][
                "peak_npu_memory_delta_bytes"
            ],
        },
        "sources": sources,
        "candidate_may_be_locked": admissible,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if not admissible:
        return report, None
    lock = {
        "schema_version": 1,
        "manifest_id": "context-mismatch-qwen3-8b-cdge-v4-1-lock-v1",
        "stage": "governance_composite_pareto_lock",
        "method": METHOD,
        "locked": True,
        "selection_partition": "operator_dev only",
        "candidate_id": "C-DGE-V4.1-27:mlp",
        "runtime_checkpoint_method": "ADSGE-V4",
        "runtime_checkpoint": sources["runtime_checkpoint"],
        "fit_report": sources["fit_report"],
        "behavior_analysis": sources["behavior_analysis"],
        "behavior_receipt": sources["behavior_receipt"],
        "controls_analysis": sources["controls_analysis"],
        "controls_receipt": sources["controls_receipt"],
        "performance_analysis": sources["performance_analysis"],
        "performance_receipt": sources["performance_receipt"],
        "composite_contract": sources["composite_contract"],
        "evaluation_contract": sources["evaluation_contract"],
        "editor_contract": sources["editor_contract"],
        "locked_metrics": report["locked_metrics"],
        "candidate_may_be_locked": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    return report, lock


def _source(path: Path, remote: str | None = None) -> dict:
    value = {"path": str(path), "sha256": sha256_file(path)}
    if remote is not None:
        if not remote.startswith(OWNED_REMOTE_ROOTS):
            raise ValueError("runtime source is outside owned cluster roots")
        value["remote_path"] = remote
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--behavior-analysis", type=Path, required=True)
    parser.add_argument("--behavior-receipt", type=Path, required=True)
    parser.add_argument("--controls-analysis", type=Path, required=True)
    parser.add_argument("--controls-receipt", type=Path, required=True)
    parser.add_argument("--performance-analysis", type=Path, required=True)
    parser.add_argument("--performance-receipt", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--composite-contract", type=Path, required=True)
    parser.add_argument("--evaluation-contract", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--runtime-checkpoint", type=Path, required=True)
    parser.add_argument("--runtime-checkpoint-remote", required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--fit-report-remote", required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--output-lock", type=Path, required=True)
    args = parser.parse_args()
    if args.output_report.exists() or args.output_lock.exists():
        raise FileExistsError("refusing existing C-DGE Pareto/lock output")
    paths = {
        "behavior_analysis": args.behavior_analysis,
        "behavior_receipt": args.behavior_receipt,
        "controls_analysis": args.controls_analysis,
        "controls_receipt": args.controls_receipt,
        "performance_analysis": args.performance_analysis,
        "performance_receipt": args.performance_receipt,
        "composite_contract": args.composite_contract,
        "evaluation_contract": args.evaluation_contract,
        "editor_contract": args.editor_contract,
    }
    sources = {name: _source(path) for name, path in paths.items()}
    sources["runtime_checkpoint"] = _source(
        args.runtime_checkpoint, args.runtime_checkpoint_remote
    )
    sources["fit_report"] = _source(args.fit_report, args.fit_report_remote)
    report, lock = build_report_and_lock(
        behavior=json.loads(args.behavior_analysis.read_text()),
        behavior_receipt=json.loads(args.behavior_receipt.read_text()),
        controls=json.loads(args.controls_analysis.read_text()),
        controls_receipt=json.loads(args.controls_receipt.read_text()),
        performance=json.loads(args.performance_analysis.read_text()),
        performance_receipt=json.loads(args.performance_receipt.read_text()),
        v3_contract=json.loads(args.v3_contract.read_text()),
        sources=sources,
    )
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if lock is None:
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit("C-DGE V4.1 is not admissible; lock was not created")
    lock["pareto_report"] = _source(args.output_report)
    atomic_write_text(args.output_lock, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
