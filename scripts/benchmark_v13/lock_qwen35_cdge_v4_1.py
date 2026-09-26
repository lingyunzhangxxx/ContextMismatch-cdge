#!/usr/bin/env python3
"""Build the frozen Qwen3.5 C-DGE V4.1 Pareto report and final-test lock."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file
from scripts.benchmark_v13.qwen35_eval_contract import (
    CANDIDATE_ID,
    CONTROL_KEY,
    CONTROL_ROWS,
    LOCK_MANIFEST_ID,
    LOCK_STAGE,
    METHOD,
    OPERATOR_KEY,
    OPERATOR_ROWS,
    PERFORMANCE_CELLS,
    PERFORMANCE_ROWS,
    require_receipt,
)


def _finite(value: object) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _all_true(value: object) -> bool:
    return isinstance(value, dict) and bool(value) and all(item is True for item in value.values())


def _behavior_checks(analysis: dict, contract: dict) -> dict[str, bool]:
    audit = analysis.get("audit", {})
    gates = contract["developmental_gates"]
    return {
        "identity": analysis.get("method") == METHOD
        and analysis.get("stage") == "qwen35_cdge_operator_dev"
        and analysis.get("model") == "Qwen3.5-9B",
        "candidate_eligible": analysis.get("candidate_eligible") is True,
        "declared_gate_checks": _all_true(analysis.get("behavior_gate_checks")),
        "audit_success": audit.get("success") is True,
        "rows_exact": audit.get("row_count") == OPERATOR_ROWS
        and audit.get("unique_job_keys") == OPERATOR_ROWS,
        "key_exact": audit.get("expected_key_sha256") == OPERATOR_KEY
        and audit.get("observed_key_sha256") == OPERATOR_KEY,
        "zero_gate_exact": float(audit.get("zero_gate_max_error", math.inf))
        == float(gates["matched_selected_logit_max_error"]),
        "gap_lower_95": float(
            analysis.get("matched_minus_mismatched_reduction_bootstrap", {}).get(
                "ci95", [-math.inf]
            )[0]
        )
        >= float(gates["gap_reduction_lower_95_minimum"]),
        "normalized_gap_lower_95": float(
            analysis.get("normalized_gap_reduction_bootstrap", {}).get("ci95", [-math.inf])[0]
        )
        >= float(gates["normalized_gap_reduction_lower_95_minimum"]),
        "safety_flags": analysis.get("candidate_may_be_locked") is False
        and analysis.get("final_test_open") is False
        and analysis.get("final_test_open_count") == 0
        and analysis.get("production_rollout_approved") is False,
        "finite": _finite(analysis),
    }


def _controls_checks(analysis: dict, contract: dict) -> dict[str, bool]:
    audit = analysis.get("audit", {})
    families = analysis.get("family_reports", {})
    expected_families = set(contract["protected_controls"]["families"])
    family_exact = set(families) == expected_families and all(
        family.get("application_gated_identity", {}).get("exact") is True
        and set(family.get("forced_on", {})) == {"positive", "negative"}
        and all(
            forced.get("reference_gate_pass") is True
            and forced.get("same_frozen_expert_metric_as_v3") is True
            for forced in family.get("forced_on", {}).values()
        )
        for family in families.values()
    )
    return {
        "identity": analysis.get("method") == METHOD
        and analysis.get("stage") == "qwen35_cdge_protected_controls"
        and analysis.get("model") == "Qwen3.5-9B",
        "audit_success": audit.get("success") is True,
        "rows_exact": audit.get("row_count") == CONTROL_ROWS
        and audit.get("unique_case_keys") == CONTROL_ROWS,
        "key_exact": audit.get("expected_key_sha256") == CONTROL_KEY
        and audit.get("observed_key_sha256") == CONTROL_KEY,
        "six_family_gates": family_exact,
        "application_identity": analysis.get("all_six_application_gated_exact_identity") is True
        and analysis.get("application_gated_identity", {}).get("exact") is True,
        "external_zero_identity": analysis.get("external_zero_gate_identity", {}).get("exact") is True,
        "forced_reference_gates": analysis.get("all_forced_direction_reference_gates_pass") is True,
        "safety_flags": analysis.get("candidate_may_be_locked") is False
        and analysis.get("final_test_open") is False
        and analysis.get("final_test_open_count") == 0
        and analysis.get("production_rollout_approved") is False,
        "finite": _finite(analysis),
    }


def _performance_checks(analysis: dict) -> dict[str, bool]:
    audit = analysis.get("audit", {})
    return {
        "identity": analysis.get("method") == METHOD
        and analysis.get("stage") == "governance_composite_performance"
        and analysis.get("model") == "Qwen3.5-9B",
        "audit_success": audit.get("success") is True,
        "rows_exact": audit.get("measurement_rows") == PERFORMANCE_ROWS
        and audit.get("unique_measurement_keys") == PERFORMANCE_ROWS,
        "cells_exact": audit.get("cells") == PERFORMANCE_CELLS,
        "paired_complete": audit.get("paired_repeats_complete") is True
        and audit.get("all_method_repeat_groups_complete") is True,
        "same_prompt_tokens": audit.get("same_prompt_and_tokens_within_cells") is True,
        "methods_exact": set(analysis.get("method_reports", {}))
        == {"baseline", "DSGE-V3", METHOD},
        "paired_reports_exact": set(analysis.get("paired_reports", {})) == {"DSGE-V3", METHOD},
        "safety_flags": analysis.get("final_test_open") is False
        and analysis.get("final_test_open_count") == 0
        and analysis.get("production_rollout_approved") is False,
        "finite": _finite(analysis),
    }


def _source(path: Path, remote_path: str | None = None) -> dict:
    value = {"path": str(path), "sha256": sha256_file(path)}
    if remote_path is not None:
        root = "/workspace/context-mismatch-qwen3-5-9b/"
        if not remote_path.startswith(root):
            raise ValueError("Qwen3.5 runtime source is outside the owned /WORK root")
        value["remote_path"] = remote_path
    return value


def build_report_and_lock(
    *, behavior: dict, behavior_receipt: dict, controls: dict, controls_receipt: dict,
    performance: dict, performance_receipt: dict, evaluation_contract: dict, sources: dict,
) -> tuple[dict, dict | None]:
    checks = {
        "behavior": _behavior_checks(behavior, evaluation_contract),
        "controls": _controls_checks(controls, evaluation_contract),
        "performance": _performance_checks(performance),
        "receipts": {},
    }
    prefixes = {
        "behavior": "qwen3-5-9b-cdge-v4-1-behavior-",
        "controls": "qwen3-5-9b-cdge-v4-1-protected-controls-",
        "performance": "qwen3-5-9b-cdge-v4-1-performance-",
    }
    for name, receipt in {
        "behavior": behavior_receipt,
        "controls": controls_receipt,
        "performance": performance_receipt,
    }.items():
        try:
            # Use a temporary in-memory equivalent of require_receipt's strict checks.
            run_id = receipt.get("run_id", "")
            job_id = receipt.get("job_id")
            terminal = receipt.get("slurm_terminal_record", "")
            checks["receipts"][name] = (
                isinstance(run_id, str)
                and run_id.startswith(prefixes[name])
                and isinstance(job_id, int)
                and job_id > 0
                and isinstance(receipt.get("archive_sha256"), str)
                and len(receipt["archive_sha256"]) == 64
                and receipt.get("cluster_shared_copy_verified") is True
                and receipt.get("host_data_copy_verified") is True
                and receipt.get("local_copy_verified") is True
                and f"JobId={job_id}" in terminal
                and "JobState=COMPLETED" in terminal
                and "ExitCode=0:0" in terminal
            )
        except (TypeError, ValueError):
            checks["receipts"][name] = False
    failures = [f"{group}:{name}" for group, values in checks.items() for name, passed in values.items() if not passed]
    admissible = not failures
    paired = performance.get("paired_reports", {}).get(METHOD, {})
    locked_metrics = {
        "gap_reduction": behavior.get("matched_minus_mismatched_reduction", {}).get(
            "equal_weight_benchmark_mean"
        ),
        "gap_reduction_lower_95": behavior.get(
            "matched_minus_mismatched_reduction_bootstrap", {}
        ).get("ci95", [None])[0],
        "normalized_gap_reduction_lower_95": behavior.get(
            "normalized_gap_reduction_bootstrap", {}
        ).get("ci95", [None])[0],
        "latency_overhead_percent": paired.get("latency_overhead_percent"),
        "throughput_ratio_to_baseline": paired.get("throughput_ratio_to_baseline"),
        "peak_npu_memory_delta_bytes": paired.get("peak_npu_memory_delta_bytes"),
    }
    report = {
        "schema_version": 1, "stage": LOCK_STAGE, "method": METHOD,
        "model": "Qwen3.5-9B", "candidate_id": CANDIDATE_ID,
        "admissible": admissible, "rejection_reasons": failures,
        "gate_checks": checks, "locked_metrics": locked_metrics, "sources": sources,
        "candidate_may_be_locked": admissible, "final_test_open": False,
        "final_test_open_count": 0, "production_rollout_approved": False,
    }
    if not admissible:
        return report, None
    lock = {
        **report, "manifest_id": LOCK_MANIFEST_ID, "locked": True,
        "selection_partition": "operator_dev only",
    }
    return report, lock


def main() -> None:
    p = argparse.ArgumentParser()
    names = (
        "behavior_analysis", "behavior_receipt", "controls_analysis", "controls_receipt",
        "performance_analysis", "performance_receipt", "evaluation_contract", "composite_contract",
        "applicability_contract", "directional_contract", "applicability_checkpoint",
        "directional_checkpoint", "fit_report", "fit_receipt", "output_report", "output_lock",
    )
    for name in names:
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--applicability-checkpoint-remote", required=True)
    p.add_argument("--directional-checkpoint-remote", required=True)
    p.add_argument("--fit-report-remote", required=True)
    args = p.parse_args()
    if args.output_report.exists() or args.output_lock.exists():
        raise FileExistsError("refusing existing Qwen3.5 Pareto output")
    source_paths = {
        name: getattr(args, name)
        for name in names
        if name not in {"output_report", "output_lock"}
    }
    sources = {name: _source(path) for name, path in source_paths.items()}
    sources["applicability_checkpoint"] = _source(
        args.applicability_checkpoint, args.applicability_checkpoint_remote
    )
    sources["directional_checkpoint"] = _source(
        args.directional_checkpoint, args.directional_checkpoint_remote
    )
    sources["fit_report"] = _source(args.fit_report, args.fit_report_remote)
    values = {name: json.loads(getattr(args, name).read_text()) for name in (
        "behavior_analysis", "behavior_receipt", "controls_analysis", "controls_receipt",
        "performance_analysis", "performance_receipt", "evaluation_contract",
    )}
    report, lock = build_report_and_lock(
        behavior=values["behavior_analysis"], behavior_receipt=values["behavior_receipt"],
        controls=values["controls_analysis"], controls_receipt=values["controls_receipt"],
        performance=values["performance_analysis"], performance_receipt=values["performance_receipt"],
        evaluation_contract=values["evaluation_contract"], sources=sources,
    )
    atomic_write_text(args.output_report, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if lock is None:
        raise SystemExit("Qwen3.5 C-DGE V4.1 is not admissible; final test remains closed")
    lock["pareto_report"] = _source(args.output_report)
    atomic_write_text(args.output_lock, json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print(json.dumps(lock, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
