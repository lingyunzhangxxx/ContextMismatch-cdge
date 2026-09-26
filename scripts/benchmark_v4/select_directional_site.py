#!/usr/bin/env python3
"""Select a DSGE-V3 site only after all three fold-7 behavior runs close."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def _selection_key(report: dict) -> tuple:
    gap_lower = float(report["matched_minus_mismatched_reduction_bootstrap"]["ci95"][0])
    matched_kl = max(
        float(value["binary_kl_upper"]["upper_one_sided_95"])
        for value in report["matched_cell_collateral"].values()
    )
    intervention = float(report["intervention_norm"]["mean_sum_site_relative_norm"])
    layer = int(str(report["candidate_id"]).rsplit(":", 1)[0].rsplit("-", 1)[-1])
    return (-gap_lower, matched_kl, intervention, layer)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, action="append", required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing directional selection: {args.output}")
    if len(args.analysis) != 3:
        raise ValueError("directional site selection requires exactly three analyses")
    fit = json.loads(args.fit_report.read_text())
    contract = json.loads(args.editor_contract.read_text())
    if fit.get("fit_complete") is not True or fit.get("candidate_count") != 3:
        raise ValueError("directional fit is not complete for three candidates")
    if fit.get("all_three_candidate_checkpoints_materialized") is not True:
        raise ValueError("directional fit did not materialize all checkpoints")
    if fit.get("editor_contract_sha256") != sha256_file(args.editor_contract):
        raise ValueError("fit report contract binding mismatch")
    expected = {
        row["checkpoint_sha256"]: row for row in fit.get("candidate_reports", [])
    }
    if len(expected) != 3:
        raise ValueError("fit report checkpoint set is incomplete or duplicated")
    records = []
    observed: set[str] = set()
    for path in args.analysis:
        report = json.loads(path.read_text())
        checkpoint_sha = str(report.get("checkpoint_sha256", ""))
        if checkpoint_sha not in expected:
            raise ValueError(f"analysis is not bound to a fit candidate: {path}")
        if checkpoint_sha in observed:
            raise ValueError("duplicate directional candidate analysis")
        observed.add(checkpoint_sha)
        if report.get("editor_contract_sha256") != sha256_file(args.editor_contract):
            raise ValueError("analysis contract binding mismatch")
        if report.get("fit_report_sha256") != sha256_file(args.fit_report):
            raise ValueError("analysis fit-report binding mismatch")
        if report.get("method") != "DSGE-V3":
            raise ValueError("analysis method mismatch")
        if report.get("evaluation_stage") != "governance_directional_fit_audit":
            raise ValueError("selector accepts only fold-7 fit-audit analyses")
        if report.get("audit", {}).get("success") is not True:
            raise ValueError("directional candidate analysis audit failed")
        if report.get("final_test_open") is not False:
            raise ValueError("directional fit-audit analysis opened final test")
        if report.get("final_test_open_count") != 0:
            raise ValueError("directional fit-audit final-test count is nonzero")
        if report.get("production_rollout_approved") is not False:
            raise ValueError("directional fit-audit analysis approved production")
        gate = report.get("selection_gate", {})
        expected_fit_eligible = bool(expected[checkpoint_sha]["fit_eligible"])
        if report.get("fit_eligible") is not expected_fit_eligible:
            raise ValueError("analysis fit eligibility differs from fit report")
        if gate.get("checks", {}).get("fit_eligible") is not expected_fit_eligible:
            raise ValueError("analysis gate fit eligibility differs from fit report")
        behavior_pass = gate.get("behavior_would_pass_without_fit_gate") is True
        expected_candidate_eligible = expected_fit_eligible and behavior_pass
        if gate.get("candidate_eligible") is not expected_candidate_eligible:
            raise ValueError("analysis candidate eligibility is internally inconsistent")
        records.append(
            {
                "candidate_id": report["candidate_id"],
                "checkpoint_sha256": checkpoint_sha,
                "analysis_path": str(path),
                "analysis_sha256": sha256_file(path),
                "fit_eligible": expected_fit_eligible,
                "behavior_would_pass_without_fit_gate": behavior_pass,
                "eligible": expected_candidate_eligible,
                "gap_reduction_lower_95": float(
                    report["matched_minus_mismatched_reduction_bootstrap"]["ci95"][0]
                ),
                "normalized_gap_reduction_lower_95": float(
                    report["normalized_gap_reduction_bootstrap"]["ci95"][0]
                ),
                "matched_selected_logit_max_error": float(
                    report["matched_selected_logit_max_error"]
                ),
                "selection_key": list(_selection_key(report)),
            }
        )
    if observed != set(expected):
        raise ValueError("directional analysis set does not cover all fit candidates")
    eligible = [row for row in records if row["eligible"]]
    selected = min(eligible, key=lambda row: tuple(row["selection_key"])) if eligible else None
    result = {
        "schema_version": 1,
        "stage": "governance_directional_fit_audit_selection",
        "method": "DSGE-V3",
        "all_three_candidates_complete": True,
        "candidate_count": 3,
        "eligible_candidate_count": len(eligible),
        "selected_candidate": selected,
        "scientific_falsifier": (
            None
            if selected is not None
            else "No independently fitted single-site DSGE-V3 candidate passed the unchanged fit and direct fold-7 behavior gates. Thresholds were not weakened and sites were not stacked."
        ),
        "candidate_records": sorted(records, key=lambda row: row["candidate_id"]),
        "fit_report_sha256": sha256_file(args.fit_report),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "selection_rule": contract["direct_behavior_site_selection"]["selection_order"],
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
