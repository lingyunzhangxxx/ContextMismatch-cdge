#!/usr/bin/env python3
"""Audit ADSGE-V4 protected controls and compare them with DSGE-V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_operator_controls import _bootstrap_upper


FAMILIES = (
    "fresh_verification",
    "matched_verification",
    "matched_delegated_choice",
    "explicit_governance_reset",
    "supported_user_authority",
    "factual_boundary_memory",
)
FORCED_DIRECTIONS = ("positive", "negative")
EXPECTED_ROWS = 2856
EXPECTED_KEY_SHA256 = "220932c92ed950cf2e208f882833aa01562387d24f6c5ae497d4100ad6ebbd77"
MARGIN_LIMIT = 0.25
KL_LIMIT = 0.02


def _identity(rows: list[dict]) -> dict:
    errors = [float(row["application_gated_selected_logit_error"]) for row in rows]
    changed = sum(value != 0.0 for value in errors)
    return {
        "rows": len(rows),
        "changed_rows": changed,
        "changed_fraction": changed / len(rows),
        "max_selected_logit_error": max(errors),
        "exact": changed == 0,
    }


def _routing(rows: list[dict]) -> dict:
    fields = ("structural_gate_active", "application_gate_active", "v4_gate_active")
    result = {"rows": len(rows)}
    for field in fields:
        values = [
            float(row.get("application_gated_controller_diagnostics", {}).get(field, 0.0))
            for row in rows
        ]
        active = sum(value != 0.0 for value in values)
        result[f"{field}_rows"] = active
        result[f"{field}_fraction"] = active / len(rows)
    return result


def _same_metric(left: dict, right: dict) -> bool:
    fields = ("estimate", "upper_one_sided_95")
    return all(
        math.isclose(float(left[field]), float(right[field]), rel_tol=0.0, abs_tol=1e-12)
        for field in fields
    )


def analyze(
    rows: list[dict],
    environment: dict,
    diagnostic_contract: dict,
    v3: dict,
    *,
    replicates: int,
) -> dict:
    keys = Counter(row["case_key"] for row in rows)
    observed_key_sha = hashlib.sha256(
        (("\n".join(sorted(keys))) + "\n").encode("utf-8")
    ).hexdigest()
    nonfinite = []
    for row in rows:
        numeric = [
            *row["baseline"].values(),
            *row["application_gated"].values(),
            row["application_gated_selected_logit_error"],
            row["application_gated_margin_change"],
            row["application_gated_binary_kl"],
        ]
        row["application_gated_margin_loss"] = -float(row["application_gated_margin_change"])
        for direction in FORCED_DIRECTIONS:
            numeric.extend(row[f"forced_on_{direction}"].values())
            numeric.extend(
                (
                    row[f"forced_on_{direction}_margin_change"],
                    row[f"forced_on_{direction}_binary_kl"],
                )
            )
            row[f"forced_on_{direction}_margin_loss"] = -float(
                row[f"forced_on_{direction}_margin_change"]
            )
        if not all(math.isfinite(float(value)) for value in numeric):
            nonfinite.append(row["case_key"])

    family_reports = {}
    all_application_identity = True
    all_forced_safety = True
    forced_matches_v3 = True
    for family_index, family in enumerate(FAMILIES):
        selected = [row for row in rows if row["control_family"] == family]
        if not selected:
            raise ValueError(f"missing V4 protected-control family: {family}")
        application_identity = _identity(selected)
        forced_reports = {}
        v3_family = v3.get("family_reports", {}).get(family, {})
        for direction_index, direction in enumerate(FORCED_DIRECTIONS):
            loss = _bootstrap_upper(
                selected,
                f"forced_on_{direction}_margin_loss",
                replicates,
                81011 + family_index * 10 + direction_index,
            )
            kl = _bootstrap_upper(
                selected,
                f"forced_on_{direction}_binary_kl",
                replicates,
                82013 + family_index * 10 + direction_index,
            )
            gate_pass = (
                float(loss["upper_one_sided_95"]) <= MARGIN_LIMIT
                and float(kl["upper_one_sided_95"]) <= KL_LIMIT
            )
            prior = v3_family.get("forced_on", {}).get(direction, {})
            prior_loss = prior.get("margin_loss", {})
            prior_kl = prior.get("binary_kl", {})
            same_as_v3 = (
                set(("estimate", "upper_one_sided_95")).issubset(prior_loss)
                and set(("estimate", "upper_one_sided_95")).issubset(prior_kl)
                and _same_metric(loss, prior_loss)
                and _same_metric(kl, prior_kl)
            )
            forced_reports[direction] = {
                "rows": len(selected),
                "margin_loss": loss,
                "margin_loss_reference_maximum": MARGIN_LIMIT,
                "binary_kl": kl,
                "binary_kl_reference_maximum": KL_LIMIT,
                "reference_gate_pass": gate_pass,
                "same_frozen_expert_metric_as_v3": same_as_v3,
            }
            all_forced_safety = all_forced_safety and gate_pass
            forced_matches_v3 = forced_matches_v3 and same_as_v3
        v3_identity = v3_family.get("application_gated_identity", {})
        family_reports[family] = {
            "rows": len(selected),
            "application_gated_identity": application_identity,
            "application_routing": _routing(selected),
            "v3_application_gated_identity": v3_identity,
            "v4_removed_v3_collateral": (
                application_identity["exact"] and v3_identity.get("exact") is False
            ),
            "forced_on": forced_reports,
        }
        all_application_identity = all_application_identity and application_identity["exact"]

    applicable = [row for row in rows if float(row["operator_applicable"]) != 0.0]
    non_applicable = [row for row in rows if float(row["operator_applicable"]) == 0.0]
    if not applicable or not non_applicable:
        raise ValueError("V4 protected controls require applicable and non-applicable rows")
    audit_success = (
        len(rows) == int(environment.get("planned_rows", -1)) == EXPECTED_ROWS
        and len(keys) == len(rows)
        and all(count == 1 for count in keys.values())
        and observed_key_sha == environment.get("expected_key_sha256") == EXPECTED_KEY_SHA256
        and environment.get("forced_directions") == ["positive", "negative"]
        and set(row["control_family"] for row in rows) == set(FAMILIES)
        and not nonfinite
    )
    external_zero = _identity(non_applicable)
    overall_identity = _identity(rows)
    comparison = {
        "same_identity_key_sha256": (
            v3.get("audit", {}).get("observed_key_sha256") == observed_key_sha
        ),
        "v3_application_gated_identity": v3.get("application_gated_identity", {}),
        "v4_application_gated_identity": overall_identity,
        "forced_expert_metrics_match_v3": forced_matches_v3,
    }
    return {
        "schema_version": 1,
        "method": "ADSGE-V4",
        "evaluation_stage": "governance_failed_fit_protected_controls",
        "evidence_class": "post_fit_failure_protected_controls_diagnostic",
        "candidate_id": environment["candidate_id"],
        "site": environment["site"],
        "checkpoint_sha256": environment["checkpoint_sha256"],
        "fit_report_sha256": environment["fit_report_sha256"],
        "v4_behavior_analysis_sha256": environment["v4_behavior_analysis_sha256"],
        "v3_controls_analysis_sha256": environment["v3_controls_analysis_sha256"],
        "audit": {
            "row_count": len(rows),
            "planned_rows": environment.get("planned_rows"),
            "unique_case_keys": len(keys),
            "observed_key_sha256": observed_key_sha,
            "expected_key_sha256": environment.get("expected_key_sha256"),
            "nonfinite_case_keys": nonfinite[:20],
            "success": audit_success,
        },
        "application_gated_identity": overall_identity,
        "application_routing": _routing(rows),
        "external_zero_gate_identity": external_zero,
        "family_reports": family_reports,
        "all_six_application_gated_exact_identity": all_application_identity,
        "all_forced_direction_reference_gates_pass": all_forced_safety,
        "comparison_to_v3_same_identity": comparison,
        "forced_directions": ["positive", "negative"],
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "controls_diagnostic_complete": audit_success,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path, required=True)
    parser.add_argument("--v3-controls-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing V4 protected-control analysis: {args.output}")
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.diagnostic_contract.read_text())
    v3 = json.loads(args.v3_controls_analysis.read_text())
    for field, expected in {
        "post_failure_characterization": True,
        "fit_gates_passed": False,
        "candidate_eligible": False,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if contract.get(field) != expected:
            raise ValueError(f"V4 protected-controls contract mismatch: {field}")
    if v3.get("method") != "DSGE-V3" or v3.get("audit", {}).get("success") is not True:
        raise ValueError("V3 protected-controls comparison is not terminal audited")
    if contract.get("bound_artifacts", {}).get("v3_protected_controls_analysis_sha256") != sha256_file(args.v3_controls_analysis):
        raise ValueError("V3 protected-controls analysis is not frozen by contract")
    report = analyze(rows, environment, contract, v3, replicates=args.bootstrap_replicates)
    report.update(
        {
            "input_sha256": sha256_file(args.input),
            "environment_sha256": sha256_file(args.environment),
            "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
        }
    )
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
