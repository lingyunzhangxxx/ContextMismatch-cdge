#!/usr/bin/env python3
"""Audit DSGE-V3 application-gated and bidirectionally forced controls."""

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


def _exact_identity_report(rows: list[dict]) -> dict:
    maximum = max(float(row["application_gated_selected_logit_error"]) for row in rows)
    return {"rows": len(rows), "max_selected_logit_error": maximum, "exact": maximum == 0.0}


def analyze(
    rows: list[dict],
    environment: dict,
    contract: dict,
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
        row["application_gated_margin_loss"] = -float(
            row["application_gated_margin_change"]
        )
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

    protected = contract["protected_controls"]
    margin_limit = float(protected["forced_on_margin_loss_upper_95_maximum"])
    kl_limit = float(protected["forced_on_binary_kl_upper_95_maximum"])
    family_reports = {}
    every_family_passes = True
    for family_index, family in enumerate(FAMILIES):
        selected = [row for row in rows if row["control_family"] == family]
        if not selected:
            raise ValueError(f"missing directional control family: {family}")
        application_identity = _exact_identity_report(selected)
        forced_reports = {}
        family_pass = application_identity["exact"]
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
                float(loss["upper_one_sided_95"]) <= margin_limit
                and float(kl["upper_one_sided_95"]) <= kl_limit
            )
            forced_reports[direction] = {
                "rows": len(selected),
                "margin_loss": loss,
                "margin_loss_reference_maximum": margin_limit,
                "binary_kl": kl,
                "binary_kl_reference_maximum": kl_limit,
                "reference_gate_pass": gate_pass,
            }
            family_pass = family_pass and gate_pass
        family_reports[family] = {
            "rows": len(selected),
            "application_gated_identity": application_identity,
            "forced_on": forced_reports,
            "family_pass": family_pass,
        }
        every_family_passes = every_family_passes and family_pass

    applicable = [row for row in rows if float(row["operator_applicable"]) != 0.0]
    non_applicable = [row for row in rows if float(row["operator_applicable"]) == 0.0]
    if not applicable or not non_applicable:
        raise ValueError("directional controls require applicable and non-applicable rows")
    application_identity = _exact_identity_report(rows)
    structural_identity = _exact_identity_report(applicable)
    external_zero_identity = _exact_identity_report(non_applicable)
    audit_success = (
        len(rows) == int(environment.get("planned_rows", -1)) == 2856
        and len(keys) == len(rows)
        and all(count == 1 for count in keys.values())
        and observed_key_sha == environment.get("expected_key_sha256")
        and environment.get("forced_directions") == ["positive", "negative"]
        and not nonfinite
    )
    prior_eligible = environment.get("prior_candidate_eligible") is True
    controls_admissible = (
        audit_success
        and prior_eligible
        and application_identity["exact"]
        and structural_identity["exact"]
        and external_zero_identity["exact"]
        and every_family_passes
    )
    return {
        "schema_version": 1,
        "method": "DSGE-V3",
        "evaluation_stage": "governance_directional_controls",
        "candidate_id": environment["candidate_id"],
        "site": environment["site"],
        "checkpoint_sha256": environment["checkpoint_sha256"],
        "fit_report_sha256": environment["fit_report_sha256"],
        "fit_audit_analysis_sha256": environment["fit_audit_analysis_sha256"],
        "operator_dev_analysis_sha256": environment["operator_dev_analysis_sha256"],
        "audit": {
            "row_count": len(rows),
            "planned_rows": environment.get("planned_rows"),
            "unique_case_keys": len(keys),
            "observed_key_sha256": observed_key_sha,
            "expected_key_sha256": environment.get("expected_key_sha256"),
            "nonfinite_case_keys": nonfinite[:20],
            "success": audit_success,
        },
        "application_gated_identity": application_identity,
        "structural_matched_gate_identity": structural_identity,
        "external_zero_gate_identity": external_zero_identity,
        "family_reports": family_reports,
        "every_family_passes_individually": every_family_passes,
        "forced_directions": ["positive", "negative"],
        "controls_admissible": controls_admissible,
        "candidate_eligible": prior_eligible and controls_admissible,
        "candidate_may_be_locked": controls_admissible,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing directional-control analysis: {args.output}")
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.editor_contract.read_text())
    report = analyze(rows, environment, contract, replicates=args.bootstrap_replicates)
    report.update(
        {
            "input_sha256": sha256_file(args.input),
            "environment_sha256": sha256_file(args.environment),
            "editor_contract_sha256": sha256_file(args.editor_contract),
        }
    )
    atomic_write_text(args.output, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["audit"]["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
