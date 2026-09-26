#!/usr/bin/env python3
"""Audit six-family AMSGE protected controls without opening the final test."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing control analysis: {args.output}")
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.editor_contract.read_text())
    keys = Counter(row["case_key"] for row in rows)
    observed_key_sha256 = hashlib.sha256(
        (("\n".join(sorted(keys))) + "\n").encode("utf-8")
    ).hexdigest()
    nonfinite = []
    for row in rows:
        numeric = [
            *row["baseline"].values(),
            *row["application_gated"].values(),
            *row["forced_on_stress"].values(),
            row["application_gated_margin_change"],
            row["forced_on_margin_change"],
            row["application_gated_binary_kl"],
            row["forced_on_binary_kl"],
        ]
        if not all(math.isfinite(float(value)) for value in numeric):
            nonfinite.append(row["case_key"])
        row["application_gated_margin_loss"] = -float(
            row["application_gated_margin_change"]
        )
        row["forced_on_margin_loss"] = -float(row["forced_on_margin_change"])

    control_contract = contract["protected_controls"]
    margin_limit = float(
        control_contract.get(
            "forced_on_margin_loss_upper_95_maximum",
            control_contract.get("forced_on_matched_margin_loss_upper_95_maximum"),
        )
    )
    kl_limit = float(
        control_contract.get(
            "forced_on_binary_kl_upper_95_maximum",
            control_contract.get("forced_on_binary_kl_upper_95_maximum", 0.02),
        )
    )
    family_reports = {}
    all_reference_gates = True
    for family_index, family in enumerate(FAMILIES):
        selected = [row for row in rows if row["control_family"] == family]
        if not selected:
            raise ValueError(f"missing governance control family: {family}")
        mode_reports = {}
        for mode_index, mode in enumerate(("application_gated", "forced_on")):
            loss = _bootstrap_upper(
                selected,
                f"{mode}_margin_loss",
                args.bootstrap_replicates,
                71011 + family_index * 10 + mode_index,
            )
            kl = _bootstrap_upper(
                selected,
                f"{mode}_binary_kl",
                args.bootstrap_replicates,
                72019 + family_index * 10 + mode_index,
            )
            reference_gate_pass = (
                float(loss["upper_one_sided_95"]) <= margin_limit
                and float(kl["upper_one_sided_95"]) <= kl_limit
            )
            mode_reports[mode] = {
                "rows": len(selected),
                "margin_loss": loss,
                "margin_loss_reference_maximum": margin_limit,
                "binary_kl": kl,
                "binary_kl_reference_maximum": kl_limit,
                "reference_gate_pass": reference_gate_pass,
            }
            all_reference_gates = all_reference_gates and reference_gate_pass
        family_reports[family] = mode_reports

    non_applicable = [
        row for row in rows if float(row["operator_applicable"]) == 0.0
    ]
    if not non_applicable:
        raise ValueError("no non-applicable governance controls were observed")
    gated_max_error = max(
        max(
            abs(
                float(row["application_gated"]["logit_a"])
                - float(row["baseline"]["logit_a"])
            ),
            abs(
                float(row["application_gated"]["logit_b"])
                - float(row["baseline"]["logit_b"])
            ),
        )
        for row in non_applicable
    )
    exact_identity = gated_max_error == 0.0
    audit_success = (
        len(rows) == int(environment.get("planned_rows", -1)) == 2856
        and len(keys) == len(rows)
        and observed_key_sha256 == environment.get("expected_key_sha256")
        and not nonfinite
    )
    diagnostic = bool(environment.get("post_failure_characterization"))
    controls_would_pass = audit_success and exact_identity and all_reference_gates
    report = {
        "schema_version": 1,
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "candidate_id": environment["candidate_id"],
        "post_failure_characterization": diagnostic,
        "fit_gates_passed": bool(environment.get("fit_gates_passed")),
        "audit": {
            "row_count": len(rows),
            "planned_rows": environment.get("planned_rows"),
            "unique_case_keys": len(keys),
            "observed_key_sha256": observed_key_sha256,
            "expected_key_sha256": environment.get("expected_key_sha256"),
            "nonfinite_case_keys": nonfinite[:20],
            "success": audit_success,
        },
        "family_reports": family_reports,
        "non_governance_application_gate": {
            "rows": len(non_applicable),
            "max_selected_logit_error": gated_max_error,
            "exact_identity": exact_identity,
        },
        "controls_would_pass_in_isolation": controls_would_pass,
        "controls_admissible": controls_would_pass and not diagnostic,
        "candidate_eligible": False if diagnostic else None,
        "candidate_may_be_locked": False,
        "forced_on_stress_required": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if diagnostic:
        report["ineligibility_reason"] = (
            "AMSGE V1 failed the frozen fit gates; complete controls are diagnostic only."
        )
    atomic_write_text(
        args.output, json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not audit_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
