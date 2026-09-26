#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.analyze_crossover import mean, percentile


TOLERANCE_FIELD = {
    "fresh_verification": "fresh_margin_loss",
    "matched_verification": "verification_margin_loss",
    "explicit_governance_reset": "reset_margin_loss",
    "supported_user_authority": "supported_authority_margin_loss",
    "factual_boundary_memory": "factual_memory_margin_loss",
}


def _bootstrap_upper(
    rows: list[dict], field: str, replicates: int, seed: int
) -> dict:
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(float(row[field]))
    cases = sorted(by_case)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [cases[rng.randrange(len(cases))] for _ in cases]
        draws.append(mean([value for case in sampled for value in by_case[case]]))
    return {
        "estimate": mean([float(row[field]) for row in rows]),
        "upper_one_sided_95": percentile(draws, 0.95),
        "replicates": replicates,
        "seed": seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--allow-overwrite", action="store_true")
    args = parser.parse_args()
    rows = load_jsonl(args.input)
    environment = json.loads(args.environment.read_text())
    contract = json.loads(args.operator_contract.read_text())
    tolerances = contract["collateral_tolerances"]
    keys = Counter(row["case_key"] for row in rows)
    observed_key_hash = hashlib.sha256(
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
    family_reports = {}
    all_admissible = True
    for family_index, family in enumerate(sorted(TOLERANCE_FIELD)):
        selected = [row for row in rows if row["control_family"] == family]
        if not selected:
            raise ValueError(f"missing protected control family: {family}")
        margin_tolerance = float(tolerances[TOLERANCE_FIELD[family]])
        mode_reports = {}
        for mode_index, mode in enumerate(("application_gated", "forced_on")):
            loss_field = f"{mode}_margin_loss"
            kl_field = f"{mode}_binary_kl"
            loss = _bootstrap_upper(
                selected,
                loss_field,
                args.bootstrap_replicates,
                61001 + family_index * 10 + mode_index,
            )
            kl = _bootstrap_upper(
                selected,
                kl_field,
                args.bootstrap_replicates,
                62003 + family_index * 10 + mode_index,
            )
            mode_admissible = (
                loss["upper_one_sided_95"] <= margin_tolerance
                and kl["upper_one_sided_95"] <= float(tolerances["binary_answer_kl"])
            )
            mode_reports[mode] = {
                "rows": len(selected),
                "margin_loss": loss,
                "margin_loss_tolerance": margin_tolerance,
                "binary_kl": kl,
                "binary_kl_tolerance": float(tolerances["binary_answer_kl"]),
                "admissible": mode_admissible,
            }
            all_admissible = all_admissible and mode_admissible
        family_reports[family] = mode_reports

    memory_rows = [row for row in rows if row["control_family"] == "factual_boundary_memory"]
    memory_gate_max_logit_error = max(
        max(
            abs(float(row["application_gated"]["logit_a"]) - float(row["baseline"]["logit_a"])),
            abs(float(row["application_gated"]["logit_b"]) - float(row["baseline"]["logit_b"])),
        )
        for row in memory_rows
    )
    applicability_gate_exact = memory_gate_max_logit_error == 0.0
    all_admissible = all_admissible and applicability_gate_exact
    audit_success = (
        len(rows) == int(environment.get("planned_rows", -1))
        and len(keys) == len(rows)
        and observed_key_hash == environment.get("expected_key_sha256")
        and not nonfinite
    )
    report = {
        "schema_version": 1,
        "input_sha256": sha256_file(args.input),
        "environment_sha256": sha256_file(args.environment),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "candidate_id": environment["candidate_id"],
        "dev_half": environment["stage"],
        "audit": {
            "row_count": len(rows),
            "planned_rows": environment.get("planned_rows"),
            "unique_case_keys": len(keys),
            "observed_key_sha256": observed_key_hash,
            "expected_key_sha256": environment.get("expected_key_sha256"),
            "nonfinite_case_keys": nonfinite[:20],
            "success": audit_success,
        },
        "family_reports": family_reports,
        "non_governance_applicability_gate": {
            "factual_memory_application_gated_max_selected_logit_error": memory_gate_max_logit_error,
            "exact_identity": applicability_gate_exact,
        },
        "controls_admissible": audit_success and all_admissible,
        "forced_on_stress_required": True,
        "candidate_may_be_locked": False,
        "final_test_open": False,
        "production_rollout_approved": False,
    }
    atomic_write_text(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        args.allow_overwrite,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not audit_success:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
