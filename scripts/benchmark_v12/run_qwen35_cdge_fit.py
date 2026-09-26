#!/usr/bin/env python3
"""Fit the frozen Qwen3.5 C-DGE structural editor and applicability veto."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


def _write(path: Path, value: object) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _require_authorization(args: argparse.Namespace) -> dict:
    authorization = json.loads(args.execution_authorization.read_text())
    required = {
        "stage": "qwen35_cdge_fit",
        "method": "C-DGE-V4.1",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(
            args.code_root / "bundle.sha256"
        ),
        "execution_allowed": True,
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "gradient_manifest_sha256": sha256_file(args.gradient_manifest),
        "protected_manifest_sha256": sha256_file(args.protected_manifest),
        "replication_protocol_sha256": sha256_file(args.replication_protocol),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "v3_template_sha256": sha256_file(args.v3_template),
        "v4_template_sha256": sha256_file(args.v4_template),
        "composite_template_sha256": sha256_file(args.composite_template),
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if authorization.get(field) != expected:
            raise ValueError(f"fit authorization mismatch: {field}")
    receipts = authorization.get("capture_archive_receipts", {})
    if set(receipts) != {"governance", "gradient", "protected"}:
        raise ValueError("fit authorization does not bind all capture receipts")
    for stage, receipt in receipts.items():
        if receipt.get("three_copy_verified") is not True:
            raise ValueError(f"{stage} capture receipt is not three-copy verified")
        if not isinstance(receipt.get("archive_sha256"), str) or len(
            receipt["archive_sha256"]
        ) != 64:
            raise ValueError(f"{stage} capture archive SHA is invalid")
    return authorization


def _derived_v3_contract(
    *,
    template: dict,
    protocol: dict,
    created_utc: str,
    args: argparse.Namespace,
) -> dict:
    frozen = dict(protocol["replication_scope"]["frozen_site"])
    frozen["max_relative_correction"] = frozen.pop("maximum_relative_correction")
    contract = dict(template)
    contract.update(
        {
            "contract_id": (
                "context-mismatch-qwen3-5-9b-directional-structural-"
                "governance-editor-v3-replication"
            ),
            "created_utc": created_utc,
            "frozen_utc": created_utc,
            "evidence_role": "prospective_cross_model_replication",
            "cross_model_replication_protocol_sha256": sha256_file(
                args.replication_protocol
            ),
            "candidate_sites": [frozen],
        }
    )
    contract["bound_inputs"] = dict(template["bound_inputs"])
    contract["bound_inputs"].update(
        {
            "governance_capture_manifest_sha256": sha256_file(
                args.capture_manifest
            ),
            "protected_capture_manifest_sha256": sha256_file(
                args.protected_manifest
            ),
            "governance_crossover_contract_sha256": sha256_file(
                args.crossover_contract
            ),
            "benchmark_manifest_sha256": protocol["frozen_lineage"][
                "benchmark_manifest_sha256"
            ],
        }
    )
    return contract


def _derived_v4_contract(
    *,
    template: dict,
    created_utc: str,
    protocol_sha256: str,
    args: argparse.Namespace,
    v3_contract: Path,
    v3_checkpoint: Path,
    v3_report: Path,
) -> dict:
    contract = dict(template)
    contract.update(
        {
            "contract_id": (
                "context-mismatch-qwen3-5-9b-abstaining-directional-"
                "governance-editor-v4-replication"
            ),
            "created_utc": created_utc,
            "frozen_utc": created_utc,
            "evidence_role": "prospective_cross_model_replication",
            "cross_model_replication_protocol_sha256": protocol_sha256,
        }
    )
    contract["bound_v3_evidence"] = dict(template["bound_v3_evidence"])
    contract["bound_v3_evidence"].update(
        {
            "v3_contract_sha256": sha256_file(v3_contract),
            "selected_checkpoint_sha256": sha256_file(v3_checkpoint),
            "fit_report_sha256": sha256_file(v3_report),
            "selected_candidate_id": "DSGE-V3-27:mlp",
        }
    )
    contract["bound_fit_inputs"] = dict(template["bound_fit_inputs"])
    contract["bound_fit_inputs"].update(
        {
            "governance_capture_manifest_sha256": sha256_file(
                args.capture_manifest
            ),
            "protected_capture_manifest_sha256": sha256_file(
                args.protected_manifest
            ),
        }
    )
    return contract


def _derived_composite_contract(
    *,
    template: dict,
    protocol: dict,
    created_utc: str,
    args: argparse.Namespace,
    v3_contract: Path,
    v3_checkpoint: Path,
    v3_report: Path,
    v4_contract: Path,
    v4_checkpoint: Path,
    v4_report: Path,
) -> dict:
    contract = dict(template)
    contract.update(
        {
            "contract_id": (
                "context-mismatch-qwen3-5-9b-composite-directional-"
                "governance-editor-v4-1-replication"
            ),
            "created_utc": created_utc,
            "frozen_utc": created_utc,
            "status": "prospective_cross_model_composite_eligibility_frozen",
            "evidence_role": "prospective_cross_model_replication",
            "confirmatory_claim_authorized": False,
            "cross_model_replication_protocol_sha256": sha256_file(
                args.replication_protocol
            ),
        }
    )
    contract["frozen_lineage"] = {
        "v3_contract_sha256": sha256_file(v3_contract),
        "v3_checkpoint_sha256": sha256_file(v3_checkpoint),
        "v3_fit_report_sha256": sha256_file(v3_report),
        "v4_contract_sha256": sha256_file(v4_contract),
        "v4_checkpoint_sha256": sha256_file(v4_checkpoint),
        "v4_fit_report_sha256": sha256_file(v4_report),
        "governance_capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_manifest),
    }
    frozen = protocol["fit_and_behavior_gates"]
    parent_gates = template["eligibility_gates"]
    contract["eligibility_gates"] = {
        "calibration_governance_matched_composite_active_fraction_maximum_each_family": frozen[
            "calibration_composite_matched_active_fraction_maximum"
        ],
        "calibration_protected_composite_active_fraction_maximum_each_family": frozen[
            "calibration_composite_protected_active_fraction_maximum_each_family"
        ],
        "calibration_composite_mismatch_true_positive_rate_minimum": frozen[
            "calibration_composite_mismatch_true_positive_rate_minimum"
        ],
        "audit_governance_matched_composite_active_fraction_maximum_each_family": frozen[
            "audit_composite_matched_active_fraction_maximum"
        ],
        "audit_protected_composite_active_fraction_maximum_each_family": frozen[
            "audit_composite_protected_active_fraction_maximum_each_family"
        ],
        "audit_composite_mismatch_true_positive_rate_minimum": frozen[
            "audit_composite_mismatch_true_positive_rate_minimum"
        ],
        "expected_governance_matched_families": parent_gates[
            "expected_governance_matched_families"
        ],
        "expected_protected_families": parent_gates["expected_protected_families"],
        "v3_embedded_checkpoint_exact_match": True,
        "v3_frozen_parameter_max_absolute_change": 0.0,
        "v4_checkpoint_contains_no_base_weights": True,
        "v4_checkpoint_contains_exactly_one_site": True,
        "all_metrics_finite": True,
        "all_source_coverage_exact": True,
    }
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "code_root",
        "replication_protocol",
        "crossover_contract",
        "v3_template",
        "v4_template",
        "composite_template",
        "capture_manifest",
        "gradient_manifest",
        "protected_manifest",
        "execution_authorization",
        "output_dir",
    ):
        parser.add_argument(
            "--" + name.replace("_", "-"), type=Path, required=True
        )
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    authorization = _require_authorization(args)
    protocol = json.loads(args.replication_protocol.read_text())
    if protocol.get("method_short_name") != "C-DGE-V4.1":
        raise ValueError("unexpected replication protocol")
    args.output_dir.mkdir(parents=True)
    created_utc = str(authorization["created_utc"])

    v3_contract = _derived_v3_contract(
        template=json.loads(args.v3_template.read_text()),
        protocol=protocol,
        created_utc=created_utc,
        args=args,
    )
    v3_path = (
        args.output_dir / "QWEN3_5_9B_DIRECTIONAL_STRUCTURAL_GOVERNANCE_EDITOR_V3.json"
    )
    _write(v3_path, v3_contract)
    v3_output = args.output_dir / "directional_fit"
    v3_authorization = {
        "schema_version": 1,
        "stage": "governance_directional_fit",
        "execution_allowed": True,
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(
            args.code_root / "bundle.sha256"
        ),
        "editor_contract_sha256": sha256_file(v3_path),
        "capture_manifest_sha256": sha256_file(args.capture_manifest),
        "protected_capture_manifest_sha256": sha256_file(args.protected_manifest),
        "gradient_capture_manifest_sha256": sha256_file(args.gradient_manifest),
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    v3_authorization_path = args.output_dir / "directional_fit_authorization.json"
    _write(v3_authorization_path, v3_authorization)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.benchmark_v4.fit_directional_governance",
            "--editor-contract",
            str(v3_path),
            "--capture-manifest",
            str(args.capture_manifest),
            "--protected-capture-manifest",
            str(args.protected_manifest),
            "--gradient-capture-manifest",
            str(args.gradient_manifest),
            "--execution-authorization",
            str(v3_authorization_path),
            "--output-dir",
            str(v3_output),
        ],
        check=True,
        cwd=args.code_root,
    )
    v3_report_path = v3_output / "directional_fit_report.json"
    v3_report = json.loads(v3_report_path.read_text())
    candidates = [
        row
        for row in v3_report.get("candidate_reports", [])
        if row.get("candidate_id") == "DSGE-V3-27:mlp"
    ]
    if len(candidates) != 1 or v3_report.get("candidate_count") != 1:
        raise ValueError("directional fit did not materialize exactly the frozen site")
    candidate = candidates[0]
    v3_checkpoint = v3_output / candidate["checkpoint"]
    if sha256_file(v3_checkpoint) != candidate.get("checkpoint_sha256"):
        raise ValueError("directional checkpoint SHA mismatch")

    applicability = {"attempted": False, "fit_eligible": False}
    composite = {"attempted": False, "candidate_eligible": False}
    if candidate.get("fit_eligible") is True:
        v4_path = (
            args.output_dir
            / "QWEN3_5_9B_ABSTAINING_DIRECTIONAL_GOVERNANCE_EDITOR_V4.json"
        )
        _write(
            v4_path,
            _derived_v4_contract(
                template=json.loads(args.v4_template.read_text()),
                created_utc=created_utc,
                protocol_sha256=sha256_file(args.replication_protocol),
                args=args,
                v3_contract=v3_path,
                v3_checkpoint=v3_checkpoint,
                v3_report=v3_report_path,
            ),
        )
        v4_output = args.output_dir / "applicability_fit"
        v4_authorization = {
            "schema_version": 1,
            "stage": "governance_abstaining_router_fit",
            "execution_allowed": True,
            "code_root": str(args.code_root),
            "immutable_code_bundle_manifest_sha256": sha256_file(
                args.code_root / "bundle.sha256"
            ),
            "editor_contract_sha256": sha256_file(v4_path),
            "v3_contract_sha256": sha256_file(v3_path),
            "v3_checkpoint_sha256": sha256_file(v3_checkpoint),
            "v3_fit_report_sha256": sha256_file(v3_report_path),
            "capture_manifest_sha256": sha256_file(args.capture_manifest),
            "protected_capture_manifest_sha256": sha256_file(
                args.protected_manifest
            ),
            "operator_dev_accessed": False,
            "protected_behavior_outputs_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        v4_authorization_path = args.output_dir / "applicability_fit_authorization.json"
        _write(v4_authorization_path, v4_authorization)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.benchmark_v5.fit_abstaining_router",
                "--editor-contract",
                str(v4_path),
                "--v3-contract",
                str(v3_path),
                "--v3-checkpoint",
                str(v3_checkpoint),
                "--v3-fit-report",
                str(v3_report_path),
                "--capture-manifest",
                str(args.capture_manifest),
                "--protected-capture-manifest",
                str(args.protected_manifest),
                "--execution-authorization",
                str(v4_authorization_path),
                "--output-dir",
                str(v4_output),
            ],
            check=True,
            cwd=args.code_root,
        )
        v4_report_path = v4_output / "abstaining_router_fit_report.json"
        v4_report = json.loads(v4_report_path.read_text())
        v4_checkpoint = v4_output / v4_report["checkpoint"]
        if sha256_file(v4_checkpoint) != v4_report.get("checkpoint_sha256"):
            raise ValueError("applicability checkpoint SHA mismatch")
        applicability = {
            "attempted": True,
            "fit_eligible": bool(v4_report["fit_eligible"]),
            "report": "applicability_fit/abstaining_router_fit_report.json",
            "checkpoint": "applicability_fit/" + v4_report["checkpoint"],
            "checkpoint_sha256": v4_report["checkpoint_sha256"],
        }
        composite_path = (
            args.output_dir
            / "QWEN3_5_9B_COMPOSITE_DIRECTIONAL_GOVERNANCE_EDITOR_V4_1.json"
        )
        _write(
            composite_path,
            _derived_composite_contract(
                template=json.loads(args.composite_template.read_text()),
                protocol=protocol,
                created_utc=created_utc,
                args=args,
                v3_contract=v3_path,
                v3_checkpoint=v3_checkpoint,
                v3_report=v3_report_path,
                v4_contract=v4_path,
                v4_checkpoint=v4_checkpoint,
                v4_report=v4_report_path,
            ),
        )
        composite_authorization = {
            "schema_version": 1,
            "stage": "governance_composite_eligibility_audit",
            "method": "C-DGE-V4.1",
            "execution_allowed": True,
            "code_root": str(args.code_root),
            "immutable_code_bundle_manifest_sha256": sha256_file(
                args.code_root / "bundle.sha256"
            ),
            "composite_contract_sha256": sha256_file(composite_path),
            "operator_dev_accessed": False,
            "protected_behavior_outputs_accessed": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
        composite_authorization_path = (
            args.output_dir / "composite_eligibility_authorization.json"
        )
        _write(composite_authorization_path, composite_authorization)
        composite_report_path = args.output_dir / "composite_eligibility_report.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.benchmark_v9.audit_composite_eligibility",
                "--composite-contract",
                str(composite_path),
                "--v4-contract",
                str(v4_path),
                "--v3-contract",
                str(v3_path),
                "--v4-checkpoint",
                str(v4_checkpoint),
                "--v4-fit-report",
                str(v4_report_path),
                "--v3-checkpoint",
                str(v3_checkpoint),
                "--v3-fit-report",
                str(v3_report_path),
                "--capture-manifest",
                str(args.capture_manifest),
                "--protected-capture-manifest",
                str(args.protected_manifest),
                "--execution-authorization",
                str(composite_authorization_path),
                "--output",
                str(composite_report_path),
            ],
            check=True,
            cwd=args.code_root,
        )
        composite_report = json.loads(composite_report_path.read_text())
        composite = {
            "attempted": True,
            "candidate_eligible": bool(composite_report["candidate_eligible"]),
            "report": "composite_eligibility_report.json",
            "report_sha256": sha256_file(composite_report_path),
            "contract": composite_path.name,
            "contract_sha256": sha256_file(composite_path),
        }

    report = {
        "schema_version": 1,
        "stage": "qwen35_cdge_fit",
        "method": "C-DGE-V4.1",
        "fit_complete": True,
        "directional_fit_eligible": bool(candidate.get("fit_eligible")),
        "directional_candidate": candidate["candidate_id"],
        "directional_checkpoint": "directional_fit/" + candidate["checkpoint"],
        "directional_checkpoint_sha256": candidate["checkpoint_sha256"],
        "applicability": applicability,
        "composite_eligibility": composite,
        "all_fit_gates_pass": bool(candidate.get("fit_eligible"))
        and composite["candidate_eligible"],
        "valid_negative_result": not (
            bool(candidate.get("fit_eligible")) and composite["candidate_eligible"]
        ),
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
        "authorization_sha256": sha256_file(args.execution_authorization),
    }
    _write(args.output_dir / "qwen35_cdge_fit_report.json", report)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
