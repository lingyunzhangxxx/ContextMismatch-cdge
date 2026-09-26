#!/usr/bin/env python3
"""Materialize one SHA-bound V5 component-discovery audit-capture authorization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"
MINIMUM_CODE_VERSION = 35


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
    ).hexdigest()


def _design_audit_bindings(design_audit: Path) -> dict[str, str]:
    """Bind both the generic runner key and the V5-specific provenance key."""
    digest = sha256_file(design_audit)
    return {
        "design_audit_sha256": digest,
        "component_discovery_design_audit_sha256": digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-version", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--router-contract", type=Path, required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--controls", type=Path, required=True)
    parser.add_argument("--mechanism-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--behavior-analysis", type=Path, required=True)
    parser.add_argument("--model-manifest-remote", required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.code_version < MINIMUM_CODE_VERSION:
        raise ValueError(
            f"V5 audit capture compatibility fix requires code-v{MINIMUM_CODE_VERSION} or newer"
        )
    if len(args.bundle_manifest_sha256) != 64:
        raise ValueError("bundle manifest SHA must be lowercase SHA256 hex")
    int(args.bundle_manifest_sha256, 16)
    if len(args.model_manifest_sha256) != 64:
        raise ValueError("model manifest SHA must be lowercase SHA256 hex")
    int(args.model_manifest_sha256, 16)
    if not args.model_manifest_remote.startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("model manifest must be inside the Qwen3-8B project root")

    router = json.loads(args.router_contract.read_text())
    crossover = json.loads(args.crossover_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    controls = json.loads(args.controls.read_text())
    behavior = json.loads(args.behavior_analysis.read_text())
    if router.get("status") != "frozen_before_any_v5_capture_or_fit":
        raise ValueError("V5 router contract is not frozen")
    if int(router.get("code_version_minimum", -1)) > args.code_version:
        raise ValueError("code version is below the V5 contract minimum")
    if router.get("final_test_open") or router.get("final_test_open_count") != 0:
        raise ValueError("V5 router contract unexpectedly opens final_test")
    if router.get("production_rollout_approved"):
        raise ValueError("V5 router contract unexpectedly approves production")
    if controls.get("frozen_before_any_v5_capture_forward") is not True:
        raise ValueError("V5 audit controls are not frozen")
    if controls.get("output_blind_construction") is not True:
        raise ValueError("V5 audit controls are not output-blind")
    if not behavior.get("audit", {}).get("success"):
        raise ValueError("bound full behavior analysis did not pass its audit")
    if design.get("stage") != "discovery" or not design.get("audit", {}).get("success"):
        raise ValueError("component-discovery design audit is invalid")
    if design.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("design/crossover SHA mismatch")
    if design.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("design/manifest SHA mismatch")

    static = router["bound_static_inputs"]
    required_static = {
        "governance_crossover_contract_sha256": sha256_file(args.crossover_contract),
        "capture_editor_contract_sha256": sha256_file(args.editor_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "component_discovery_design_audit_sha256": sha256_file(args.design_audit),
        "v5_audit_controls_sha256": sha256_file(args.controls),
    }
    for field, expected in required_static.items():
        if static.get(field) != expected:
            raise ValueError(f"V5 static input mismatch: {field}")

    jobs = enumerate_jobs(load_jsonl(args.manifest), crossover, "discovery")
    expected_governance_rows = 6144
    if len(jobs) != expected_governance_rows:
        raise ValueError(f"unexpected discovery row count: {len(jobs)}")
    expected_governance_key = _key_hash(jobs)
    audit_contract = router["developmental_audit_capture"]
    if audit_contract.get("governance_expected_key_sha256") != expected_governance_key:
        raise ValueError("V5 contract discovery key SHA mismatch")

    value = {
        "schema_version": 1,
        "authorization_id": (
            f"qwen3-8b-governance-consensus-router-audit-capture-code-v{args.code_version}"
        ),
        "created_utc": args.created_utc,
        "stage": "governance_consensus_router_audit_capture",
        "code_root": f"{REMOTE_ROOT}/code-v{args.code_version}",
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "execution_allowed": True,
        "router_contract_sha256": sha256_file(args.router_contract),
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        **_design_audit_bindings(args.design_audit),
        "v5_audit_controls_sha256": sha256_file(args.controls),
        "mechanism_contract_sha256": sha256_file(args.mechanism_contract),
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "execution_contract_sha256": sha256_file(args.execution_contract),
        "behavior_analysis_sha256": sha256_file(args.behavior_analysis),
        "governance_crossover_stage": "discovery",
        "governance_partition": "component_discovery",
        "expected_governance_rows": expected_governance_rows,
        "expected_rows": expected_governance_rows,
        "expected_governance_key_sha256": expected_governance_key,
        "expected_key_sha256": expected_governance_key,
        "protected_partition": "component_discovery",
        "expected_protected_rows": 4008,
        "expected_protected_family_rows": audit_contract["protected_family_rows"],
        "model_manifest": {
            "path": args.model_manifest_remote,
            "sha256": args.model_manifest_sha256,
        },
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
