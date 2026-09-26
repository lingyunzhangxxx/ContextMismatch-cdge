#!/usr/bin/env python3
"""Create one immutable SHA-bound code-v35 GRC-DGE-V5 fit authorization."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _source(local: Path, remote: str) -> dict:
    if not local.is_file():
        raise FileNotFoundError(local)
    if not remote.startswith(REMOTE_ROOT + "/") or ".." in Path(remote).parts:
        raise ValueError(f"unsafe remote source path: {remote}")
    return {"path": remote, "sha256": sha256_file(local)}


def _safe_contract(value: dict, *, method: str) -> None:
    required = {
        "method_short_name": method,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required.items():
        if value.get(field) != expected:
            raise ValueError(f"V5 router-contract mismatch: {field}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-version", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--router-contract", type=Path, required=True)
    parser.add_argument("--fit-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-remote", required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--v3-fit-report-remote", required=True)
    parser.add_argument("--train-governance-manifest", type=Path, required=True)
    parser.add_argument("--train-governance-manifest-remote", required=True)
    parser.add_argument("--train-protected-manifest", type=Path, required=True)
    parser.add_argument("--train-protected-manifest-remote", required=True)
    parser.add_argument("--audit-governance-manifest", type=Path, required=True)
    parser.add_argument("--audit-governance-manifest-remote", required=True)
    parser.add_argument("--audit-protected-manifest", type=Path, required=True)
    parser.add_argument("--audit-protected-manifest-remote", required=True)
    parser.add_argument("--capture-authorization", type=Path, required=True)
    parser.add_argument("--capture-authorization-remote", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing V5 fit authorization: {args.output}")
    if args.code_version != 35:
        raise ValueError("GRC-DGE-V5 fit authorization is fixed to code-v35")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_manifest_sha256):
        raise ValueError("bundle SHA must be lowercase SHA256 hex")
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        args.created_utc,
    ):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")

    router = json.loads(args.router_contract.read_text())
    fit = json.loads(args.fit_contract.read_text())
    capture_authorization = json.loads(args.capture_authorization.read_text())
    _safe_contract(router, method="GRC-DGE-V5")
    if router.get("status") != "frozen_before_any_v5_capture_or_fit":
        raise ValueError("V5 router contract is not frozen")
    required_fit = {
        "method_short_name": "GRC-DGE-V5",
        "status": "frozen_while_v5_capture_job_9113_pending_before_any_capture_output",
        "capture_outputs_examined_before_freeze": False,
        "architecture_search_forbidden": True,
        "seed_search_forbidden": True,
        "epoch_search_forbidden": True,
        "threshold_search_forbidden": True,
    }
    for field, expected in required_fit.items():
        if fit.get(field) != expected:
            raise ValueError(f"V5 fit-contract mismatch: {field}")
    if fit.get("router_contract_sha256") != sha256_file(args.router_contract):
        raise ValueError("V5 fit/router contract SHA mismatch")
    if fit.get("capture_authorization_sha256") != sha256_file(
        args.capture_authorization
    ):
        raise ValueError("V5 fit/capture authorization SHA mismatch")
    if capture_authorization.get("stage") != "governance_consensus_router_audit_capture":
        raise ValueError("unexpected V5 capture authorization stage")
    for field, expected in {
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }.items():
        if capture_authorization.get(field) != expected:
            raise ValueError(f"V5 capture authorization safety mismatch: {field}")

    sources = {
        "v3_checkpoint": _source(args.v3_checkpoint, args.v3_checkpoint_remote),
        "v3_fit_report": _source(args.v3_fit_report, args.v3_fit_report_remote),
        "train_governance_manifest": _source(
            args.train_governance_manifest, args.train_governance_manifest_remote
        ),
        "train_protected_manifest": _source(
            args.train_protected_manifest, args.train_protected_manifest_remote
        ),
        "audit_governance_manifest": _source(
            args.audit_governance_manifest, args.audit_governance_manifest_remote
        ),
        "audit_protected_manifest": _source(
            args.audit_protected_manifest, args.audit_protected_manifest_remote
        ),
        "capture_authorization": _source(
            args.capture_authorization, args.capture_authorization_remote
        ),
    }
    prior = router["bound_prior_evidence"]
    static = router["bound_static_inputs"]
    for field, expected in {
        "v3_checkpoint_sha256": sources["v3_checkpoint"]["sha256"],
        "v3_fit_report_sha256": sources["v3_fit_report"]["sha256"],
    }.items():
        if prior.get(field) != expected:
            raise ValueError(f"V5 prior-evidence mismatch: {field}")
    for field, source in (
        (
            "subspace_fit_governance_capture_manifest_sha256",
            sources["train_governance_manifest"],
        ),
        (
            "subspace_fit_protected_capture_manifest_sha256",
            sources["train_protected_manifest"],
        ),
    ):
        if static.get(field) != source["sha256"]:
            raise ValueError(f"V5 static training input mismatch: {field}")

    expected_authorization_sha = sources["capture_authorization"]["sha256"]
    expected_router_sha = sha256_file(args.router_contract)
    for name, path, rows in (
        ("governance", args.audit_governance_manifest, 6144),
        ("protected", args.audit_protected_manifest, 4008),
    ):
        manifest = json.loads(path.read_text())
        required = {
            "partition": "component_discovery",
            "rows": rows,
            "authorization_sha256": expected_authorization_sha,
            "router_contract_sha256": expected_router_sha,
            "final_test_open": False,
            "production_rollout_approved": False,
        }
        for field, expected in required.items():
            if manifest.get(field) != expected:
                raise ValueError(f"V5 audit {name} manifest mismatch: {field}")
        if manifest.get("final_test_open_count", 0) != 0:
            raise ValueError(f"V5 audit {name} manifest opens final_test")
    governance = json.loads(args.audit_governance_manifest.read_text())
    if governance.get("complete") is not True:
        raise ValueError("V5 governance audit capture is incomplete")
    if governance.get("unique_job_keys") != 6144 or governance.get(
        "counterfactual_pairs"
    ) != 3072:
        raise ValueError("V5 governance audit identity count mismatch")
    expected_key = router["developmental_audit_capture"][
        "governance_expected_key_sha256"
    ]
    if governance.get("observed_key_sha256") != expected_key or governance.get(
        "expected_key_sha256"
    ) != expected_key:
        raise ValueError("V5 governance audit key SHA mismatch")
    protected = json.loads(args.audit_protected_manifest.read_text())
    if protected.get("family_rows") != router["developmental_audit_capture"][
        "protected_family_rows"
    ]:
        raise ValueError("V5 protected audit family count mismatch")

    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-governance-consensus-router-fit-code-v35",
        "created_utc": args.created_utc,
        "stage": "governance_consensus_router_fit",
        "code_root": f"{REMOTE_ROOT}/code-v35",
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "execution_allowed": True,
        "router_contract_sha256": sha256_file(args.router_contract),
        "fit_contract_sha256": sha256_file(args.fit_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        **sources,
        **{f"{name}_sha256": source["sha256"] for name, source in sources.items()},
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "authorization": str(args.output),
                "authorization_sha256": sha256_file(args.output),
                "stage": value["stage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
