#!/usr/bin/env python3
"""Create one immutable SHA-bound code-v33 ADSGE-V4 router-fit authorization."""

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--v3-contract", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-checkpoint-remote", required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--v3-fit-report-remote", required=True)
    parser.add_argument("--capture-manifest", type=Path, required=True)
    parser.add_argument("--capture-manifest-remote", required=True)
    parser.add_argument("--protected-capture-manifest", type=Path, required=True)
    parser.add_argument("--protected-capture-manifest-remote", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing V4 authorization: {args.output}")
    if not re.fullmatch(r"[0-9a-f]{64}", args.bundle_manifest_sha256):
        raise ValueError("bundle SHA must be lowercase SHA256 hex")
    if not re.fullmatch(
        r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z",
        args.created_utc,
    ):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    contract = json.loads(args.editor_contract.read_text())
    if contract.get("method_short_name") != "ADSGE-V4":
        raise ValueError("unexpected V4 contract")
    if contract.get("final_test_open") is not False or contract.get("final_test_open_count") != 0:
        raise ValueError("V4 contract unexpectedly opens final test")
    if contract.get("production_rollout_approved") is not False:
        raise ValueError("V4 contract unexpectedly approves production")
    sources = {
        "v3_checkpoint": _source(args.v3_checkpoint, args.v3_checkpoint_remote),
        "v3_fit_report": _source(args.v3_fit_report, args.v3_fit_report_remote),
        "capture_manifest": _source(args.capture_manifest, args.capture_manifest_remote),
        "protected_capture_manifest": _source(
            args.protected_capture_manifest, args.protected_capture_manifest_remote
        ),
    }
    bindings = contract["bound_v3_evidence"]
    fit_inputs = contract["bound_fit_inputs"]
    expected = {
        "v3_contract_sha256": sha256_file(args.v3_contract),
        "selected_checkpoint_sha256": sources["v3_checkpoint"]["sha256"],
        "fit_report_sha256": sources["v3_fit_report"]["sha256"],
    }
    for field, value in expected.items():
        if bindings.get(field) != value:
            raise ValueError(f"V4 V3 evidence binding mismatch: {field}")
    for field, source in (
        ("governance_capture_manifest_sha256", sources["capture_manifest"]),
        ("protected_capture_manifest_sha256", sources["protected_capture_manifest"]),
    ):
        if fit_inputs.get(field) != source["sha256"]:
            raise ValueError(f"V4 fit-input binding mismatch: {field}")
    value = {
        "schema_version": 1,
        "authorization_id": "qwen3-8b-governance-abstaining-router-fit-code-v33",
        "created_utc": args.created_utc,
        "stage": "governance_abstaining_router_fit",
        "code_root": f"{REMOTE_ROOT}/code-v33",
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "execution_allowed": True,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "v3_contract_sha256": sha256_file(args.v3_contract),
        **sources,
        "v3_checkpoint_sha256": sources["v3_checkpoint"]["sha256"],
        "v3_fit_report_sha256": sources["v3_fit_report"]["sha256"],
        "capture_manifest_sha256": sources["capture_manifest"]["sha256"],
        "protected_capture_manifest_sha256": sources["protected_capture_manifest"]["sha256"],
        "operator_dev_accessed": False,
        "protected_behavior_outputs_accessed": False,
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
