#!/usr/bin/env python3
"""Create one immutable, SHA-bound crossover-stage execution authorization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, sha256_file


SHA256_RE = re.compile(r"[0-9a-f]{64}")
REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _sha256(value: str, field: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise ValueError(f"invalid SHA256 for {field}")
    return value


def _prerequisite(value: str) -> dict[str, str]:
    path, separator, digest = value.rpartition("=")
    if not separator or not path.startswith(f"{REMOTE_ROOT}/"):
        raise argparse.ArgumentTypeError(
            "prerequisite must be /workspace/context-mismatch-qwen3-8b/...=SHA256"
        )
    try:
        digest = _sha256(digest, "prerequisite")
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    return {"path": path, "sha256": digest}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("smoke", "discovery", "replication"))
    parser.add_argument("--code-version", required=True, type=int)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--prerequisite", action="append", default=[], type=_prerequisite)
    parser.add_argument("--created-utc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.code_version < 17:
        raise ValueError("crossover code version must be at least 17")
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    contract = json.loads(args.contract.read_text())
    design = json.loads(args.design_audit.read_text())
    manifest_sha = sha256_file(args.manifest)
    contract_sha = sha256_file(args.contract)
    if contract.get("status") != "frozen_before_any_crossover_forward":
        raise ValueError("crossover contract is not frozen")
    if contract.get("production_rollout_approved") is not False:
        raise ValueError("production boundary changed")
    if design.get("stage") != args.stage or design.get("audit", {}).get("success") is not True:
        raise ValueError("stage design audit did not pass")
    if design.get("contract_sha256") != contract_sha:
        raise ValueError("design audit contract binding mismatch")
    if design.get("manifest_sha256") != manifest_sha:
        raise ValueError("design audit manifest binding mismatch")
    expected_rows = int(contract["stages"][args.stage]["expected_rows"])
    if design["audit"].get("row_count") != expected_rows:
        raise ValueError("design audit row count mismatch")
    expected_key_sha = _sha256(
        design["audit"].get("expected_key_sha256", ""), "expected key set"
    )
    bundle_sha = _sha256(args.bundle_manifest_sha256, "bundle manifest")
    created_utc = args.created_utc or (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created_utc):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    code_root = f"{REMOTE_ROOT}/code-v{args.code_version}"
    authorization = {
        "schema_version": 1,
        "authorization_id": (
            f"context-mismatch-qwen3-8b-governance-crossover-{args.stage}-"
            f"code-v{args.code_version}"
        ),
        "created_utc": created_utc,
        "stage": args.stage,
        "code_root": code_root,
        "execution_allowed": True,
        "production_rollout_approved": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "contract_sha256": contract_sha,
        "manifest_sha256": manifest_sha,
        "design_audit_sha256": sha256_file(args.design_audit),
        "expected_key_sha256": expected_key_sha,
        "expected_rows": expected_rows,
        "immutable_code_bundle_manifest_sha256": bundle_sha,
        "prerequisite_artifacts": args.prerequisite,
        "notes": (
            "Stage-specific development authorization. It is immutable after the first "
            "model forward and never opens final_test or production rollout."
        ),
    }
    atomic_write_text(args.output, json.dumps(authorization, indent=2, sort_keys=True) + "\n")
    print(json.dumps(authorization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
