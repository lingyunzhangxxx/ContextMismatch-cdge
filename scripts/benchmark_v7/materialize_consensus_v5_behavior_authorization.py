#!/usr/bin/env python3
"""Create a fresh SHA-bound authorization for the V5/full-DGE behavior pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v2.run_operator_candidate import _half_jobs
from scripts.benchmark_v7.run_same_identity import (
    EXPECTED_KEY_SHA256,
    EXPECTED_ROWS,
    METHOD_SHARDS,
)


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"
EXPECTED_V3_CHECKPOINT_SHA256 = (
    "ed5262bbd27470047c3379172e58de23e6bb98a542b9fb5101f828138b11d752"
)
EXPECTED_V3_FIT_REPORT_SHA256 = (
    "0cda38b3f3ad44460e6f91278277bc1cd66349a5d34f128890d6bc60f60a5f0c"
)


def _record(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    if not str(path).startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("source artifact must remain under the cluster project root")
    return {"path": str(path), "sha256": sha256_file(path)}


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
    ).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--behavior-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest", type=Path, required=True)
    parser.add_argument("--v3-checkpoint", type=Path, required=True)
    parser.add_argument("--v3-fit-report", type=Path, required=True)
    parser.add_argument("--v5-checkpoint", type=Path, required=True)
    parser.add_argument("--v5-fit-report", type=Path, required=True)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if not str(args.code_root).startswith(f"{REMOTE_ROOT}/code-v"):
        raise ValueError("invalid immutable code root")
    code_version = int(str(args.code_root).rsplit("code-v", 1)[1])
    if code_version < 38:
        raise ValueError("V5 behavior requires code-v38 or newer")
    bundle = args.code_root / "bundle.sha256"
    if not bundle.is_file():
        raise FileNotFoundError(bundle)

    behavior = json.loads(args.behavior_contract.read_text())
    if behavior.get("status") != "frozen_before_any_consensus_v5_behavior_forward":
        raise ValueError("V5 behavior contract is not frozen")
    scope = behavior.get("scope", {})
    if scope.get("methods") != list(METHOD_SHARDS["v5"]):
        raise ValueError("V5 behavior method pair changed")
    if scope.get("expected_rows_per_method") != EXPECTED_ROWS:
        raise ValueError("V5 behavior row count changed")
    if scope.get("expected_job_key_sha256") != EXPECTED_KEY_SHA256:
        raise ValueError("V5 behavior key SHA changed")

    crossover = json.loads(args.crossover_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    if design.get("stage") != "operator_dev" or design.get("audit", {}).get("success") is not True:
        raise ValueError("operator-dev design audit is invalid")
    if design.get("contract_sha256") != sha256_file(args.crossover_contract):
        raise ValueError("design/crossover binding mismatch")
    if design.get("manifest_sha256") != sha256_file(args.manifest):
        raise ValueError("design/manifest binding mismatch")
    jobs = _half_jobs(
        enumerate_jobs(load_jsonl(args.manifest), crossover, "operator_dev"), "selection"
    )
    if len(jobs) != EXPECTED_ROWS or _key_hash(jobs) != EXPECTED_KEY_SHA256:
        raise ValueError("frozen V5 behavior job enumeration changed")

    v3_checkpoint = _record(args.v3_checkpoint)
    v3_fit_report = _record(args.v3_fit_report)
    if v3_checkpoint["sha256"] != EXPECTED_V3_CHECKPOINT_SHA256:
        raise ValueError("unexpected V3 checkpoint")
    if v3_fit_report["sha256"] != EXPECTED_V3_FIT_REPORT_SHA256:
        raise ValueError("unexpected V3 fit report")
    v5_checkpoint = _record(args.v5_checkpoint)
    v5_fit_report = _record(args.v5_fit_report)
    report = json.loads(args.v5_fit_report.read_text())
    required_fit = {
        "stage": "governance_consensus_router_fit",
        "method": "GRC-DGE-V5",
        "fit_complete": True,
        "fit_eligible": True,
        "checkpoint_sha256": v5_checkpoint["sha256"],
        "v3_checkpoint_sha256": v3_checkpoint["sha256"],
        "v3_fit_report_sha256": v3_fit_report["sha256"],
        "operator_dev_accessed": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for field, expected in required_fit.items():
        if report.get(field) != expected:
            raise ValueError(f"V5 fit report mismatch: {field}")
    if not report.get("gate_checks") or not all(report["gate_checks"].values()):
        raise ValueError("V5 fit gates did not all pass")

    model_manifest = _record(args.model_manifest)
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-consensus-v5-same-identity-code-v{code_version}",
        "created_utc": args.created_utc,
        "stage": "governance_v5_same_identity_behavior",
        "code_root": str(args.code_root),
        "immutable_code_bundle_manifest_sha256": sha256_file(bundle),
        "execution_allowed": True,
        "explicit_user_execution_authorization": True,
        "method_shard": "v5",
        "methods": list(METHOD_SHARDS["v5"]),
        "supplement_contract_sha256": sha256_file(args.behavior_contract),
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "design_audit_sha256": sha256_file(args.design_audit),
        "model_manifest_sha256": model_manifest["sha256"],
        "model_manifest": model_manifest,
        "expected_rows_per_method": EXPECTED_ROWS,
        "expected_key_sha256": EXPECTED_KEY_SHA256,
        "sources": {
            "v3_checkpoint": v3_checkpoint,
            "v3_fit_report": v3_fit_report,
            "v5_checkpoint": v5_checkpoint,
            "v5_fit_report": v5_fit_report,
        },
        "operator_dev_accessed_for_fit": False,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
