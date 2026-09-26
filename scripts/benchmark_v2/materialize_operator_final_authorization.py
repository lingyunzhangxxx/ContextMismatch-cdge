#!/usr/bin/env python3
"""Open the locked Qwen3-8B operator final test exactly once in a SHA-bound record."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key


SHA256_RE = re.compile(r"[0-9a-f]{64}")
REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _sha(value: str) -> str:
    if not SHA256_RE.fullmatch(value):
        raise argparse.ArgumentTypeError("invalid SHA256")
    return value


def _expected_key_hash(manifest: list[dict], contract: dict) -> tuple[int, str]:
    jobs = enumerate_jobs(manifest, contract, "final_test")
    if len(jobs) != 6144:
        raise ValueError(f"final-test design has {len(jobs)} rows instead of 6,144")
    keys = sorted(job_key(job) for job in jobs)
    return len(keys), hashlib.sha256((("\n".join(keys)) + "\n").encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-version", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True, type=_sha)
    parser.add_argument("--operator-lock", type=Path, required=True)
    parser.add_argument("--operator-lock-remote", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-contract", type=Path, required=True)
    parser.add_argument("--extension-contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path, required=True)
    parser.add_argument("--model-manifest-remote", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True, type=_sha)
    parser.add_argument("--created-utc")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.code_version < 18:
        raise ValueError("operator final-test code version must be at least 18")
    if args.output.exists():
        raise FileExistsError(f"refusing existing final authorization: {args.output}")
    if not str(args.model_manifest_remote).startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("model manifest is outside the owned project")
    if not str(args.operator_lock_remote).startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("operator lock is outside the owned project")

    lock = json.loads(args.operator_lock.read_text())
    extension = json.loads(args.extension_contract.read_text())
    contract = json.loads(args.crossover_contract.read_text())
    design = json.loads(args.design_audit.read_text())
    if lock.get("locked") is not True or lock.get("selection_partition") != "operator_dev only":
        raise ValueError("operator lock is not a completed operator-dev lock")
    if lock.get("final_test_open") is not False or lock.get("final_test_open_count") != 0:
        raise ValueError("operator lock indicates that final test was already opened")
    if lock.get("production_rollout_approved") is not False:
        raise ValueError("operator lock changes production boundary")
    if extension.get("status") != "frozen_before_operator_forward":
        raise ValueError("operator extension is not frozen before forward")
    if extension.get("production_rollout_approved") is not False:
        raise ValueError("extension changes production boundary")
    if lock.get("operator_contract_sha256") != sha256_file(args.operator_contract):
        raise ValueError("operator lock/contract binding mismatch")
    if lock.get("extension_contract_sha256") != sha256_file(args.extension_contract):
        raise ValueError("operator lock/extension binding mismatch")

    candidate_manifest = Path(lock["candidate_manifest"])
    candidate_tensor = Path(lock["candidate_tensor"])
    if not candidate_manifest.is_file() or sha256_file(candidate_manifest) != lock.get(
        "candidate_manifest_sha256"
    ):
        raise ValueError("locked local candidate manifest SHA mismatch")
    if not candidate_tensor.is_file() or sha256_file(candidate_tensor) != lock.get(
        "candidate_tensor_sha256"
    ):
        raise ValueError("locked local candidate tensor SHA mismatch")
    manifest_remote = lock.get("candidate_manifest_remote", "")
    tensor_remote = lock.get("candidate_tensor_remote", "")
    if not manifest_remote.startswith(f"{REMOTE_ROOT}/") or not tensor_remote.startswith(
        f"{REMOTE_ROOT}/"
    ):
        raise ValueError("locked candidate lacks owned remote artifact paths")

    contract_sha = sha256_file(args.crossover_contract)
    manifest_sha = sha256_file(args.manifest)
    if design.get("stage") != "final_test" or design.get("audit", {}).get("success") is not True:
        raise ValueError("final-test design audit did not pass")
    if design.get("contract_sha256") != contract_sha:
        raise ValueError("final-test design/contract binding mismatch")
    if design.get("manifest_sha256") != manifest_sha:
        raise ValueError("final-test design/manifest binding mismatch")
    expected_rows, expected_key_sha = _expected_key_hash(load_jsonl(args.manifest), contract)
    if design["audit"].get("row_count") != expected_rows:
        raise ValueError("final-test design row count mismatch")
    if design["audit"].get("expected_key_sha256") != expected_key_sha:
        raise ValueError("final-test design key hash mismatch")

    created_utc = args.created_utc or (
        dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", created_utc):
        raise ValueError("created UTC must use YYYY-MM-DDTHH:MM:SSZ")
    value = {
        "schema_version": 1,
        "authorization_id": f"context-mismatch-qwen3-8b-operator-final-test-code-v{args.code_version}",
        "created_utc": created_utc,
        "stage": "operator_final_test",
        "code_root": f"{REMOTE_ROOT}/code-v{args.code_version}",
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "execution_allowed": True,
        "production_rollout_approved": False,
        "final_test_open": True,
        "final_test_open_count": 1,
        "operator_lock_sha256": sha256_file(args.operator_lock),
        "operator_lock": {
            "path": str(args.operator_lock_remote),
            "sha256": sha256_file(args.operator_lock),
        },
        "candidate_id": lock["chosen_candidate_id"],
        "candidate_manifest": {
            "path": manifest_remote,
            "sha256": lock["candidate_manifest_sha256"],
        },
        "candidate_tensor": {
            "path": tensor_remote,
            "sha256": lock["candidate_tensor_sha256"],
        },
        "candidate_manifest_sha256": lock["candidate_manifest_sha256"],
        "candidate_tensor_sha256": lock["candidate_tensor_sha256"],
        "crossover_contract_sha256": contract_sha,
        "operator_contract_sha256": sha256_file(args.operator_contract),
        "extension_contract_sha256": sha256_file(args.extension_contract),
        "benchmark_manifest_sha256": manifest_sha,
        "design_audit_sha256": sha256_file(args.design_audit),
        "expected_key_sha256": expected_key_sha,
        "expected_rows": expected_rows,
        "model_manifest": {
            "path": str(args.model_manifest_remote),
            "sha256": args.model_manifest_sha256,
        },
        "notes": "One-time final-test opening after SHA-bound operator lock; production rollout remains closed.",
    }
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
