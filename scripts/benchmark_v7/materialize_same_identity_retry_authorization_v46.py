#!/usr/bin/env python3
"""Bind a fresh code-v40 authorization to a verified runtime-restore retry lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


PROJECT_ROOT = Path("/workspace/context-mismatch-qwen3-8b")
CODE_ROOT = PROJECT_ROOT / "code-v40"
CONTROL_ROOT = PROJECT_ROOT / "code-v46"
EXPECTED_FAILED = {
    "operators": ("9164", "qwen3-8b-dge-same-identity-operators-20260729T090137Z", "bcd51802996cacb5ee20190859434ec5ae0e4b43b426234176bf328cd00dece3"),
    "experts": ("9165", "qwen3-8b-dge-same-identity-experts-20260729T090137Z", "42420424ad504f445dd38f07c575b92c4b1c22bff8322bfd44be5a0b86c08f04"),
    "routing": ("9166", "qwen3-8b-dge-same-identity-routing-20260729T090137Z", "c6d8211dda046f064212f30f772eab0039f39f4820c80478a9cbb4f63821f1ea"),
}
EXPECTED_RESTORE_RUN = "qwen3-8b-runtime-restore-20260729T091648Z"
EXPECTED_RESTORE_JOB = "9169"
EXPECTED_RESTORE_ARCHIVE_SHA256 = "e7c33a1613532f69ec02aca0813ad5c30423907b226af573e01654f91f206247"
MISSING_RUNTIME = "missing runtime prerequisite: /workspace/node-local/context-mismatch-ascend/runtime-v1/python-3.11.13-standalone/python/bin/python3\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def record(path: Path) -> dict[str, str]:
    path = path.resolve(strict=True)
    if not str(path).startswith(f"{PROJECT_ROOT}/"):
        raise ValueError("artifact escaped project root")
    return {"path": str(path), "sha256": sha256_file(path)}


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-authorization", type=Path, required=True)
    parser.add_argument("--materializer-root", type=Path, required=True)
    parser.add_argument("--failed-run-dir", type=Path, required=True)
    parser.add_argument("--failed-authorization", type=Path, required=True)
    parser.add_argument("--runtime-restore-run-dir", type=Path, required=True)
    parser.add_argument("--runtime-restore-evidence-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if args.materializer_root != CONTROL_ROOT:
        raise ValueError("retry materialization requires immutable code-v46")
    if not (CONTROL_ROOT / "bundle.sha256").is_file():
        raise FileNotFoundError("code-v46 bundle manifest missing")

    base = json.loads(args.base_authorization.read_text())
    shard = base.get("method_shard")
    if shard not in EXPECTED_FAILED:
        raise ValueError("invalid method shard")
    failed_job, failed_run, failed_auth_sha = EXPECTED_FAILED[shard]
    if args.failed_run_dir != PROJECT_ROOT / "runs" / failed_run:
        raise ValueError("failed run lineage mismatch")
    if sha256_file(args.failed_authorization) != failed_auth_sha:
        raise ValueError("failed authorization SHA mismatch")
    failed_status = json.loads((args.failed_run_dir / "exit_status.json").read_text())
    failed_submission = json.loads((args.failed_run_dir / "submission.json").read_text())
    failed_log = args.failed_run_dir / f"slurm-{failed_job}.out"
    if failed_status.get("job_id") != failed_job or failed_status.get("exit_code") != 2:
        raise ValueError("failed job-local status mismatch")
    if failed_status.get("hostname") != "a06" or failed_submission.get("method_shard") != shard:
        raise ValueError("failed allocation/shard mismatch")
    if (args.failed_run_dir / "COMPLETE").exists() or failed_log.read_text() != MISSING_RUNTIME:
        raise ValueError("failed attempt was not the bounded runtime-prerequisite failure")

    if args.runtime_restore_run_dir != PROJECT_ROOT / "runs" / EXPECTED_RESTORE_RUN:
        raise ValueError("runtime restore run mismatch")
    restore_status = json.loads((args.runtime_restore_run_dir / "exit_status.json").read_text())
    restore_audit = json.loads((args.runtime_restore_run_dir / "runtime_restore_audit.json").read_text())
    if restore_status.get("job_id") != EXPECTED_RESTORE_JOB or restore_status.get("exit_code") != 0:
        raise ValueError("runtime restore terminal status mismatch")
    if restore_status.get("hostname") != "a06" or restore_audit.get("success") is not True:
        raise ValueError("runtime restore audit mismatch")
    if not (args.runtime_restore_run_dir / "COMPLETE").is_file():
        raise ValueError("runtime restore COMPLETE marker missing")
    if sha256_file(args.runtime_restore_evidence_archive) != EXPECTED_RESTORE_ARCHIVE_SHA256:
        raise ValueError("runtime restore evidence archive SHA mismatch")

    required_base = {
        "stage": "dge_same_identity_supplement",
        "code_root": str(CODE_ROOT),
        "execution_node": "a06",
        "method_shard": shard,
        "expected_rows_per_method": 3072,
        "expected_key_sha256": "74fb3f84b166068a04cb9f0d61ad331651793510f6688000c54dc1d51d66a12e",
        "orchestration_only_migration": True,
        "scientific_identity_unchanged": True,
        "execution_allowed": True,
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    for key, value in required_base.items():
        if base.get(key) != value:
            raise ValueError(f"base authorization mismatch: {key}")

    value = dict(base)
    value.update(
        {
            "schema_version": 3,
            "authorization_id": f"qwen3-8b-dge-same-identity-{shard}-code-v40-runtime-retry",
            "authorization_retry_materializer_root": str(CONTROL_ROOT),
            "authorization_retry_materializer_bundle_manifest_sha256": sha256_file(CONTROL_ROOT / "bundle.sha256"),
            "authorization_retry_materializer": record(Path(__file__)),
            "parent_authorization": record(args.base_authorization),
            "retry_after_verified_runtime_restore": True,
            "failed_attempt": {
                "job_id": int(failed_job),
                "run_id": failed_run,
                "run_exit_status": record(args.failed_run_dir / "exit_status.json"),
                "run_submission": record(args.failed_run_dir / "submission.json"),
                "run_log": record(failed_log),
                "authorization": record(args.failed_authorization),
                "scientific_forward_executed": False,
                "rows_written": 0,
            },
            "runtime_restore": {
                "job_id": int(EXPECTED_RESTORE_JOB),
                "run_id": EXPECTED_RESTORE_RUN,
                "exit_status": record(args.runtime_restore_run_dir / "exit_status.json"),
                "audit": record(args.runtime_restore_run_dir / "runtime_restore_audit.json"),
                "evidence_archive": record(args.runtime_restore_evidence_archive),
                "slurm_completed_verified": True,
            },
            "execution_allowed": True,
            "explicit_user_execution_authorization": True,
            "operator_dev_accessed_for_fit": False,
            "final_test_open": False,
            "final_test_open_count": 0,
            "production_rollout_approved": False,
        }
    )
    atomic_write(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
