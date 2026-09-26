#!/usr/bin/env python3
"""Create immutable SHA-bound AMSGE execution authorizations."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from scripts.benchmark_v1.common import atomic_write_text, load_jsonl, sha256_file
from scripts.benchmark_v1.controls import load_controls
from scripts.benchmark_v2.crossover import enumerate_jobs, job_key
from scripts.benchmark_v3.governance_control_design import (
    governance_control_groups,
    governance_control_key_hash,
    operator_selection_items,
)


REMOTE_ROOT = "/workspace/context-mismatch-qwen3-8b"


def _key_hash(jobs: list[dict]) -> str:
    return hashlib.sha256(
        ("\n".join(sorted(job_key(job) for job in jobs)) + "\n").encode("utf-8")
    ).hexdigest()


def _smoke_jobs(jobs: list[dict]) -> list[dict]:
    first = {}
    for job in sorted(jobs, key=lambda row: (row["item"]["benchmark"], row["item"]["item_id"])):
        first.setdefault(job["item"]["benchmark"], job["item"]["item_id"])
    selected = [
        job for job in jobs if first[job["item"]["benchmark"]] == job["item"]["item_id"]
    ]
    if len(selected) != 192:
        raise ValueError("fit/capture smoke must contain 192 rows")
    return selected


def _selection_jobs(jobs: list[dict]) -> list[dict]:
    chosen = set()
    by_benchmark = defaultdict(set)
    for job in jobs:
        by_benchmark[job["item"]["benchmark"]].add(job["item"]["item_id"])
    for benchmark, item_ids in sorted(by_benchmark.items()):
        ordered = sorted(item_ids)
        if len(ordered) != 32:
            raise ValueError(f"operator_dev {benchmark} does not contain 32 items")
        chosen.update((benchmark, item_id) for item_id in ordered[16:])
    selected = [
        job
        for job in jobs
        if (job["item"]["benchmark"], job["item"]["item_id"]) in chosen
    ]
    if len(selected) != 3072:
        raise ValueError("operator_dev selection must contain 3,072 rows")
    return selected


def _artifact(path: Path, remote: str | None) -> dict:
    if not remote or not remote.startswith(f"{REMOTE_ROOT}/"):
        raise ValueError("artifact remote path must be inside the Qwen3-8B project root")
    return {"path": remote, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=[
            "governance_capture_smoke",
            "governance_capture_full",
            "governance_fit",
            "governance_behavior_fit_smoke",
            "governance_behavior_selection",
            "governance_controls_selection",
            "governance_failed_fit_smoke",
            "governance_failed_fit_selection",
            "governance_failed_fit_controls",
        ],
        required=True,
    )
    parser.add_argument("--code-version", type=int, required=True)
    parser.add_argument("--bundle-manifest-sha256", required=True)
    parser.add_argument("--editor-contract", type=Path, required=True)
    parser.add_argument("--crossover-contract", type=Path, required=True)
    parser.add_argument("--operator-site-manifest", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--design-audit", type=Path)
    parser.add_argument("--model-manifest-remote")
    parser.add_argument("--model-manifest-sha256")
    parser.add_argument("--capture-manifest", type=Path)
    parser.add_argument("--capture-manifest-remote")
    parser.add_argument("--protected-capture-manifest", type=Path)
    parser.add_argument("--protected-capture-manifest-remote")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-remote")
    parser.add_argument("--fit-report", type=Path)
    parser.add_argument("--fit-report-remote")
    parser.add_argument("--identity-report", type=Path)
    parser.add_argument("--identity-report-remote")
    parser.add_argument("--controls", type=Path)
    parser.add_argument("--diagnostic-contract", type=Path)
    parser.add_argument("--created-utc", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing existing authorization: {args.output}")
    if len(args.bundle_manifest_sha256) != 64:
        raise ValueError("bundle manifest SHA must be lowercase SHA256 hex")
    int(args.bundle_manifest_sha256, 16)
    contract = json.loads(args.editor_contract.read_text())
    if contract.get("status") not in {
        "frozen_before_any_code_v22_forward",
        "frozen_before_any_v1_post_failure_operator_dev_forward",
    }:
        raise ValueError("editor contract is not frozen")
    minimum_code_version = int(contract.get("code_version_minimum", 22))
    if args.code_version < minimum_code_version:
        raise ValueError(
            f"adaptive governance authorization requires code-v{minimum_code_version} or newer"
        )
    code_root = f"{REMOTE_ROOT}/code-v{args.code_version}"
    value = {
        "schema_version": 1,
        "authorization_id": f"qwen3-8b-{args.stage}-code-v{args.code_version}",
        "created_utc": args.created_utc,
        "stage": args.stage,
        "code_root": code_root,
        "immutable_code_bundle_manifest_sha256": args.bundle_manifest_sha256,
        "execution_allowed": True,
        "editor_contract_sha256": sha256_file(args.editor_contract),
        "editor_contract_filename": args.editor_contract.name,
        "crossover_contract_sha256": sha256_file(args.crossover_contract),
        "operator_site_manifest_sha256": sha256_file(args.operator_site_manifest),
        "benchmark_manifest_sha256": sha256_file(args.manifest),
        "final_test_open": False,
        "final_test_open_count": 0,
        "production_rollout_approved": False,
    }
    if args.stage.startswith("governance_capture_"):
        if args.design_audit is None or not args.model_manifest_remote or not args.model_manifest_sha256:
            raise ValueError("capture authorization requires design audit and model manifest")
        jobs = enumerate_jobs(load_jsonl(args.manifest), contract=json.loads(args.crossover_contract.read_text()), stage_name="replication")
        if args.stage.endswith("smoke"):
            jobs = _smoke_jobs(jobs)
        value.update(
            {
                "expected_rows": len(jobs),
                "expected_key_sha256": _key_hash(jobs),
                "design_audit_sha256": sha256_file(args.design_audit),
                "model_manifest": {
                    "path": args.model_manifest_remote,
                    "sha256": args.model_manifest_sha256,
                },
            }
        )
    elif args.stage == "governance_fit":
        if args.capture_manifest is None or args.protected_capture_manifest is None:
            raise ValueError("fit authorization requires both capture manifests")
        value.update(
            {
                "capture_manifest": _artifact(
                    args.capture_manifest, args.capture_manifest_remote
                ),
                "capture_manifest_sha256": sha256_file(args.capture_manifest),
                "protected_capture_manifest": _artifact(
                    args.protected_capture_manifest,
                    args.protected_capture_manifest_remote,
                ),
                "protected_capture_manifest_sha256": sha256_file(
                    args.protected_capture_manifest
                ),
            }
        )
    else:
        if (
            args.checkpoint is None
            or args.fit_report is None
            or args.design_audit is None
            or not args.model_manifest_remote
            or not args.model_manifest_sha256
        ):
            raise ValueError("behavior authorization requires checkpoint, report, audit, and model")
        diagnostic = args.stage.startswith("governance_failed_fit_")
        controls_stage = args.stage in {
            "governance_controls_selection",
            "governance_failed_fit_controls",
        }
        fit_value = json.loads(args.fit_report.read_text())
        if diagnostic:
            if args.diagnostic_contract is None:
                raise ValueError("failed-fit authorization requires diagnostic contract")
            diagnostic_contract = json.loads(args.diagnostic_contract.read_text())
            if diagnostic_contract.get("post_failure_characterization") is not True:
                raise ValueError("invalid post-failure diagnostic contract")
            if fit_value.get("all_fit_gates_pass") is not False:
                raise ValueError("failed-fit authorization requires failed fit report")
            value.update(
                {
                    "diagnostic_contract_sha256": sha256_file(args.diagnostic_contract),
                    "diagnostic_contract_filename": args.diagnostic_contract.name,
                    "post_failure_characterization": True,
                    "fit_gates_passed": False,
                    "candidate_eligible": False,
                    "candidate_may_be_locked": False,
                }
            )
        elif fit_value.get("all_fit_gates_pass") is not True:
            raise ValueError("ordinary behavior authorization requires passed fit gates")
        if controls_stage:
            if args.identity_report is None or args.controls is None:
                raise ValueError("controls authorization requires identity report and controls")
            items = operator_selection_items(load_jsonl(args.manifest))
            crossover = json.loads(args.crossover_contract.read_text())
            groups = governance_control_groups(
                items,
                load_controls(args.controls),
                list(crossover["factorial"]["declared_roles"]),
                list(crossover["factorial"]["history_styles"]),
            )
            expected_rows = 2856
            expected_key = governance_control_key_hash(groups)
        elif args.stage.endswith("fit_smoke"):
            jobs = _smoke_jobs(
                enumerate_jobs(
                    load_jsonl(args.manifest),
                    json.loads(args.crossover_contract.read_text()),
                    "replication",
                )
            )
            expected_rows = len(jobs)
            expected_key = _key_hash(jobs)
        elif args.stage.endswith("failed_fit_smoke"):
            jobs = _smoke_jobs(
                enumerate_jobs(
                    load_jsonl(args.manifest),
                    json.loads(args.crossover_contract.read_text()),
                    "replication",
                )
            )
            expected_rows = len(jobs)
            expected_key = _key_hash(jobs)
        else:
            jobs = _selection_jobs(
                enumerate_jobs(
                    load_jsonl(args.manifest),
                    json.loads(args.crossover_contract.read_text()),
                    "operator_dev",
                )
            )
            expected_rows = len(jobs)
            expected_key = _key_hash(jobs)
        value.update(
            {
                "checkpoint": _artifact(args.checkpoint, args.checkpoint_remote),
                "checkpoint_sha256": sha256_file(args.checkpoint),
                "fit_report": _artifact(args.fit_report, args.fit_report_remote),
                "fit_report_sha256": sha256_file(args.fit_report),
                "design_audit_sha256": sha256_file(args.design_audit),
                "expected_rows": expected_rows,
                "expected_key_sha256": expected_key,
                "model_manifest": {
                    "path": args.model_manifest_remote,
                    "sha256": args.model_manifest_sha256,
                },
            }
        )
        if controls_stage:
            value.update(
                {
                    "identity_report": _artifact(
                        args.identity_report, args.identity_report_remote
                    ),
                    "identity_report_sha256": sha256_file(args.identity_report),
                    "controls_sha256": sha256_file(args.controls),
                }
            )
    atomic_write_text(args.output, json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
